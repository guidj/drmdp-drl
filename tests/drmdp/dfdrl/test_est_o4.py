"""Tests for O4 reward estimation with learned input mask.

Critical tests focus on:
1. InputMask: all three strategies (sigmoid, ste, gumbel), binary_mask(), invalid type
2. RNetwork: output shape unchanged, gradient flows to mask logits
3. Sparsity loss: mean(p_active) ∈ (0,1), minimising pushes p_active → 0
4. Training integration: on_batch_end receives "sparsity" key, mask logged
5. DictDataset and collation (shared with est_o2 — verifies base is intact)
6. Episode boundary constraint
"""

import json
from typing import Any, List, Mapping

import gymnasium as gym
import numpy as np
import pytest
import torch

from drmdp import rewdelay
from drmdp.dfdrl import est_o4

# =============================================================================
# TestInputMask
# =============================================================================


class TestInputMask:
    """Tests for the InputMask module."""

    def test_sigmoid_output_range(self):
        """sigmoid mask values should be strictly in (0, 1)."""
        mask = est_o4.InputMask(input_dim=6, mask_type="sigmoid")
        values = mask()
        assert values.shape == (6,)
        assert (values > 0).all()
        assert (values < 1).all()

    def test_sigmoid_gradient_flows(self):
        """sigmoid mask should have non-zero gradient w.r.t. logits."""
        mask = est_o4.InputMask(input_dim=4, mask_type="sigmoid")
        dummy_input = torch.ones(4)
        loss = (dummy_input * mask()).sum()
        loss.backward()
        assert mask.logits.grad is not None
        assert (mask.logits.grad != 0).all()

    def test_ste_forward_is_binary(self):
        """STE mask must be exactly 0 or 1 in the forward pass."""
        torch.manual_seed(0)
        mask = est_o4.InputMask(input_dim=8, mask_type="ste")
        # Initialise with mixed logits — each row is [inactive_logit, active_logit]
        with torch.no_grad():
            mask.logits.copy_(torch.randn(8, 2))
        values = mask()
        assert values.shape == (8,)
        assert set(values.tolist()).issubset({0.0, 1.0})

    def test_ste_gradient_flows(self):
        """STE gradient must flow through the sigmoid approximation."""
        mask = est_o4.InputMask(input_dim=4, mask_type="ste")
        dummy_input = torch.ones(4)
        loss = (dummy_input * mask()).sum()
        loss.backward()
        assert mask.logits.grad is not None
        assert (mask.logits.grad != 0).all()

    def test_gumbel_forward_is_binary_during_training(self):
        """Gumbel mask values must be exactly 0 or 1 in the forward pass (hard=True)."""
        mask = est_o4.InputMask(input_dim=8, mask_type="gumbel")
        mask.train()
        for _ in range(5):  # Multiple passes to account for stochastic noise
            values = mask()
            assert set(values.tolist()).issubset({0.0, 1.0}), (
                f"Expected binary values, got {values.tolist()}"
            )

    def test_gumbel_gradient_flows(self):
        """Gumbel mask gradient must flow (STE inside F.gumbel_softmax)."""
        mask = est_o4.InputMask(input_dim=4, mask_type="gumbel")
        mask.train()
        dummy_input = torch.ones(4)
        loss = (dummy_input * mask()).sum()
        loss.backward()
        assert mask.logits.grad is not None

    def test_gumbel_eval_is_binary_threshold(self):
        """At eval time gumbel mask must equal argmax(logits, dim=-1)."""
        mask = est_o4.InputMask(input_dim=6, mask_type="gumbel")
        # Rows where active logit (col 1) > inactive (col 0) → expected active (1)
        # Rows where active logit (col 1) < inactive (col 0) → expected inactive (0)
        with torch.no_grad():
            mask.logits.copy_(
                torch.tensor(
                    [
                        [-1.0, 1.0],
                        [1.0, -1.0],
                        [-0.5, 0.5],
                        [0.5, -0.5],
                        [2.0, 0.5],
                        [-2.0, 2.0],
                    ]
                )
            )
        mask.eval()
        values = mask()
        expected = mask.logits.argmax(dim=-1).float()
        assert torch.equal(values, expected)

    def test_binary_mask_always_zero_or_one(self):
        """binary_mask() must return exactly 0/1 regardless of mask_type."""
        for mask_type in ("sigmoid", "ste", "gumbel"):
            mask = est_o4.InputMask(input_dim=6, mask_type=mask_type)
            with torch.no_grad():
                mask.logits.copy_(torch.randn(6, 2))
            bm = mask.binary_mask()
            assert bm.shape == (6,)
            assert set(bm.tolist()).issubset({0.0, 1.0}), (
                f"binary_mask() returned non-binary values for {mask_type}"
            )

    def test_invalid_mask_type_raises(self):
        """An unknown mask_type should raise ValueError in forward()."""
        mask = est_o4.InputMask(input_dim=4, mask_type="invalid")
        with pytest.raises(ValueError, match="Unknown mask_type"):
            mask()

    def test_sigmoid_returns_softmax_active_logit(self):
        """Sigmoid mask value equals softmax([inactive, active])[1] for each dim."""
        mask = est_o4.InputMask(input_dim=4, mask_type="sigmoid")
        with torch.no_grad():
            mask.logits[:, 0] = 0.0
            mask.logits[:, 1] = 1.0
        values = mask()
        expected = torch.nn.functional.softmax(torch.tensor([0.0, 1.0]), dim=0)[1].item()
        np.testing.assert_allclose(values.detach().numpy(), expected, atol=1e-6)
        assert mask.binary_mask().sum().item() == 4  # active > inactive → all ones

    def test_positive_logits_increase_active_fraction(self):
        """Pushing the active logit column positive should increase the active fraction."""
        mask = est_o4.InputMask(input_dim=10, mask_type="ste")
        with torch.no_grad():
            mask.logits.fill_(0.0)
        baseline = mask.binary_mask().sum().item()

        with torch.no_grad():
            mask.logits[:, 1] = 5.0  # active logit >> inactive logit for all dims
        high = mask.binary_mask().sum().item()
        assert high >= baseline


# =============================================================================
# TestRNetworkWithMask
# =============================================================================


class TestRNetworkWithMask:
    """Tests for RNetwork with the integrated InputMask."""

    def test_output_shape(self):
        """Output shape must match (batch, seq_len, 1) — same as est_o2."""
        model = est_o4.RNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        for batch_size, seq_len in [(1, 3), (4, 5), (8, 10)]:
            state = torch.randn(batch_size, seq_len, 4)
            action = torch.randn(batch_size, seq_len, 2)
            term = torch.zeros(batch_size, seq_len, 1)
            output = model(state, action, term)
            assert output.shape == (batch_size, seq_len, 1)

    def test_mask_logits_receive_gradient(self):
        """A reward loss backward pass must produce a non-zero gradient on mask logits."""
        model = est_o4.RNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        state = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        term = torch.zeros(2, 3, 1)
        output = model(state, action, term)
        loss = output.sum()
        loss.backward()
        assert model.input_mask.logits.grad is not None
        assert model.input_mask.logits.grad.abs().sum() > 0

    def test_mask_type_propagated(self):
        """mask_type passed to RNetwork must appear in the InputMask."""
        for mask_type in ("sigmoid", "ste", "gumbel"):
            model = est_o4.RNetwork(
                state_dim=3, action_dim=1, hidden_dim=16, mask_type=mask_type
            )
            assert model.input_mask.mask_type == mask_type

    def test_fully_masked_output_is_zero_effect(self):
        """When all logits are very negative (all masked), output depends only on
        the zero vector regardless of state, action, or term values."""
        model = est_o4.RNetwork(
            state_dim=4, action_dim=2, hidden_dim=16, mask_type="ste"
        )
        with torch.no_grad():
            model.input_mask.logits.fill_(-100.0)  # all masked out

        term = torch.zeros(2, 3, 1)
        out1 = model(torch.randn(2, 3, 4), torch.randn(2, 3, 2), term)
        out2 = model(torch.randn(2, 3, 4), torch.randn(2, 3, 2), torch.ones(2, 3, 1))
        assert torch.allclose(out1, out2, atol=1e-5), (
            "Fully masked model should ignore all inputs"
        )

    def test_input_mask_dimension(self):
        """InputMask logits must have shape (state_dim + action_dim + 1, 2)."""
        state_dim, action_dim = 5, 3
        model = est_o4.RNetwork(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=16
        )
        assert model.input_mask.logits.shape == (state_dim + action_dim + 1, 2)


# =============================================================================
# TestSparsityLoss
# =============================================================================


class TestSparsityLoss:
    """Tests for the sparsity term mean(softmax(φ)[:, 1]) = mean(p_active).

    Minimising this term pushes p_active → 0 (sparse mask).
    The loss is bounded in (0, 1), compatible with mask_lam ≈ 0.1.
    """

    def test_sparsity_loss_in_unit_interval(self):
        """mean(p_active) must lie in (0, 1) for any logit values."""
        for seed in range(5):
            torch.manual_seed(seed)
            model = est_o4.RNetwork(state_dim=4, action_dim=2, hidden_dim=16)
            with torch.no_grad():
                model.input_mask.logits.copy_(
                    torch.randn_like(model.input_mask.logits) * 5
                )
            p_active = torch.nn.functional.softmax(model.input_mask.logits, dim=-1)[
                :, 1
            ]
            sparsity = p_active.mean()
            assert 0 < sparsity.item() < 1

    def test_sparsity_loss_equals_mean_softmax_active_logit(self):
        """p_active = mean(softmax(logits, dim=-1)[:, 1]) when logits are [0.0, 1.0]."""
        model = est_o4.RNetwork(state_dim=4, action_dim=2, hidden_dim=16)
        with torch.no_grad():
            model.input_mask.logits[:, 0] = 0.0
            model.input_mask.logits[:, 1] = 1.0
        p_active = torch.nn.functional.softmax(model.input_mask.logits, dim=-1)[:, 1]
        sparsity = p_active.mean().item()
        expected = torch.nn.functional.softmax(torch.tensor([0.0, 1.0]), dim=0)[1].item()
        np.testing.assert_allclose(sparsity, expected, atol=1e-6)

    def test_low_active_logit_decreases_sparsity_loss(self):
        """Setting active logit below inactive logit should reduce p_active and
        lower the sparsity loss (more sparse)."""
        model = est_o4.RNetwork(state_dim=4, action_dim=2, hidden_dim=16)
        with torch.no_grad():
            model.input_mask.logits.fill_(0.0)
        p_active_base = torch.nn.functional.softmax(model.input_mask.logits, dim=-1)[
            :, 1
        ]
        baseline = p_active_base.mean().item()

        with torch.no_grad():
            model.input_mask.logits[:, 1] = -5.0  # active logit well below inactive
        p_active_low = torch.nn.functional.softmax(model.input_mask.logits, dim=-1)[
            :, 1
        ]
        low = p_active_low.mean().item()

        assert low < baseline

    def test_sparsity_gradient_reduces_active_logit(self):
        """Gradient of mean(p_active) w.r.t. active logit column must be positive
        so gradient descent pushes the active logit down (reducing p_active)."""
        model = est_o4.RNetwork(state_dim=4, action_dim=2, hidden_dim=16)
        p_active = torch.nn.functional.softmax(model.input_mask.logits, dim=-1)[:, 1]
        sparsity_loss = p_active.mean()
        sparsity_loss.backward()
        # Positive gradient on active column → gradient descent reduces active logit
        assert (model.input_mask.logits.grad[:, 1] > 0).all()
        # Negative gradient on inactive column → gradient descent increases inactive logit
        assert (model.input_mask.logits.grad[:, 0] < 0).all()


# =============================================================================
# TestDictDataset
# =============================================================================


class TestDictDataset:
    """Tests for DictDataset."""

    def test_length(self):
        """__len__ returns the number of examples."""
        inputs = [{"state": torch.zeros(3, 2)} for _ in range(5)]
        labels = [{"aggregate_reward": torch.tensor(1.0)} for _ in range(5)]
        ds = est_o4.DictDataset(inputs, labels)
        assert len(ds) == 5

    def test_getitem(self):
        """__getitem__ returns the correct (input, label) pair."""
        inputs = [{"idx": torch.tensor(idx)} for idx in range(3)]
        labels = [{"val": torch.tensor(idx * 2)} for idx in range(3)]
        ds = est_o4.DictDataset(inputs, labels)
        inp, lbl = ds[1]
        assert inp["idx"].item() == 1
        assert lbl["val"].item() == 2


# =============================================================================
# TestCollateVariableLengthSequences
# =============================================================================


class TestCollateVariableLengthSequences:
    """Tests for the custom collate function."""

    def _make_example(self, seq_len: int, state_dim: int = 3, action_dim: int = 2):
        inputs = {
            "state": torch.randn(seq_len, state_dim),
            "action": torch.randn(seq_len, action_dim),
            "term": torch.zeros(seq_len, 1),
        }
        labels = {
            "aggregate_reward": torch.tensor(1.0),
            "per_step_rewards": torch.ones(seq_len),
            "start_return": torch.tensor(0.0),
            "end_return": torch.tensor(float(seq_len)),
        }
        return inputs, labels

    def test_padded_to_max_len(self):
        """Sequences must be zero-padded to the maximum length in the batch."""
        batch = [self._make_example(seq_len) for seq_len in (2, 4, 3)]
        batched_inputs, _ = est_o4.collate_variable_length_sequences(batch)
        assert batched_inputs["state"].shape == (3, 4, 3)
        assert batched_inputs["action"].shape == (3, 4, 2)

    def test_padding_is_zero(self):
        """Padded positions must be filled with zeros."""
        batch = [self._make_example(2), self._make_example(4)]
        batched_inputs, _ = est_o4.collate_variable_length_sequences(batch)
        # First example was length 2, positions 2 and 3 must be zero
        assert torch.equal(batched_inputs["state"][0, 2:], torch.zeros(2, 3))

    def test_labels_stacked(self):
        """Scalar labels must be stacked into 1-D tensors of shape (batch,)."""
        batch = [self._make_example(seq_len) for seq_len in (2, 3)]
        _, batched_labels = est_o4.collate_variable_length_sequences(batch)
        assert batched_labels["aggregate_reward"].shape == (2,)
        assert batched_labels["start_return"].shape == (2,)
        assert batched_labels["end_return"].shape == (2,)


# =============================================================================
# TestDelayedRewardData
# =============================================================================


class TestDelayedRewardData:
    """Tests for delayed_reward_data()."""

    def _make_buffer(self, num_steps: int = 20):
        """Simple synthetic buffer: constant reward 1.0, no terminals."""
        np.random.seed(0)
        state_dim, action_dim = 3, 2
        buffer = []
        for _ in range(num_steps):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            reward = 1.0
            term = False
            buffer.append((state, action, next_state, reward, term))
        return buffer

    def test_returns_list_of_examples(self):
        """delayed_reward_data must return a list of (inputs, labels) tuples."""
        buffer = self._make_buffer(30)
        delay = rewdelay.FixedDelay(3)
        examples = est_o4.delayed_reward_data(buffer, delay)
        assert isinstance(examples, list)
        assert len(examples) > 0
        inp, lbl = examples[0]
        assert "state" in inp
        assert "aggregate_reward" in lbl
        assert "start_return" in lbl
        assert "end_return" in lbl

    def test_aggregate_equals_sum_of_per_step(self):
        """aggregate_reward must equal the sum of per_step_rewards."""
        buffer = self._make_buffer(30)
        delay = rewdelay.FixedDelay(3)
        examples = est_o4.delayed_reward_data(buffer, delay)
        for _, lbl in examples:
            agg = lbl["aggregate_reward"].item()
            per_step_sum = lbl["per_step_rewards"].sum().item()
            np.testing.assert_allclose(agg, per_step_sum, atol=1e-5)

    def test_return_consistency(self):
        """end_return - start_return must equal aggregate_reward for every window."""
        buffer = self._make_buffer(50)
        delay = rewdelay.FixedDelay(4)
        examples = est_o4.delayed_reward_data(buffer, delay)
        for _, lbl in examples:
            delta = lbl["end_return"].item() - lbl["start_return"].item()
            agg = lbl["aggregate_reward"].item()
            np.testing.assert_allclose(delta, agg, atol=1e-5)

    def test_empty_buffer_returns_empty(self):
        """An empty buffer must return an empty list."""
        examples = est_o4.delayed_reward_data([], rewdelay.FixedDelay(3))
        assert examples == []

    def test_episode_boundary_no_span(self):
        """Windows must never span episode boundaries (terminal at position T)."""
        state_dim, action_dim = 2, 1
        # Episode 1: 5 steps, then terminal
        buffer = []
        for step_idx in range(5):
            buffer.append(
                (
                    np.zeros(state_dim, dtype=np.float32),
                    np.zeros(action_dim, dtype=np.float32),
                    np.zeros(state_dim, dtype=np.float32),
                    1.0,
                    step_idx == 4,  # terminal on last step of episode
                )
            )
        # Episode 2: 5 more steps
        for _ in range(5):
            buffer.append(
                (
                    np.ones(state_dim, dtype=np.float32),
                    np.ones(action_dim, dtype=np.float32),
                    np.ones(state_dim, dtype=np.float32),
                    2.0,
                    False,
                )
            )

        delay = rewdelay.FixedDelay(3)
        examples = est_o4.delayed_reward_data(buffer, delay)
        for _, lbl in examples:
            # All rewards within an episode should be the same value (1.0 or 2.0)
            per_step = lbl["per_step_rewards"].tolist()
            assert len(set(per_step)) == 1, f"Window spans episode boundary: {per_step}"


# =============================================================================
# TestTrainWithMask
# =============================================================================


class TestTrainWithMask:
    """Integration tests for the train() function with mask training."""

    def _make_dataset(
        self,
        num_examples: int = 30,
        seq_len: int = 3,
        state_dim: int = 2,
        action_dim: int = 1,
    ) -> est_o4.DictDataset:
        """Build a small synthetic dataset."""
        inputs_list = []
        labels_list = []
        for _ in range(num_examples):
            inputs = {
                "state": torch.randn(seq_len, state_dim),
                "action": torch.randn(seq_len, action_dim),
                "term": torch.zeros(seq_len, 1),
            }
            agg = float(np.random.randn())
            labels = {
                "aggregate_reward": torch.tensor(agg),
                "per_step_rewards": torch.tensor([agg / seq_len] * seq_len),
                "start_return": torch.tensor(0.0),
                "end_return": torch.tensor(agg),
            }
            inputs_list.append(inputs)
            labels_list.append(labels)
        return est_o4.DictDataset(inputs_list, labels_list)

    @pytest.mark.parametrize("mask_type", ["sigmoid", "ste", "gumbel"])
    def test_train_runs_without_error(self, mask_type: str, tmp_path):
        """train() must complete without raising for all three mask types."""
        torch.manual_seed(42)
        env = gym.make("MountainCarContinuous-v0")
        ds = self._make_dataset()
        est_o4.train(
            env=env,
            dataset=ds,
            train_epochs=2,
            batch_size=8,
            eval_steps=2,
            log_episode_frequency=1,
            regu_lam=1.0,
            mask_lam=0.1,
            mask_type=mask_type,
            output_dir=str(tmp_path),
            seed=0,
        )
        env.close()

    def test_on_batch_end_receives_sparsity_key(self, tmp_path):
        """on_batch_end callback must receive a dict with a 'sparsity' key."""
        torch.manual_seed(0)
        env = gym.make("MountainCarContinuous-v0")
        ds = self._make_dataset()

        received: List[Mapping[str, Any]] = []
        est_o4.train(
            env=env,
            dataset=ds,
            train_epochs=1,
            batch_size=8,
            eval_steps=1,
            log_episode_frequency=1,
            regu_lam=1.0,
            mask_lam=0.1,
            output_dir=str(tmp_path),
            seed=0,
            on_batch_end=received.append,
        )

        assert len(received) > 0
        for batch_metrics in received:
            assert "sparsity" in batch_metrics, (
                f"on_batch_end dict missing 'sparsity' key: {batch_metrics}"
            )
            assert "reward" in batch_metrics
            assert "regu" in batch_metrics
            assert "total" in batch_metrics

        env.close()

    def test_mask_lam_zero_matches_base_total_loss(self, tmp_path):
        """With mask_lam=0, total_loss must equal reward_loss + regu_lam * regu_loss."""
        torch.manual_seed(7)
        env = gym.make("MountainCarContinuous-v0")
        ds = self._make_dataset()

        received: List[Mapping[str, Any]] = []
        est_o4.train(
            env=env,
            dataset=ds,
            train_epochs=1,
            batch_size=8,
            eval_steps=1,
            log_episode_frequency=1,
            regu_lam=1.0,
            mask_lam=0.0,
            output_dir=str(tmp_path),
            seed=0,
            on_batch_end=received.append,
        )

        for batch_metrics in received:
            expected_total = batch_metrics["reward"] + batch_metrics["regu"]
            np.testing.assert_allclose(
                batch_metrics["total"], expected_total, atol=1e-5
            )

        env.close()

    def test_config_saved_with_mask_type(self, tmp_path):
        """save_config_and_metrics must record mask_type in config.json."""
        torch.manual_seed(1)
        env = gym.make("MountainCarContinuous-v0")
        ds = self._make_dataset()

        est_o4.train(
            env=env,
            dataset=ds,
            train_epochs=1,
            batch_size=8,
            eval_steps=1,
            log_episode_frequency=1,
            regu_lam=1.0,
            mask_lam=0.1,
            mask_type="ste",
            output_dir=str(tmp_path),
            seed=0,
        )

        config_path = tmp_path / "config.json"
        assert config_path.exists()
        with open(config_path, encoding="UTF-8") as fh:
            config = json.load(fh)
        assert config["mask_type"] == "ste"
        assert config["spec"] == "o4"

        env.close()
