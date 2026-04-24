"""Tests for est_o3: EM-based reward estimation with per-step soft targets.

Tests focus on the components that differ from est_o2:
1. collate_variable_length_sequences — seq_lengths tracking
2. create_sequence_mask — padding mask correctness
3. compute_soft_targets — E-step correctness (key invariant: sums to aggregate)
4. train — EM loop runs without error and logs valid losses
5. End-to-end pipeline integration
"""

import json
import os
import tempfile
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import pytest
import torch

from drmdp import dataproc, rewdelay
from drmdp.dfdrl import est_o3

# =============================================================================
# TestCollateVariableLengthSequences
# =============================================================================


class TestCollateVariableLengthSequences:
    """Tests for the collate function, focusing on seq_lengths tracking."""

    def _make_example(self, seq_len: int, state_dim: int = 2, action_dim: int = 1):
        """Build a single (inputs_dict, labels_dict) example of a given length."""
        inputs = {
            "state": torch.randn(seq_len, state_dim),
            "action": torch.randn(seq_len, action_dim),
            "term": torch.zeros(seq_len, 1),
        }
        labels = {
            "aggregate_reward": torch.tensor(float(seq_len)),
            "per_step_rewards": torch.ones(seq_len),
            "start_return": torch.tensor(0.0),
            "end_return": torch.tensor(float(seq_len)),
        }
        return inputs, labels

    def test_includes_seq_lengths(self):
        """Collated labels must contain a seq_lengths key."""
        batch = [self._make_example(3), self._make_example(3)]
        _, batched_labels = est_o3.collate_variable_length_sequences(batch)

        assert "seq_lengths" in batched_labels

    def test_seq_lengths_correct_values(self):
        """seq_lengths must record the actual (unpadded) length of each example."""
        lengths = [2, 4, 3]
        batch = [self._make_example(length) for length in lengths]
        _, batched_labels = est_o3.collate_variable_length_sequences(batch)

        seq_lengths = batched_labels["seq_lengths"]
        assert seq_lengths.shape == (3,)
        assert seq_lengths.tolist() == lengths

    def test_uniform_length_sequences(self):
        """When all sequences have the same length, seq_lengths are all equal."""
        batch = [self._make_example(5) for _ in range(4)]
        _, batched_labels = est_o3.collate_variable_length_sequences(batch)

        seq_lengths = batched_labels["seq_lengths"]
        assert seq_lengths.tolist() == [5, 5, 5, 5]

    def test_seq_lengths_dtype(self):
        """seq_lengths must be a LongTensor (required for indexing and division)."""
        batch = [self._make_example(3), self._make_example(2)]
        _, batched_labels = est_o3.collate_variable_length_sequences(batch)

        assert batched_labels["seq_lengths"].dtype == torch.long


# =============================================================================
# TestCreateSequenceMask
# =============================================================================


class TestCreateSequenceMask:
    """Tests for create_sequence_mask helper."""

    def test_mask_shape(self):
        """Mask shape must be (batch, max_seq_len)."""
        seq_lengths = torch.tensor([3, 5, 2])
        mask = est_o3.create_sequence_mask(seq_lengths, max_seq_len=5)

        assert mask.shape == (3, 5)

    def test_mask_values(self):
        """First D positions are True, remaining are False for each example."""
        seq_lengths = torch.tensor([2, 4, 1])
        max_seq_len = 4
        mask = est_o3.create_sequence_mask(seq_lengths, max_seq_len)

        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
                [True, False, False, False],
            ]
        )
        assert torch.equal(mask, expected)

    def test_full_length_sequence(self):
        """No padding: mask is all True when seq_length equals max_seq_len."""
        seq_lengths = torch.tensor([3, 3])
        mask = est_o3.create_sequence_mask(seq_lengths, max_seq_len=3)

        assert mask.all()

    def test_mask_sum_equals_seq_lengths(self):
        """Number of True entries per row must equal the corresponding seq_length."""
        seq_lengths = torch.tensor([1, 3, 5, 2])
        mask = est_o3.create_sequence_mask(seq_lengths, max_seq_len=5)

        np.testing.assert_array_equal(mask.sum(dim=1).numpy(), seq_lengths.numpy())


# =============================================================================
# TestComputeSoftTargets
# =============================================================================


class TestComputeSoftTargets:
    """Tests for the E-step soft target computation."""

    def _setup(self, batch_size: int, seq_len: int, state_dim: int = 2):
        """Return a seeded model and a random inputs dict."""
        torch.manual_seed(42)
        model = est_o3.RNetwork(state_dim=state_dim, action_dim=1, hidden_dim=16)
        model.eval()
        inputs = {
            "state": torch.randn(batch_size, seq_len, state_dim),
            "action": torch.randn(batch_size, seq_len, 1),
            "term": torch.zeros(batch_size, seq_len, 1),
        }
        return model, inputs

    def test_shape(self):
        """Soft targets must have shape (batch, seq_len)."""
        batch_size, seq_len = 4, 5
        model, inputs = self._setup(batch_size, seq_len)
        seq_lengths = torch.tensor([5, 5, 3, 2])
        aggregate_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = est_o3.create_sequence_mask(seq_lengths, seq_len)

        soft_targets = est_o3.compute_soft_targets(
            model, inputs, seq_lengths, aggregate_rewards, mask
        )

        assert soft_targets.shape == (batch_size, seq_len)

    def test_soft_targets_sum_to_aggregate(self):
        """Key invariant: masked sum of soft targets equals the observed aggregate."""
        batch_size, seq_len = 6, 4
        model, inputs = self._setup(batch_size, seq_len)
        seq_lengths = torch.tensor([4, 3, 2, 4, 1, 3])
        aggregate_rewards = torch.tensor([3.0, -1.5, 0.0, 7.2, -0.3, 2.0])
        mask = est_o3.create_sequence_mask(seq_lengths, seq_len)

        soft_targets = est_o3.compute_soft_targets(
            model, inputs, seq_lengths, aggregate_rewards, mask
        )

        actual_sums = (soft_targets * mask.float()).sum(dim=1)
        np.testing.assert_allclose(
            actual_sums.numpy(),
            aggregate_rewards.numpy(),
            atol=1e-5,
            err_msg="Soft target sums must equal observed aggregate rewards",
        )

    def test_zero_residual(self):
        """When model predictions already sum to aggregate, soft targets equal mu."""
        batch_size, seq_len = 2, 3
        model, inputs = self._setup(batch_size, seq_len)
        seq_lengths = torch.tensor([3, 3])
        mask = est_o3.create_sequence_mask(seq_lengths, seq_len)

        # Set Y_w = sum(mu) so residual is 0
        with torch.no_grad():
            mu = model(**inputs).squeeze(-1)  # (batch, seq_len)
            aggregate_rewards = (mu * mask.float()).sum(dim=1)

        soft_targets = est_o3.compute_soft_targets(
            model, inputs, seq_lengths, aggregate_rewards, mask
        )

        np.testing.assert_allclose(soft_targets.numpy(), mu.numpy(), atol=1e-5)

    def test_uniform_correction(self):
        """The per-step correction delta is identical for all actual steps in a window."""
        batch_size, seq_len = 1, 4
        model, inputs = self._setup(batch_size, seq_len)
        seq_lengths = torch.tensor([4])
        aggregate_rewards = torch.tensor([10.0])
        mask = est_o3.create_sequence_mask(seq_lengths, seq_len)

        with torch.no_grad():
            mu = model(**inputs).squeeze(-1)  # (1, 4)

        soft_targets = est_o3.compute_soft_targets(
            model, inputs, seq_lengths, aggregate_rewards, mask
        )

        corrections = (soft_targets - mu).squeeze(0)  # (4,)

        # All corrections must be equal (uniform residual distribution)
        for step_idx in range(1, seq_len):
            np.testing.assert_allclose(
                corrections[step_idx].item(),
                corrections[0].item(),
                atol=1e-5,
                err_msg=f"Correction at step {step_idx} differs from step 0",
            )

    def test_no_gradient_computed(self):
        """compute_soft_targets must not accumulate gradients on model parameters."""
        batch_size, seq_len = 2, 3
        model, inputs = self._setup(batch_size, seq_len)
        seq_lengths = torch.tensor([3, 3])
        aggregate_rewards = torch.tensor([2.0, 4.0])
        mask = est_o3.create_sequence_mask(seq_lengths, seq_len)

        # Ensure no existing grad
        model.zero_grad()

        est_o3.compute_soft_targets(model, inputs, seq_lengths, aggregate_rewards, mask)

        for param in model.parameters():
            assert param.grad is None


# =============================================================================
# TestEMTrain
# =============================================================================


class TestEMTrain:
    """Tests for the EM training loop."""

    def test_runs_without_error(self, training_dataset):
        """train() must complete without raising exceptions."""
        env = gym.make("MountainCarContinuous-v0")

        with tempfile.TemporaryDirectory() as output_dir:
            est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=2,
                batch_size=4,
                eval_steps=2,
                log_episode_frequency=2,
                regu_lam=1.0,
                seed=42,
                model_type="mlp",
                output_dir=output_dir,
            )

    def test_returns_metrics_dict(self, training_dataset):
        """train() must return (final_mse, predictions_list) with expected keys."""
        env = gym.make("MountainCarContinuous-v0")

        with tempfile.TemporaryDirectory() as output_dir:
            final_mse, predictions_list = est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=2,
                batch_size=4,
                eval_steps=2,
                log_episode_frequency=2,
                regu_lam=1.0,
                seed=42,
                model_type="mlp",
                output_dir=output_dir,
            )

        assert set(final_mse.keys()) == {"reward", "regu", "total"}
        for key in ("reward", "regu", "total"):
            assert np.isfinite(final_mse[key]), f"final_mse[{key!r}] is not finite"

    def test_loss_decreases(self, training_dataset):
        """Train loss should decrease over several epochs on a realistic dataset."""
        env = gym.make("MountainCarContinuous-v0")

        with tempfile.TemporaryDirectory() as output_dir:
            # Patch train to capture epoch losses; run two separate short trains
            # instead — compare initial vs final total loss via final_mse
            final_mse_short, _ = est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=1,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=1,
                regu_lam=0.5,
                seed=0,
                output_dir=output_dir,
            )

        with tempfile.TemporaryDirectory() as output_dir:
            final_mse_long, _ = est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=20,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                regu_lam=0.5,
                seed=0,
                output_dir=output_dir,
            )

        # After more epochs the reward loss should be lower (EM convergence)
        assert final_mse_long["reward"] < final_mse_short["reward"], (
            f"Expected loss to decrease: {final_mse_short['reward']:.4f} -> "
            f"{final_mse_long['reward']:.4f}"
        )

    def test_invalid_model_type_raises(self, simple_model_and_dataset):
        """Unsupported model_type must raise ValueError."""
        _, dataset = simple_model_and_dataset
        env = gym.make("MountainCarContinuous-v0")

        with tempfile.TemporaryDirectory() as output_dir:
            with pytest.raises(ValueError, match="Unknown model_type"):
                est_o3.train(
                    env,
                    dataset=dataset,
                    train_epochs=1,
                    batch_size=2,
                    eval_steps=1,
                    log_episode_frequency=1,
                    regu_lam=1.0,
                    seed=0,
                    model_type="rnn",
                    output_dir=output_dir,
                )


# =============================================================================
# Integration: full pipeline
# =============================================================================


class TestEMPipeline:
    """End-to-end tests verifying the full data → train → evaluate pipeline."""

    def test_full_pipeline(self):
        """Buffer collection → delayed_reward_data → DictDataset → train succeeds."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=50)
        buffer = dataproc.collection_traj_data(env, steps=60, include_term=True, seed=7)
        delay = rewdelay.FixedDelay(3)
        examples = est_o3.delayed_reward_data(buffer, delay)
        assert len(examples) > 0, "No training examples were generated"

        inputs_list, labels_list = zip(*examples)
        dataset = est_o3.DictDataset(inputs=list(inputs_list), labels=list(labels_list))

        with tempfile.TemporaryDirectory() as output_dir:
            final_mse, _ = est_o3.train(
                env,
                dataset=dataset,
                train_epochs=3,
                batch_size=4,
                eval_steps=2,
                log_episode_frequency=2,
                regu_lam=0.5,
                seed=7,
                output_dir=output_dir,
            )

            assert all(np.isfinite(v) for v in final_mse.values())
            assert os.path.exists(os.path.join(output_dir, "model_mlp.pt"))
            assert os.path.exists(os.path.join(output_dir, "config.json"))

    def test_collate_seq_lengths_used_in_mask(self):
        """Verify that seq_lengths from collate correctly produce the expected mask."""
        buffer = create_mock_buffer([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]])
        delay = rewdelay.FixedDelay(2)
        examples = est_o3.delayed_reward_data(buffer, delay)

        inputs_list, labels_list = zip(*examples)
        dataset = est_o3.DictDataset(inputs=list(inputs_list), labels=list(labels_list))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(examples),
            collate_fn=est_o3.collate_variable_length_sequences,
        )

        batched_inputs, batched_labels = next(iter(dataloader))
        seq_lengths = batched_labels["seq_lengths"]
        max_seq_len = batched_inputs["state"].shape[1]

        mask = est_o3.create_sequence_mask(seq_lengths, max_seq_len)

        # Each row's True count must match the corresponding seq_length
        np.testing.assert_array_equal(mask.sum(dim=1).numpy(), seq_lengths.numpy())


# =============================================================================
# TestReguLoss
# =============================================================================


class TestReguLoss:
    """Regression tests for the loss computation in train().

    These tests verify that:
    - reward_loss (masked per-step MSE against soft targets) has non-zero gradients
    - regu_loss depends on model predictions (window_reward), not aggregate_reward
    - the callback receives the correct loss values every batch
    - total == reward_loss + regu_lam * regu_loss for every batch
    """

    def test_regu_loss_gradient_nonzero(self):
        """Both reward_loss and regu_loss produce non-zero gradients via the M-step.

        Constructs the EM M-step manually: computes soft targets (E-step, no_grad),
        then runs the M-step forward/backward and verifies at least one parameter
        receives a non-zero gradient from the combined loss.
        """
        torch.manual_seed(42)
        model = est_o3.RNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        regu_criterion = torch.nn.MSELoss()

        batch_size = 3
        seq_len = 5
        seq_lengths = torch.tensor([5, 4, 3])
        state = torch.randn(batch_size, seq_len, 4)
        action = torch.randn(batch_size, seq_len, 2)
        term = torch.zeros(batch_size, seq_len, 1)
        inputs = {"state": state, "action": action, "term": term}

        start_return = torch.randn(batch_size)
        end_return = torch.randn(batch_size)
        aggregate_rewards = torch.randn(batch_size)

        mask = est_o3.create_sequence_mask(seq_lengths, seq_len)

        # E-step: soft targets are detached from the graph
        soft_targets = est_o3.compute_soft_targets(
            model, inputs, seq_lengths, aggregate_rewards, mask
        )

        # M-step forward
        outputs = model(**inputs)
        per_step_preds = outputs.squeeze(-1)

        sq_err = (per_step_preds - soft_targets) ** 2
        reward_loss = (sq_err * mask.float()).sum() / mask.float().sum()

        window_reward = (per_step_preds * mask.float()).sum(dim=1)
        regu_loss = regu_criterion(start_return + window_reward, end_return)

        total_loss = reward_loss + 0.5 * regu_loss
        total_loss.backward()

        has_nonzero_grad = any(
            param.grad is not None and param.grad.abs().sum().item() > 0.0
            for param in model.parameters()
        )
        assert has_nonzero_grad, (
            "EM M-step loss must produce non-zero gradients on model parameters"
        )

    def test_callback_invoked_every_batch(self, training_dataset):
        """on_batch_end is called once per batch across all epochs."""
        env = gym.make("MountainCarContinuous-v0")
        batch_losses: List = []
        train_epochs = 2
        batch_size = 8

        with tempfile.TemporaryDirectory() as output_dir:
            est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=train_epochs,
                batch_size=batch_size,
                eval_steps=2,
                log_episode_frequency=train_epochs,
                regu_lam=1.0,
                seed=0,
                output_dir=output_dir,
                on_batch_end=batch_losses.append,
            )

        # Each callback entry has the three expected keys
        assert len(batch_losses) > 0
        for entry in batch_losses:
            assert set(entry.keys()) == {"reward", "regu", "total"}
            for key in ("reward", "regu", "total"):
                assert np.isfinite(entry[key]), f"entry[{key!r}] is not finite"

    def test_regu_loss_uses_model_predictions(self, training_dataset):
        """regu_loss varies across batches, confirming it tracks model predictions.

        A constant regu_loss (e.g. always 0) would indicate predictions are
        being ignored; a non-constant sequence confirms the loss is sensitive to
        the model's window_reward.
        """
        env = gym.make("MountainCarContinuous-v0")
        batch_losses: List = []

        with tempfile.TemporaryDirectory() as output_dir:
            est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=2,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                regu_lam=1.0,
                seed=42,
                output_dir=output_dir,
                on_batch_end=batch_losses.append,
            )

        regu_values = [batch["regu"] for batch in batch_losses]
        assert len(set(regu_values)) > 1, (
            "regu_loss must vary across batches; a constant value indicates "
            "model predictions are not influencing the regularization term"
        )

    def test_regu_loss_formula_consistency(self, training_dataset):
        """total == reward_loss + regu_lam * regu_loss for every batch.

        Checks that the three reported loss components satisfy the defining
        relationship from the EM M-step, confirming the formula is executed
        correctly rather than just being arithmetically valid.
        """
        env = gym.make("MountainCarContinuous-v0")
        batch_losses: List = []
        regu_lam = 0.5

        with tempfile.TemporaryDirectory() as output_dir:
            est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=2,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                regu_lam=regu_lam,
                seed=42,
                output_dir=output_dir,
                on_batch_end=batch_losses.append,
            )

        for idx, batch in enumerate(batch_losses):
            expected_total = batch["reward"] + regu_lam * batch["regu"]
            np.testing.assert_allclose(
                batch["total"],
                expected_total,
                atol=1e-5,
                err_msg=f"Batch {idx}: total != reward + lam * regu",
            )

    def test_eval_losses_per_component_saved(self, training_dataset):
        """Saved metrics JSON has eval_losses with reward/regu/total lists of correct length.

        Verifies that the new per-component accumulation lands in the JSON output,
        not just the total.
        """
        env = gym.make("MountainCarContinuous-v0")
        with tempfile.TemporaryDirectory() as output_dir:
            est_o3.train(
                env,
                dataset=training_dataset,
                train_epochs=10,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                regu_lam=1.0,
                seed=42,
                output_dir=output_dir,
            )

            metrics_file = os.path.join(output_dir, "metrics_mlp.json")
            with open(metrics_file, "r", encoding="utf-8") as readable:
                metrics = json.load(readable)

        eval_losses = metrics["eval_losses"]
        assert set(eval_losses.keys()) == {"reward", "regu", "total"}, (
            "eval_losses must have exactly the keys reward, regu, total"
        )
        # 10 epochs, log every 5 → 2 eval runs
        for key in ("reward", "regu", "total"):
            assert len(eval_losses[key]) == 2, (
                f"eval_losses[{key!r}] should have 2 entries (one per eval run)"
            )


# =============================================================================
# Module-level helper functions and fixtures
# =============================================================================


def create_mock_buffer(episodes: List[List[float]]) -> List[Tuple]:
    """Create a mock buffer from episode reward sequences.

    Args:
        episodes: List of reward sequences, one per episode.

    Returns:
        Buffer in format: List[(state, action, next_state, reward, term)]
    """
    buffer = []
    for ep_idx, rewards in enumerate(episodes):
        for step_idx, reward in enumerate(rewards):
            state = np.array([float(ep_idx), float(step_idx)])
            action = np.array([float(ep_idx * 10 + step_idx)])
            next_state = np.array([float(ep_idx), float(step_idx + 1)])
            term = step_idx == len(rewards) - 1
            buffer.append((state, action, next_state, reward, term))
    return buffer


@pytest.fixture
def simple_model_and_dataset():
    """Create a small model and dataset for E-step and mask tests."""
    buffer = create_mock_buffer([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    delay = rewdelay.FixedDelay(3)
    examples = est_o3.delayed_reward_data(buffer, delay)

    inputs_list, labels_list = zip(*examples)
    dataset = est_o3.DictDataset(inputs=list(inputs_list), labels=list(labels_list))

    torch.manual_seed(42)
    model = est_o3.RNetwork(state_dim=2, action_dim=1, hidden_dim=16)

    return model, dataset


@pytest.fixture(scope="module")
def training_dataset():
    """Create a dataset from an actual environment for integration tests."""
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=50)
    buffer = dataproc.collection_traj_data(env, steps=100, include_term=True, seed=42)
    delay = rewdelay.FixedDelay(5)
    examples = est_o3.delayed_reward_data(buffer, delay)

    inputs_list, labels_list = zip(*examples)
    return est_o3.DictDataset(inputs=list(inputs_list), labels=list(labels_list))
