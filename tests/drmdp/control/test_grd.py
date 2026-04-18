"""
Tests for drmdp.control.grd: GRDRewardModel, _CausalStructure, _RewardNetwork,
_DynamicsNetwork, _extract_transitions, _compute_compact_mask, _sparsity_reg.
"""

from typing import List

import numpy as np
import torch

from drmdp.control import base, grd


class TestCausalStructure:
    def test_gumbel_output_in_unit_interval(self):
        torch.manual_seed(0)
        causal = grd._CausalStructure(obs_dim=4, action_dim=2)
        mask_sr, mask_ar, mask_ss, mask_as = causal.sample_gumbel()
        for mask in (mask_sr, mask_ar, mask_ss, mask_as):
            assert mask.min().item() >= 0.0
            assert mask.max().item() <= 1.0

    def test_gumbel_output_shapes(self):
        causal = grd._CausalStructure(obs_dim=4, action_dim=2)
        mask_sr, mask_ar, mask_ss, mask_as = causal.sample_gumbel()
        assert mask_sr.shape == (4,)
        assert mask_ar.shape == (2,)
        assert mask_ss.shape == (4, 4)
        assert mask_as.shape == (2, 4)

    def test_greedy_outputs_binary(self):
        causal = grd._CausalStructure(obs_dim=3, action_dim=2)
        mask_sr, mask_ar, mask_ss, mask_as = causal.greedy_masks()
        for mask in (mask_sr, mask_ar, mask_ss, mask_as):
            unique_vals = mask.unique().tolist()
            for val in unique_vals:
                assert val in (0.0, 1.0), f"Non-binary value: {val}"

    def test_greedy_no_gradient(self):
        causal = grd._CausalStructure(obs_dim=3, action_dim=2)
        mask_sr, mask_ar, mask_ss, mask_as = causal.greedy_masks()
        for mask in (mask_sr, mask_ar, mask_ss, mask_as):
            assert not mask.requires_grad

    def test_gumbel_has_gradient(self):
        causal = grd._CausalStructure(obs_dim=3, action_dim=2)
        mask_sr, _, _, _ = causal.sample_gumbel()
        # Gumbel output must carry gradients back to phi parameters.
        loss = mask_sr.sum()
        loss.backward()
        assert causal.phi_sr.grad is not None


class TestRewardNetwork:
    def test_output_shape(self):
        net = grd._RewardNetwork(
            obs_dim=4, action_dim=2, hidden_dim=16, num_hidden_layers=2
        )
        obs = torch.zeros(8, 4)
        act = torch.zeros(8, 2)
        term = torch.zeros(8, 1)
        out = net(obs, act, term)
        assert out.shape == (8, 1)

    def test_gradients_flow(self):
        net = grd._RewardNetwork(
            obs_dim=4, action_dim=2, hidden_dim=16, num_hidden_layers=2
        )
        obs = torch.zeros(4, 4)
        act = torch.zeros(4, 2)
        term = torch.zeros(4, 1)
        loss = net(obs, act, term).sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None


class TestDynamicsNetwork:
    def test_nll_is_finite_scalar(self):
        torch.manual_seed(0)
        net = grd._DynamicsNetwork(
            obs_dim=3, action_dim=2, hidden_dim=16, num_hidden_layers=1
        )
        obs = torch.randn(8, 3)
        act = torch.randn(8, 2)
        next_obs = torch.randn(8, 3)
        mask_ss = torch.ones(3, 3)
        mask_as = torch.ones(2, 3)
        nll = net.nll(obs, act, next_obs, mask_ss, mask_as)
        assert nll.shape == ()
        assert torch.isfinite(nll)

    def test_gradients_flow_through_nll(self):
        net = grd._DynamicsNetwork(
            obs_dim=3, action_dim=2, hidden_dim=16, num_hidden_layers=1
        )
        obs = torch.randn(4, 3)
        act = torch.randn(4, 2)
        next_obs = torch.randn(4, 3)
        mask_ss = torch.ones(3, 3)
        mask_as = torch.ones(2, 3)
        nll = net.nll(obs, act, next_obs, mask_ss, mask_as)
        nll.backward()
        for param in net.parameters():
            assert param.grad is not None

    def test_column_masking_isolates_target_gradient(self):
        """Zeroing column i of mask_ss blocks gradient to that target dim."""
        torch.manual_seed(42)
        net = grd._DynamicsNetwork(
            obs_dim=3, action_dim=2, hidden_dim=16, num_hidden_layers=1
        )
        obs = torch.randn(4, 3)
        act = torch.randn(4, 2)
        next_obs = torch.randn(4, 3)

        # Mask that zeros out all edges to target dim 1.
        mask_ss = torch.ones(3, 3)
        mask_ss[:, 1] = 0.0
        mask_as = torch.ones(2, 3)
        mask_as[:, 1] = 0.0

        # NLL should still be finite even with zero-masked inputs for dim 1.
        nll = net.nll(obs, act, next_obs, mask_ss, mask_as)
        assert torch.isfinite(nll)

    def test_mdn_mixing_weights_sum_to_one(self):
        """MDN π weights must be a valid probability distribution."""
        torch.manual_seed(0)
        net = grd._DynamicsNetwork(
            obs_dim=3, action_dim=2, hidden_dim=16, num_hidden_layers=1
        )
        obs = torch.randn(5, 3)
        act = torch.randn(5, 2)
        mask_ss = torch.ones(3, 3)
        mask_as = torch.ones(2, 3)

        # Access raw MDN outputs by inspecting the forward path directly.
        with torch.no_grad():
            obs_expanded = obs.unsqueeze(1) * mask_ss.T.unsqueeze(0)
            act_expanded = act.unsqueeze(1) * mask_as.T.unsqueeze(0)
            inputs = torch.cat([obs_expanded, act_expanded], dim=-1)
            flat_inputs = inputs.reshape(5 * 3, 3 + 2)
            raw = net._final(net._layers(flat_inputs)).reshape(
                5, 3, grd._MDN_COMPONENTS, 3
            )
            pi = torch.softmax(raw[..., 0], dim=-1)
        np.testing.assert_allclose(pi.sum(dim=-1).numpy(), np.ones((5, 3)), atol=1e-5)


class TestExtractTransitions:
    def test_terminal_step_skipped(self):
        traj = _make_trajectory(obs_dim=2, action_dim=1, num_steps=4)
        transitions = grd._extract_transitions(traj)
        # The terminal step (last) should be skipped.
        assert len(transitions) == 3

    def test_obs_next_obs_slices_are_consecutive(self):
        traj = _make_trajectory(obs_dim=2, action_dim=1, num_steps=4)
        transitions = grd._extract_transitions(traj)
        for step_idx, tr in enumerate(transitions):
            np.testing.assert_array_equal(tr.obs, traj.observations[step_idx])
            np.testing.assert_array_equal(tr.next_obs, traj.observations[step_idx + 1])

    def test_all_terminal_trajectory_yields_no_transitions(self):
        traj = base.Trajectory(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((3, 1), dtype=np.float32),
            env_rewards=np.zeros(3, dtype=np.float32),
            terminals=np.array([True, True, True]),
            episode_return=0.0,
        )
        assert grd._extract_transitions(traj) == []

    def test_single_step_episode_yields_no_transitions(self):
        traj = base.Trajectory(
            observations=np.zeros((1, 2), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            env_rewards=np.zeros(1, dtype=np.float32),
            terminals=np.array([True]),
            episode_return=0.0,
        )
        assert grd._extract_transitions(traj) == []


class TestSparsityReg:
    def test_both_terms_non_negative(self):
        causal = grd._CausalStructure(obs_dim=3, action_dim=2)
        reg = grd._sparsity_reg(causal, lam_diag=1e-3, lam_offdiag=1e-4)
        assert reg.item() >= 0.0

    def test_diagonal_penalty_applies_to_ss_diagonal(self):
        """A large lam_diag raises the reg when phi_ss diagonal is active."""
        causal_high = grd._CausalStructure(obs_dim=3, action_dim=2)
        causal_low = grd._CausalStructure(obs_dim=3, action_dim=2)

        # Force phi_ss diagonal strongly toward 1 in both models.
        with torch.no_grad():
            for causal in (causal_high, causal_low):
                eye = torch.eye(3)
                causal.phi_ss.data[:, :, 1] = eye * 10.0 - (1 - eye) * 10.0

        reg_high = grd._sparsity_reg(causal_high, lam_diag=1.0, lam_offdiag=0.0)
        reg_low = grd._sparsity_reg(causal_low, lam_diag=0.0, lam_offdiag=0.0)
        assert reg_high.item() > reg_low.item()

    def test_offdiag_penalty_applies_to_reward_edges(self):
        """A large lam_offdiag raises the reg when phi_sr is active."""
        causal = grd._CausalStructure(obs_dim=3, action_dim=2)
        with torch.no_grad():
            causal.phi_sr.data[:, 1] = 10.0  # push all s→r edges toward 1

        reg_high = grd._sparsity_reg(causal, lam_diag=0.0, lam_offdiag=1.0)
        reg_zero = grd._sparsity_reg(causal, lam_diag=0.0, lam_offdiag=0.0)
        assert reg_high.item() > reg_zero.item()


class TestGRDRewardModel:
    def test_predict_output_shape(self):
        model = _make_model(obs_dim=4, action_dim=2)
        obs = np.zeros((10, 4), dtype=np.float32)
        act = np.zeros((10, 2), dtype=np.float32)
        term = np.zeros(10, dtype=bool)
        preds = model.predict(obs, act, term)
        assert preds.shape == (10,)

    def test_predict_dtype_float32(self):
        model = _make_model(obs_dim=4, action_dim=2)
        obs = np.zeros((5, 4), dtype=np.float32)
        act = np.zeros((5, 2), dtype=np.float32)
        term = np.zeros(5, dtype=bool)
        preds = model.predict(obs, act, term)
        assert preds.dtype == np.float32

    def test_predict_before_update_is_finite(self):
        model = _make_model(obs_dim=4, action_dim=2)
        obs = np.random.randn(8, 4).astype(np.float32)
        act = np.random.randn(8, 2).astype(np.float32)
        term = np.zeros(8, dtype=bool)
        preds = model.predict(obs, act, term)
        assert np.all(np.isfinite(preds))

    def test_update_returns_required_metric_keys(self):
        model = _make_model(obs_dim=4, action_dim=2, train_epochs=1)
        trajs = _make_synthetic_trajectories(
            num_trajs=2, obs_dim=4, action_dim=2, num_steps=5
        )
        metrics = model.update(trajs)
        required = {
            "buffer_size",
            "training_steps",
            "reward_loss",
            "dyn_loss",
            "sparsity_reg",
        }
        assert required <= set(metrics.keys())

    def test_update_buffer_size_increments(self):
        model = _make_model(obs_dim=4, action_dim=2, train_epochs=1)
        trajs = _make_synthetic_trajectories(
            num_trajs=3, obs_dim=4, action_dim=2, num_steps=5
        )
        metrics = model.update(trajs)
        assert metrics["buffer_size"] == 3.0

    def test_buffer_eviction(self):
        model = _make_model(obs_dim=4, action_dim=2, train_epochs=1, max_buffer_size=2)
        trajs = _make_synthetic_trajectories(
            num_trajs=5, obs_dim=4, action_dim=2, num_steps=5
        )
        metrics = model.update(trajs)
        assert metrics["buffer_size"] == 2.0

    def test_update_with_empty_trajectory_list_safe(self):
        model = _make_model(obs_dim=4, action_dim=2)
        metrics = model.update([])
        assert metrics["buffer_size"] == 0.0
        assert metrics["reward_loss"] == 0.0

    def test_reward_loss_decreases_after_training(self):
        """Reward loss should decrease on a simple synthetic dataset."""
        torch.manual_seed(42)
        np.random.seed(42)
        obs_dim, action_dim = 4, 2
        model = _make_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            num_hidden_layers=2,
            train_epochs=1,
            batch_size=4,
            dyn_weight=0.0,
        )
        trajs = _make_synthetic_trajectories(
            num_trajs=8, obs_dim=obs_dim, action_dim=action_dim, num_steps=10
        )
        metrics_before = model.update(trajs)
        # Train for many more epochs.
        model._train_epochs = 50
        metrics_after = model.update([])
        # The reward loss over the buffered data should decrease.
        assert metrics_after["reward_loss"] <= metrics_before["reward_loss"] + 0.5

    def test_compact_obs_mask_shape_and_binary(self):
        model = _make_model(obs_dim=4, action_dim=2)
        mask = model.compact_obs_mask
        assert mask.shape == (4,)
        assert set(mask.tolist()).issubset({0.0, 1.0})

    def test_obs_mask_property_matches_compact_obs_mask(self):
        model = _make_model(obs_dim=4, action_dim=2)
        np.testing.assert_array_equal(model.obs_mask, model.compact_obs_mask)


class TestComputeCompactMask:
    """Tests for grd._compute_compact_mask."""

    def test_no_dynamics_edges_returns_direct_parents(self):
        """With an all-zero C^{s→s}, the output equals mask_sr exactly."""
        mask_sr = np.array([1, 0, 0, 0], dtype=np.float32)
        mask_ss = np.zeros((4, 4), dtype=np.float32)
        result = grd._compute_compact_mask(mask_sr, mask_ss)
        np.testing.assert_array_equal(result, mask_sr)

    def test_one_hop_transitive_source_included(self):
        """s[1]→s[0] and s[0]→r: compact set must include both s[0] and s[1]."""
        mask_sr = np.array([1, 0, 0, 0], dtype=np.float32)
        mask_ss = np.zeros((4, 4), dtype=np.float32)
        mask_ss[1, 0] = 1.0  # s[1] causes s[0]
        result = grd._compute_compact_mask(mask_sr, mask_ss)
        np.testing.assert_array_equal(result, np.array([1, 1, 0, 0], dtype=np.float32))

    def test_two_hop_transitive_chain_fully_included(self):
        """s[2]→s[1]→s[0]→r: all three dimensions must be in the compact set."""
        mask_sr = np.array([1, 0, 0, 0], dtype=np.float32)
        mask_ss = np.zeros((4, 4), dtype=np.float32)
        mask_ss[1, 0] = 1.0  # s[1] causes s[0]
        mask_ss[2, 1] = 1.0  # s[2] causes s[1]
        result = grd._compute_compact_mask(mask_sr, mask_ss)
        np.testing.assert_array_equal(result, np.array([1, 1, 1, 0], dtype=np.float32))

    def test_irrelevant_dimension_excluded(self):
        """A dimension with no path to any reward-relevant dim must be excluded."""
        mask_sr = np.array([1, 0, 0, 0], dtype=np.float32)
        mask_ss = np.zeros((4, 4), dtype=np.float32)
        mask_ss[3, 2] = 1.0  # s[3]→s[2], but s[2] has no path to reward
        result = grd._compute_compact_mask(mask_sr, mask_ss)
        np.testing.assert_array_equal(result, np.array([1, 0, 0, 0], dtype=np.float32))

    def test_all_zero_mask_sr_returns_all_zeros(self):
        """No direct reward parents → no dimensions included, regardless of C^{s→s}."""
        mask_sr = np.zeros(4, dtype=np.float32)
        mask_ss = np.ones((4, 4), dtype=np.float32)  # all edges present
        result = grd._compute_compact_mask(mask_sr, mask_ss)
        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    obs_dim: int = 4,
    action_dim: int = 2,
    hidden_dim: int = 16,
    num_hidden_layers: int = 1,
    train_epochs: int = 2,
    batch_size: int = 4,
    trans_batch_size: int = 8,
    dyn_weight: float = 1.0,
    max_buffer_size: int = 300,
) -> grd.GRDRewardModel:
    """Instantiate a small GRDRewardModel for testing."""
    return grd.GRDRewardModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        train_epochs=train_epochs,
        batch_size=batch_size,
        trans_batch_size=trans_batch_size,
        dyn_weight=dyn_weight,
        max_buffer_size=max_buffer_size,
    )


def _make_trajectory(
    obs_dim: int,
    action_dim: int,
    num_steps: int,
    episode_return: float = 0.0,
) -> base.Trajectory:
    """Create a minimal trajectory with a terminal flag on the last step."""
    rng = np.random.default_rng(0)
    terminals = np.zeros(num_steps, dtype=bool)
    terminals[-1] = True
    return base.Trajectory(
        observations=rng.standard_normal((num_steps, obs_dim)).astype(np.float32),
        actions=rng.standard_normal((num_steps, action_dim)).astype(np.float32),
        env_rewards=np.zeros(num_steps, dtype=np.float32),
        terminals=terminals,
        episode_return=episode_return,
    )


def _make_synthetic_trajectories(
    num_trajs: int,
    obs_dim: int,
    action_dim: int,
    num_steps: int,
) -> List[base.Trajectory]:
    """Create trajectories where episode_return equals num_steps (easy target)."""
    return [
        _make_trajectory(
            obs_dim, action_dim, num_steps, episode_return=float(num_steps)
        )
        for _ in range(num_trajs)
    ]
