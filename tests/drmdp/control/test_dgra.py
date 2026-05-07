"""
Tests for drmdp.control.dgra: DGRARewardModel, _extract_windows, _Window, _RNetwork.
"""

from typing import List

import numpy as np
import pytest
import torch

from drmdp.control import base, dgra


class TestExtractWindows:
    def test_single_interval_produces_one_window(self):
        """Trajectory with one non-zero reward produces exactly one window."""
        env_rewards = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        traj = _make_trajectory(
            obs=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((3, 1), dtype=np.float32),
            env_rewards=env_rewards,
        )

        windows = dgra._extract_windows(traj)

        assert len(windows) == 1
        assert windows[0].aggregate_reward == pytest.approx(3.0)
        assert windows[0].start_return == pytest.approx(0.0)
        assert windows[0].end_return == pytest.approx(3.0)
        assert windows[0].observations.shape == (3, 2)

    def test_two_intervals_produce_two_windows(self):
        """Trajectory with two non-zero rewards produces two windows in order."""
        env_rewards = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 2.5], dtype=np.float32)
        traj = _make_trajectory(
            obs=np.arange(12, dtype=np.float32).reshape(6, 2),
            actions=np.zeros((6, 1), dtype=np.float32),
            env_rewards=env_rewards,
        )

        windows = dgra._extract_windows(traj)

        assert len(windows) == 2
        assert windows[0].aggregate_reward == pytest.approx(1.5)
        assert windows[0].start_return == pytest.approx(0.0)
        assert windows[0].end_return == pytest.approx(1.5)
        assert windows[1].aggregate_reward == pytest.approx(2.5)
        assert windows[1].start_return == pytest.approx(1.5)
        assert windows[1].end_return == pytest.approx(4.0)

    def test_partial_tail_skipped(self):
        """Steps after the last non-zero reward are not included in any window."""
        env_rewards = np.array([0.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        traj = _make_trajectory(
            obs=np.zeros((5, 2), dtype=np.float32),
            actions=np.zeros((5, 1), dtype=np.float32),
            env_rewards=env_rewards,
        )

        windows = dgra._extract_windows(traj)

        assert len(windows) == 1
        assert windows[0].observations.shape == (2, 2)  # steps 0 and 1 only

    def test_all_zero_rewards_returns_empty(self):
        """All-zero env_rewards produce no windows."""
        traj = _make_trajectory(
            obs=np.zeros((4, 2), dtype=np.float32),
            actions=np.zeros((4, 1), dtype=np.float32),
            env_rewards=np.zeros(4, dtype=np.float32),
        )

        assert dgra._extract_windows(traj) == []

    def test_cumulative_return_chains_across_windows(self):
        """Each window's start_return equals the preceding window's end_return.

        Verifies that _extract_windows correctly threads the running cumulative
        return through consecutive windows rather than resetting it between them.
        """
        env_rewards = np.array([0.0, 1.0, 0.0, -0.5, 0.0, 3.0], dtype=np.float32)
        traj = _make_trajectory(
            obs=np.zeros((6, 2), dtype=np.float32),
            actions=np.zeros((6, 1), dtype=np.float32),
            env_rewards=env_rewards,
        )

        windows = dgra._extract_windows(traj)

        assert len(windows) == 3
        # The start of each window after the first must equal the end of the previous.
        for idx in range(1, len(windows)):
            assert windows[idx].start_return == pytest.approx(
                windows[idx - 1].end_return
            )
        # The first window always starts at zero.
        assert windows[0].start_return == pytest.approx(0.0)
        # The final end_return equals the sum of all aggregate rewards.
        assert windows[-1].end_return == pytest.approx(
            float(env_rewards[env_rewards != 0.0].sum())
        )

    def test_terminal_flag_preserved_in_window(self):
        """terminals in each window match the corresponding trajectory slice."""
        env_rewards = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        terminals = np.array([False, False, False, True])
        traj = base.Trajectory(
            observations=np.zeros((4, 2), dtype=np.float32),
            actions=np.zeros((4, 1), dtype=np.float32),
            env_rewards=env_rewards,
            terminals=terminals,
            episode_return=float(env_rewards.sum()),
        )

        windows = dgra._extract_windows(traj)

        assert len(windows) == 2
        np.testing.assert_array_equal(windows[0].terminals, terminals[0:2])
        np.testing.assert_array_equal(windows[1].terminals, terminals[2:4])

    def test_window_observations_are_correct_slices(self):
        """observations in each window match the corresponding trajectory slice."""
        obs = np.arange(8, dtype=np.float32).reshape(4, 2)
        env_rewards = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        traj = _make_trajectory(
            obs=obs, actions=np.zeros((4, 1)), env_rewards=env_rewards
        )

        windows = dgra._extract_windows(traj)

        np.testing.assert_array_equal(windows[0].observations, obs[0:2])
        np.testing.assert_array_equal(windows[1].observations, obs[2:4])


class TestWindowDataclass:
    def test_window_is_frozen(self):
        """Assigning to a _Window field raises FrozenInstanceError."""
        window = dgra._Window(
            observations=np.zeros((2, 2), dtype=np.float32),
            actions=np.zeros((2, 1), dtype=np.float32),
            terminals=np.zeros(2, dtype=bool),
            aggregate_reward=1.0,
            start_return=0.0,
            end_return=1.0,
        )
        with pytest.raises(Exception):
            window.aggregate_reward = 99.0  # type: ignore[misc]


class TestDGRARewardModel:
    def test_predict_output_shape(self):
        """predict returns a 1-D array of length T."""
        model = dgra.DGRARewardModel(obs_dim=3, action_dim=2)
        obs = np.zeros((5, 3), dtype=np.float32)
        actions = np.zeros((5, 2), dtype=np.float32)
        terminals = np.zeros(5, dtype=bool)

        result = model.predict(obs, actions, terminals)

        assert result.shape == (5,)

    def test_predict_output_dtype_is_float32(self):
        """predict always returns dtype float32."""
        model = dgra.DGRARewardModel(obs_dim=3, action_dim=2)
        obs = np.zeros((4, 3), dtype=np.float64)
        actions = np.zeros((4, 2), dtype=np.float64)
        terminals = np.zeros(4, dtype=bool)

        result = model.predict(obs, actions, terminals)

        assert result.dtype == np.float32

    def test_predict_before_update_does_not_crash(self):
        """predict on a freshly constructed (untrained) model returns finite values."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1)
        obs = np.random.default_rng(0).uniform(size=(6, 2)).astype(np.float32)
        actions = np.random.default_rng(1).uniform(size=(6, 1)).astype(np.float32)
        terminals = np.zeros(6, dtype=bool)

        result = model.predict(obs, actions, terminals)

        assert np.all(np.isfinite(result))

    def test_update_returns_required_metric_keys(self):
        """update() returns a mapping with buffer_size, training_steps, reward_loss, regu_loss."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1, train_epochs=1)
        traj = _make_trajectory_with_window(obs_dim=2, action_dim=1)

        metrics = model.update([traj])

        for key in ("buffer_size", "training_steps", "reward_loss", "regu_loss"):
            assert key in metrics

    def test_update_buffer_size_increments(self):
        """buffer_size grows with each update call."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1, train_epochs=1)
        traj = _make_trajectory_with_window(obs_dim=2, action_dim=1)

        metrics1 = model.update([traj])
        metrics2 = model.update([traj])

        assert metrics2["buffer_size"] > metrics1["buffer_size"]

    def test_buffer_eviction(self):
        """buffer_size never exceeds max_buffer_size; oldest windows are evicted."""
        model = dgra.DGRARewardModel(
            obs_dim=2, action_dim=1, train_epochs=1, max_buffer_size=2
        )
        for _ in range(3):
            model.update([_make_trajectory_with_window(obs_dim=2, action_dim=1)])

        assert len(model._buffer) == 2

    def test_training_steps_positive_after_update(self):
        """training_steps is > 0 when at least one window was processed."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1, train_epochs=1)
        traj = _make_trajectory_with_window(obs_dim=2, action_dim=1)

        metrics = model.update([traj])

        assert metrics["training_steps"] > 0

    def test_update_with_no_windows_returns_safely(self):
        """Trajectory with all-zero rewards causes no exception; returns empty metrics."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1)
        traj = _make_trajectory(
            obs=np.zeros((4, 2), dtype=np.float32),
            actions=np.zeros((4, 1), dtype=np.float32),
            env_rewards=np.zeros(4, dtype=np.float32),
        )

        metrics = model.update([traj])

        assert metrics["buffer_size"] == 0.0
        assert metrics["training_steps"] == 0.0

    def test_convergence_on_synthetic_data(self):
        """reward_loss decreases substantially after many training epochs on simple data.

        Uses synthetic trajectories where every step's reward is 1.0, so the
        aggregate reward per delay-window is exactly `window_length`. A small
        network trained for enough epochs should drive reward_loss toward zero.
        """
        rng = np.random.default_rng(42)
        obs_dim, action_dim = 1, 1

        model = dgra.DGRARewardModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            num_hidden_layers=2,
            learning_rate=5e-3,
            train_epochs=5,
            batch_size=8,
            regu_lam=1.0,
        )

        trajectories = _make_synthetic_trajectories(
            rng=rng, obs_dim=obs_dim, action_dim=action_dim, n_trajs=20
        )

        first_metrics = model.update(trajectories)
        for _ in range(9):
            last_metrics = model.update(trajectories)

        assert last_metrics["reward_loss"] < first_metrics["reward_loss"]


class TestVectorisedUpdate:
    def test_buffer_invalidation_on_extend(self):
        """_stacked_dirty flips True after each update that adds windows."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1, train_epochs=1)
        traj = _make_trajectory_with_window(obs_dim=2, action_dim=1)

        model.update([traj])
        assert model._stacked_dirty is False

        model.update([traj])
        assert model._stacked_dirty is False
        assert model._stacked_obs is not None

    def test_variable_length_windows_padded_correctly(self):
        """Windows of different lengths are padded; padded positions don't leak."""
        # Window 1: 2 steps with reward at step 1 (length=2).
        traj_short = _make_trajectory(
            obs=np.ones((2, 2), dtype=np.float32),
            actions=np.ones((2, 1), dtype=np.float32),
            env_rewards=np.array([0.0, 1.0], dtype=np.float32),
        )
        # Window 2: 4 steps with reward at step 3 (length=4).
        traj_long = _make_trajectory(
            obs=np.ones((4, 2), dtype=np.float32),
            actions=np.ones((4, 1), dtype=np.float32),
            env_rewards=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1, train_epochs=1)
        model.update([traj_short, traj_long])

        assert model._stacked_obs.shape == (2, 4, 2)
        assert model._stacked_mask.shape == (2, 4, 1)
        # Short window: first 2 steps real, last 2 padded.
        np.testing.assert_array_equal(
            model._stacked_mask[0, :, 0].numpy(),
            np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
        # Long window: all 4 steps real.
        np.testing.assert_array_equal(
            model._stacked_mask[1, :, 0].numpy(),
            np.ones(4, dtype=np.float32),
        )

    def test_padded_positions_excluded_from_sum_preds(self):
        """Per-window sum_preds is independent of padded input values.

        Verifies that the (preds * mask).sum(dim=1) aggregation in
        update() is invariant to whatever was written into padded slots —
        the mask must zero out those contributions.
        """
        obs_dim, action_dim = 2, 1
        traj_short = _make_trajectory(
            obs=np.ones((2, obs_dim), dtype=np.float32),
            actions=np.ones((2, action_dim), dtype=np.float32),
            env_rewards=np.array([0.0, 1.0], dtype=np.float32),
        )
        traj_long = _make_trajectory(
            obs=np.ones((4, obs_dim), dtype=np.float32),
            actions=np.ones((4, action_dim), dtype=np.float32),
            env_rewards=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        model = dgra.DGRARewardModel(
            obs_dim=obs_dim, action_dim=action_dim, train_epochs=1
        )
        model.update([traj_short, traj_long])

        # Snapshot stacked state, then perturb padded positions of the
        # short window (indices [0, 2:, :]) with a large value.
        obs_a = model._stacked_obs.clone()
        acts_a = model._stacked_acts.clone()
        terms_a = model._stacked_terms.clone()
        mask = model._stacked_mask

        obs_b = obs_a.clone()
        obs_b[0, 2:, :] = 1e6
        acts_b = acts_a.clone()
        acts_b[0, 2:, :] = 1e6
        terms_b = terms_a.clone()
        terms_b[0, 2:, :] = 1.0

        def _sum_preds(obs_t, act_t, term_t):
            batch_n, window_w, _ = obs_t.shape
            flat = model._net(
                obs_t.reshape(batch_n * window_w, obs_dim),
                act_t.reshape(batch_n * window_w, action_dim),
                term_t.reshape(batch_n * window_w, 1),
            )
            return (flat.reshape(batch_n, window_w, 1) * mask).sum(dim=1).squeeze(-1)

        with torch.no_grad():
            sum_a = _sum_preds(obs_a, acts_a, terms_a)
            sum_b = _sum_preds(obs_b, acts_b, terms_b)

        torch.testing.assert_close(sum_a, sum_b, atol=1e-5, rtol=0.0)

    def test_update_deterministic_with_fixed_seed(self):
        """Same buffer + same torch seed produces identical _net parameters."""

        def _train_once() -> List[torch.Tensor]:
            torch.manual_seed(0)
            np.random.seed(0)
            model = dgra.DGRARewardModel(
                obs_dim=2,
                action_dim=1,
                hidden_dim=16,
                num_hidden_layers=1,
                train_epochs=2,
                batch_size=2,
            )
            trajectories = [
                _make_trajectory_with_window(obs_dim=2, action_dim=1) for _ in range(4)
            ]
            model.update(trajectories)
            return [param.detach().clone() for param in model._net.parameters()]

        params_a = _train_once()
        params_b = _train_once()
        assert len(params_a) == len(params_b)
        for tensor_a, tensor_b in zip(params_a, params_b):
            torch.testing.assert_close(tensor_a, tensor_b, atol=1e-6, rtol=0.0)


class TestPredictMode:
    def test_predict_leaves_net_in_eval_mode(self):
        """predict() always leaves the net in eval mode regardless of starting state."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1)
        model._net.train()  # force train mode

        model.predict(
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.zeros(3, dtype=bool),
        )

        assert model._net.training is False

    def test_predict_skips_eval_when_already_eval(self):
        """The eval() switch is skipped when the net is already in eval mode."""
        model = dgra.DGRARewardModel(obs_dim=2, action_dim=1)
        model._net.eval()

        call_counter = {"n": 0}
        original_eval = model._net.eval

        def _counting_eval():
            call_counter["n"] += 1
            return original_eval()

        model._net.eval = _counting_eval  # type: ignore[method-assign]

        for _ in range(3):
            model.predict(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
                np.zeros(2, dtype=bool),
            )

        assert call_counter["n"] == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(
    obs: np.ndarray,
    actions: np.ndarray,
    env_rewards: np.ndarray,
) -> base.Trajectory:
    """Build a Trajectory with the given imputed reward signal."""
    terminals = np.zeros(len(obs), dtype=bool)
    terminals[-1] = True
    return base.Trajectory(
        observations=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        env_rewards=env_rewards.astype(np.float32),
        terminals=terminals,
        episode_return=float(env_rewards.sum()),
    )


def _make_trajectory_with_window(obs_dim: int, action_dim: int) -> base.Trajectory:
    """Build a minimal 3-step trajectory with one complete delay window."""
    obs = np.zeros((3, obs_dim), dtype=np.float32)
    actions = np.zeros((3, action_dim), dtype=np.float32)
    # Non-zero reward at the last step marks one complete interval.
    env_rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _make_trajectory(obs, actions, env_rewards)


def _make_synthetic_trajectories(
    rng: np.random.Generator,
    obs_dim: int,
    action_dim: int,
    n_trajs: int,
) -> List[base.Trajectory]:
    """Build trajectories where every step reward is 1.0 and window size is 3."""
    trajs = []
    for _ in range(n_trajs):
        obs = rng.uniform(size=(6, obs_dim)).astype(np.float32)
        actions = rng.uniform(size=(6, action_dim)).astype(np.float32)
        # Two complete windows of 3 steps each; aggregate = 3.0 per window.
        env_rewards = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 3.0], dtype=np.float32)
        trajs.append(_make_trajectory(obs, actions, env_rewards))
    return trajs
