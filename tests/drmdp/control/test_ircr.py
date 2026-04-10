"""
Tests for drmdp.control.ircr: IRCRRewardModel.
"""

import numpy as np

from drmdp.control import base, ircr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(
    obs: np.ndarray,
    actions: np.ndarray,
    episode_return: float,
) -> base.Trajectory:
    """Build a Trajectory from obs/actions arrays with a given episode return."""
    n_steps = len(obs)
    env_rewards = np.full(n_steps, episode_return / max(n_steps, 1), dtype=np.float32)
    terminals = np.zeros(n_steps, dtype=bool)
    terminals[-1] = True
    return base.Trajectory(
        observations=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        env_rewards=env_rewards,
        terminals=terminals,
        episode_return=episode_return,
    )


# ---------------------------------------------------------------------------
# TestIRCRRewardModel
# ---------------------------------------------------------------------------


class TestIRCRRewardModel:
    def test_predict_returns_zeros_when_buffer_empty(self):
        """No crash and all-zero output before any trajectories are added."""
        model = ircr.IRCRRewardModel(max_buffer_size=10, k_neighbors=3)
        obs = np.random.default_rng(0).uniform(size=(5, 3)).astype(np.float32)
        actions = np.random.default_rng(1).uniform(size=(5, 2)).astype(np.float32)
        terminals = np.zeros(5, dtype=bool)

        result = model.predict(obs, actions, terminals)

        assert result.shape == (5,)
        np.testing.assert_array_equal(result, 0.0)

    def test_update_stores_trajectories(self):
        """After update with one trajectory, buffer_size metric equals 1."""
        model = ircr.IRCRRewardModel(max_buffer_size=10, k_neighbors=1)
        obs = np.zeros((4, 2), dtype=np.float32)
        actions = np.zeros((4, 1), dtype=np.float32)
        traj = _make_trajectory(obs, actions, episode_return=1.0)

        metrics = model.update([traj])

        assert metrics["buffer_size"] == 1.0

    def test_update_metrics_keys(self):
        """Returned metrics contain buffer_size, r_min, and r_max."""
        model = ircr.IRCRRewardModel(max_buffer_size=10, k_neighbors=1)
        traj = _make_trajectory(np.zeros((3, 2)), np.zeros((3, 1)), episode_return=5.0)

        metrics = model.update([traj])

        assert "buffer_size" in metrics
        assert "r_min" in metrics
        assert "r_max" in metrics

    def test_predict_returns_values_in_unit_interval(self):
        """All guidance rewards are in [0, 1] after adding diverse trajectories."""
        model = ircr.IRCRRewardModel(max_buffer_size=20, k_neighbors=2)
        rng = np.random.default_rng(42)
        for episode_return in [-5.0, 0.0, 3.0, 10.0]:
            obs = rng.uniform(size=(6, 3)).astype(np.float32)
            actions = rng.uniform(size=(6, 2)).astype(np.float32)
            model.update([_make_trajectory(obs, actions, episode_return)])

        query_obs = rng.uniform(size=(20, 3)).astype(np.float32)
        query_actions = rng.uniform(size=(20, 2)).astype(np.float32)
        result = model.predict(query_obs, query_actions, np.zeros(20, dtype=bool))

        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_buffer_size_limit_evicts_oldest(self):
        """Buffer never exceeds max_buffer_size; oldest trajectories are dropped."""
        max_size = 3
        model = ircr.IRCRRewardModel(max_buffer_size=max_size, k_neighbors=1)
        for idx in range(5):
            traj = _make_trajectory(
                np.full((2, 2), idx, dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
                episode_return=float(idx),
            )
            model.update([traj])

        assert len(model._trajectories) == max_size

    def test_k_neighbors_averaging(self):
        """Guidance reward is the average return of the K nearest trajectories."""
        model = ircr.IRCRRewardModel(max_buffer_size=10, k_neighbors=2)

        # Two single-step trajectories so each occupies exactly one point in the
        # flat SA matrix.  With k=2 both are always selected, enabling averaging.
        obs_low = np.array([[0.0]], dtype=np.float32)
        obs_high = np.array([[10.0]], dtype=np.float32)
        actions = np.zeros((1, 1), dtype=np.float32)
        model.update([_make_trajectory(obs_low, actions, episode_return=0.0)])
        model.update([_make_trajectory(obs_high, actions, episode_return=1.0)])

        # Query exactly between the two points; both are selected by k=2 KNN.
        query_obs = np.array([[5.0]], dtype=np.float32)
        query_actions = np.zeros((1, 1), dtype=np.float32)
        result = model.predict(query_obs, query_actions, np.zeros(1, dtype=bool))

        # Mean return = 0.5; after normalising to [0,1]: (0.5 - 0) / (1 - 0) = 0.5.
        np.testing.assert_allclose(result[0], 0.5, atol=0.05)

    def test_identical_state_action_returns_high_guidance(self):
        """Query that only appears in high-return trajectories gets guidance > 0.5."""
        model = ircr.IRCRRewardModel(max_buffer_size=10, k_neighbors=1)

        # Low-return trajectory far from query.
        obs_far = np.full((3, 2), -100.0, dtype=np.float32)
        actions = np.zeros((3, 1), dtype=np.float32)
        model.update([_make_trajectory(obs_far, actions, episode_return=0.0)])

        # High-return trajectory at the exact query point.
        obs_near = np.full((3, 2), 1.0, dtype=np.float32)
        model.update([_make_trajectory(obs_near, actions, episode_return=5.0)])

        query_obs = np.full((1, 2), 1.0, dtype=np.float32)
        query_actions = np.zeros((1, 1), dtype=np.float32)
        result = model.predict(query_obs, query_actions, np.zeros(1, dtype=bool))

        assert result[0] > 0.5

    def test_single_trajectory_prediction_is_zero(self):
        """With a single trajectory, guidance rewards collapse to 0.0 (r_min == r_max)."""
        model = ircr.IRCRRewardModel(max_buffer_size=10, k_neighbors=1)
        obs = np.zeros((4, 2), dtype=np.float32)
        actions = np.zeros((4, 1), dtype=np.float32)
        model.update([_make_trajectory(obs, actions, episode_return=3.0)])

        result = model.predict(obs, actions, np.zeros(4, dtype=bool))

        # r_min == r_max == 3.0; denom = max(0.0, 1e-8) = 1e-8;
        # (3.0 - 3.0) / 1e-8 = 0.0 — normalisation collapses to zero.
        assert result.shape == (4,)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_predict_output_dtype_is_float32(self):
        """Output array has dtype float32."""
        model = ircr.IRCRRewardModel(max_buffer_size=5, k_neighbors=1)
        obs = np.zeros((3, 2), dtype=np.float32)
        actions = np.zeros((3, 1), dtype=np.float32)
        model.update([_make_trajectory(obs, actions, episode_return=1.0)])

        result = model.predict(obs, actions, np.zeros(3, dtype=bool))

        assert result.dtype == np.float32
