"""
Tests for drmdp.control.ircr: IRCR dual-buffer reward model.
"""

import numpy as np
import pytest

from drmdp.control import base, ircr

# ---------------------------------------------------------------------------
# TestFIFOTransitionBuffer
# ---------------------------------------------------------------------------


class TestFIFOTransitionBuffer:
    def test_add_and_sample(self):
        """Add a trajectory and sample transitions with correct shapes."""
        buf = ircr.FIFOTransitionBuffer(obs_dim=3, action_dim=2, max_steps=100)
        traj = _make_trajectory(
            np.random.default_rng(0).uniform(size=(5, 3)).astype(np.float32),
            np.random.default_rng(1).uniform(size=(5, 2)).astype(np.float32),
            episode_return=10.0,
        )
        buf.add_trajectories([traj])

        assert len(buf) == 5
        sample = buf.sample(3)
        assert sample is not None
        assert sample["observations"].shape == (3, 3)
        assert sample["actions"].shape == (3, 2)
        assert sample["next_observations"].shape == (3, 3)
        assert sample["dones"].shape == (3, 1)
        assert sample["credits"].shape == (3, 1)

    def test_ring_buffer_eviction(self):
        """Oldest transitions are overwritten when capacity is exceeded."""
        buf = ircr.FIFOTransitionBuffer(obs_dim=2, action_dim=1, max_steps=10)
        for idx in range(3):
            traj = _make_trajectory(
                np.full((5, 2), idx, dtype=np.float32),
                np.zeros((5, 1), dtype=np.float32),
                episode_return=float(idx),
            )
            buf.add_trajectories([traj])

        assert len(buf) == 10
        stored_obs_vals = set(buf.obs[: len(buf), 0].tolist())
        assert len(stored_obs_vals) == 2
        assert 0.0 not in stored_obs_vals, "oldest batch (idx=0) should be evicted"
        assert 1.0 in stored_obs_vals
        assert 2.0 in stored_obs_vals

    def test_credit_is_mean_return(self):
        """Credit per transition equals episode_return / episode_length."""
        buf = ircr.FIFOTransitionBuffer(obs_dim=2, action_dim=1, max_steps=100)
        traj = _make_trajectory(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((4, 1), dtype=np.float32),
            episode_return=8.0,
        )
        buf.add_trajectories([traj])

        expected_credit = 8.0 / 4.0
        np.testing.assert_allclose(buf.credits[:4, 0], expected_credit)
        np.testing.assert_allclose(buf.credits[5:, 0], 0.0)

    def test_min_max_credit_tracking(self):
        """min_credit_val and max_credit_val are updated after add."""
        buf = ircr.FIFOTransitionBuffer(obs_dim=2, action_dim=1, max_steps=100)

        traj_low = _make_trajectory(
            np.zeros((2, 2)), np.zeros((2, 1)), episode_return=2.0
        )
        traj_high = _make_trajectory(
            np.ones((3, 2)), np.zeros((3, 1)), episode_return=9.0
        )
        buf.add_trajectories([traj_low, traj_high])

        assert buf.min_credit_val == pytest.approx(2.0 / 2)
        assert buf.max_credit_val == pytest.approx(9.0 / 3)

    def test_sample_returns_none_when_insufficient(self):
        """sample() returns None when fewer transitions than batch_size."""
        buf = ircr.FIFOTransitionBuffer(obs_dim=2, action_dim=1, max_steps=100)
        traj = _make_trajectory(np.zeros((3, 2)), np.zeros((3, 1)), episode_return=1.0)
        buf.add_trajectories([traj])

        assert buf.sample(4) is None


# ---------------------------------------------------------------------------
# TestMinHeapTrajectoryBuffer
# ---------------------------------------------------------------------------


class TestMinHeapTrajectoryBuffer:
    def test_rebuild_index_flattens_trajs(self):
        """All trajectories are flattened for sampling"""
        buf = ircr.MinHeapTrajectoryBuffer(obs_dim=2, action_dim=1, max_trajs=2)
        for ret in [1.0, 5.0, 3.0, 10.0]:
            traj = _make_trajectory(
                np.full((3, 2), ret, dtype=np.float32),
                np.zeros((3, 1), dtype=np.float32),
                episode_return=ret,
            )
            buf.add_trajectories([traj])

        np.testing.assert_array_equal(
            buf.obs,
            np.concatenate(
                [
                    np.full((3, 2), 10, dtype=np.float32),
                    np.full((3, 2), 5, dtype=np.float32),
                ]
            ),
        )
        np.testing.assert_array_equal(
            buf.next_obs,
            np.concatenate(
                [
                    np.full((3, 2), 10, dtype=np.float32),
                    np.full((3, 2), 5, dtype=np.float32),
                ]
            ),
        )
        np.testing.assert_array_equal(
            buf.actions,
            np.concatenate(
                [np.zeros((3, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]
            ),
        )
        np.testing.assert_array_equal(
            buf.dones, np.expand_dims(np.array([0, 0, 1, 0, 0, 1]), -1)
        )
        np.testing.assert_allclose(
            buf.credits,
            np.expand_dims(
                np.array(
                    [10 / 3, 10 / 3, 10 / 3, 5 / 3, 5 / 3, 5 / 3], dtype=np.float32
                ),
                -1,
            ),
        )

    def test_retains_best_trajectories(self):
        """After adding more than capacity, only the best are kept."""
        buf = ircr.MinHeapTrajectoryBuffer(obs_dim=2, action_dim=1, max_trajs=2)
        for ret in [1.0, 5.0, 3.0, 10.0]:
            traj = _make_trajectory(
                np.full((3, 2), ret, dtype=np.float32),
                np.zeros((3, 1), dtype=np.float32),
                episode_return=ret,
            )
            buf.add_trajectories([traj])

        stored_returns = sorted(traj.episode_return for traj in buf._traj_data.values())
        assert stored_returns == [5.0, 10.0]

    def test_does_not_replace_when_return_lower(self):
        """Low-return trajectory is rejected when heap is full."""
        buf = ircr.MinHeapTrajectoryBuffer(obs_dim=2, action_dim=1, max_trajs=2)
        for ret in [5.0, 10.0]:
            traj = _make_trajectory(
                np.zeros((3, 2)), np.zeros((3, 1)), episode_return=ret
            )
            buf.add_trajectories([traj])

        low_traj = _make_trajectory(
            np.zeros((3, 2)), np.zeros((3, 1)), episode_return=1.0
        )
        buf.add_trajectories([low_traj])

        stored_returns = sorted(traj.episode_return for traj in buf._traj_data.values())
        assert stored_returns == [5.0, 10.0]

    def test_credit_is_mean_return(self):
        """Credit per transition equals episode_return / episode_length."""
        buf = ircr.MinHeapTrajectoryBuffer(obs_dim=2, action_dim=1, max_trajs=5)
        traj = _make_trajectory(np.zeros((4, 2)), np.zeros((4, 1)), episode_return=12.0)
        buf.add_trajectories([traj])

        assert buf.credits is not None
        expected_credit = 12.0 / 4.0
        np.testing.assert_allclose(buf.credits[:, 0], expected_credit)

    def test_sample_with_replacement(self):
        """Allows oversampling when fewer transitions than batch_size."""
        buf = ircr.MinHeapTrajectoryBuffer(obs_dim=2, action_dim=1, max_trajs=5)
        traj = _make_trajectory(np.zeros((2, 2)), np.zeros((2, 1)), episode_return=1.0)
        buf.add_trajectories([traj])

        sample = buf.sample(10)
        assert sample is not None
        assert sample["observations"].shape == (10, 2)

    def test_sample_returns_none_when_empty(self):
        """sample() returns None when no trajectories are stored."""
        buf = ircr.MinHeapTrajectoryBuffer(obs_dim=2, action_dim=1, max_trajs=5)
        assert buf.sample(5) is None


# ---------------------------------------------------------------------------
# TestIRCRRewardModel
# ---------------------------------------------------------------------------


class TestIRCRRewardModel:
    def test_update_adds_to_both_buffers(self):
        """Trajectories are added to both FIFO and MinHeap."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=100, heap_capacity=5, obs_dim=2, action_dim=1
        )
        traj = _make_trajectory(np.zeros((4, 2)), np.zeros((4, 1)), episode_return=3.0)
        metrics = model.update([traj])

        assert metrics["fifo_size"] == 4.0
        assert metrics["heap_size"] == 4.0

    def test_predict_returns_zeros_when_empty(self):
        """No crash and all-zero output before any trajectories are added."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=100, heap_capacity=5, obs_dim=3, action_dim=2
        )
        obs = np.random.default_rng(0).uniform(size=(5, 3)).astype(np.float32)
        actions = np.random.default_rng(1).uniform(size=(5, 2)).astype(np.float32)
        terminals = np.zeros(5, dtype=bool)

        result = model.predict(obs, actions, terminals)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, 0.0)

    def test_predict_returns_values_in_unit_interval(self):
        """Guidance rewards are in [0, 1] after adding diverse trajectories."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=1000, heap_capacity=5, obs_dim=3, action_dim=2
        )
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

    def test_single_trajectory_prediction_is_zero(self):
        """With one trajectory, r_min == r_max so guidance collapses to ~0."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=100, heap_capacity=5, obs_dim=2, action_dim=1
        )
        traj = _make_trajectory(np.zeros((4, 2)), np.zeros((4, 1)), episode_return=3.0)
        model.update([traj])

        result = model.predict(
            np.zeros((4, 2)), np.zeros((4, 1)), np.zeros(4, dtype=bool)
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_guidance_reward_normalization(self):
        """compute_guidance_rewards maps credits to [0, 1] via min-max."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=1000, heap_capacity=5, obs_dim=2, action_dim=1
        )
        traj_low = _make_trajectory(
            np.zeros((2, 2)), np.zeros((2, 1)), episode_return=2.0
        )
        traj_high = _make_trajectory(
            np.ones((2, 2)), np.zeros((2, 1)), episode_return=10.0
        )
        model.update([traj_low, traj_high])

        credits = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        result = model.compute_guidance_rewards(credits)

        np.testing.assert_allclose(result[0], (1.0 - 1.0) / (5.0 - 1.0), atol=1e-6)
        np.testing.assert_allclose(result[1], (3.0 - 1.0) / (5.0 - 1.0), atol=1e-6)
        np.testing.assert_allclose(result[2], (5.0 - 1.0) / (5.0 - 1.0), atol=1e-6)

    def test_r_min_r_max_span_both_buffers(self):
        """Normalization uses global min/max across FIFO and MinHeap."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=100, heap_capacity=2, obs_dim=2, action_dim=1
        )
        for ret in [1.0, 5.0, 10.0]:
            traj = _make_trajectory(
                np.full((3, 2), ret), np.zeros((3, 1)), episode_return=ret
            )
            model.update([traj])

        fifo_min = model._fifo.min_credit_val
        fifo_max = model._fifo.max_credit_val
        heap_min = model._heap.min_credit_val
        heap_max = model._heap.max_credit_val

        assert fifo_min is not None
        assert heap_min is not None
        assert model.r_min == min(fifo_min, heap_min)
        assert model.r_max == max(fifo_max, heap_max)

    def test_sample_returns_none_when_empty(self):
        """sample() returns None when buffers are empty."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=100, heap_capacity=5, obs_dim=2, action_dim=1
        )
        assert model.sample(10) is None

    def test_sample_returns_merged_batch(self):
        """sample() returns merged transitions from both buffers."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=1000, heap_capacity=5, obs_dim=2, action_dim=1
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            traj = _make_trajectory(
                rng.uniform(size=(20, 2)).astype(np.float32),
                rng.uniform(size=(20, 1)).astype(np.float32),
                episode_return=rng.uniform(0, 10),
            )
            model.update([traj])

        batch = model.sample(64)
        assert batch is not None
        assert batch["observations"].shape == (64, 2)
        assert batch["guidance_rewards"].shape == (64,)
        assert batch["guidance_rewards"].min() >= -1e-6
        assert batch["guidance_rewards"].max() <= 1.0 + 1e-6

    def test_predict_output_dtype_is_float32(self):
        """Output array has dtype float32."""
        model = ircr.IRCRRewardModel(
            fifo_capacity=100, heap_capacity=5, obs_dim=2, action_dim=1
        )
        traj = _make_trajectory(np.zeros((3, 2)), np.zeros((3, 1)), episode_return=1.0)
        model.update([traj])
        result = model.predict(
            np.zeros((3, 2)), np.zeros((3, 1)), np.zeros(3, dtype=bool)
        )
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestIRCRReplayBuffer
# ---------------------------------------------------------------------------


class TestIRCRReplayBuffer:
    def test_sample_draws_from_ircr_model(self):
        """When IRCR buffers have data, sample() returns guidance-reward batches."""
        import gymnasium

        env = gymnasium.make("Pendulum-v1")
        model = ircr.IRCRRewardModel(
            fifo_capacity=1000, heap_capacity=5, obs_dim=3, action_dim=1
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            traj = _make_trajectory(
                rng.uniform(size=(20, 3)).astype(np.float32),
                rng.uniform(-2, 2, size=(20, 1)).astype(np.float32),
                episode_return=rng.uniform(0, 10),
            )
            model.update([traj])

        buf = ircr.IRCRReplayBuffer(
            buffer_size=100,
            observation_space=env.observation_space,
            action_space=env.action_space,
            reward_model=model,
        )
        batch = buf.sample(32)
        assert batch.rewards.shape == (32, 1)
        assert batch.observations.shape[1] == 3
        env.close()

    def test_fallback_when_buffers_empty(self):
        """Falls back to standard sampling with zeroed rewards."""
        import gymnasium

        env = gymnasium.make("Pendulum-v1")
        model = ircr.IRCRRewardModel(
            fifo_capacity=1000, heap_capacity=5, obs_dim=3, action_dim=1
        )
        buf = ircr.IRCRReplayBuffer(
            buffer_size=100,
            observation_space=env.observation_space,
            action_space=env.action_space,
            reward_model=model,
        )
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, _, info = env.step(action)
        buf.add(obs, next_obs, action, reward, terminated, [info])
        batch = buf.sample(1)
        assert batch.rewards.item() == 0.0
        env.close()


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
    infos = tuple({"interval_end": False} for _ in range(n_steps))
    return base.Trajectory(
        observations=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        env_rewards=env_rewards,
        terminals=terminals,
        infos=infos,
        episode_return=episode_return,
    )
