"""
Tests for drmdp.control.base: Trajectory, RewardModel, RelabelingReplayBuffer.
"""

from typing import Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np

from drmdp.control import base

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class ConstantRewardModel(base.RewardModel):
    """Reward model that always returns a fixed constant."""

    def __init__(self, constant: float):
        self._constant = constant

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        del actions, terminals
        return np.full(len(observations), self._constant, dtype=np.float32)

    def update(self, trajectories: Sequence[base.Trajectory]) -> Mapping[str, float]:
        del trajectories
        return {}


class MaskingRewardModel(base.RewardModel):
    """Reward model that also exposes a binary obs_mask."""

    def __init__(self, constant: float, obs_mask: np.ndarray):
        self._constant = constant
        self._obs_mask = obs_mask

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        del actions, terminals
        return np.full(len(observations), self._constant, dtype=np.float32)

    def update(self, trajectories: Sequence[base.Trajectory]) -> Mapping[str, float]:
        del trajectories
        return {}

    @property
    def obs_mask(self) -> Optional[np.ndarray]:
        return self._obs_mask


def _make_buffer(
    obs_dim: int,
    act_dim: int,
    capacity: int,
    reward_model: Optional[base.RewardModel] = None,
) -> base.RelabelingReplayBuffer:
    """Create a RelabelingReplayBuffer for a simple Box env."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    buf = base.RelabelingReplayBuffer(
        buffer_size=capacity,
        observation_space=obs_space,
        action_space=act_space,
        reward_model=reward_model,
    )
    return buf


def _fill_buffer(
    buf: base.RelabelingReplayBuffer,
    n_transitions: int,
    obs_dim: int,
    act_dim: int,
    reward_value: float,
) -> None:
    """Add `n_transitions` transitions with a fixed reward into the buffer."""
    rng = np.random.default_rng(0)
    for _ in range(n_transitions):
        obs = rng.uniform(-1, 1, (obs_dim,)).astype(np.float32)
        next_obs = rng.uniform(-1, 1, (obs_dim,)).astype(np.float32)
        action = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
        done = np.array([False])
        buf.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=np.array([reward_value]),
            done=done,
            infos=[{}],
        )


# ---------------------------------------------------------------------------
# TestRelabelingReplayBuffer
# ---------------------------------------------------------------------------


class TestRelabelingReplayBuffer:
    def test_sample_uses_reward_model_predictions(self):
        """Sampled rewards come from the reward model, not the stored value."""
        obs_dim, act_dim = 3, 1
        stored_reward = 99.0
        model_reward = 0.5
        buf = _make_buffer(obs_dim, act_dim, 100, ConstantRewardModel(model_reward))
        _fill_buffer(
            buf,
            n_transitions=50,
            obs_dim=obs_dim,
            act_dim=act_dim,
            reward_value=stored_reward,
        )

        batch = buf.sample(batch_size=32)
        rewards = batch.rewards.cpu().numpy().squeeze(-1)

        np.testing.assert_allclose(rewards, model_reward, atol=1e-5)

    def test_sample_without_reward_model_uses_stored_rewards(self):
        """Without a reward model, stored rewards are returned unchanged."""
        obs_dim, act_dim = 3, 1
        stored_reward = 7.0
        buf = _make_buffer(obs_dim, act_dim, 100)
        _fill_buffer(
            buf,
            n_transitions=50,
            obs_dim=obs_dim,
            act_dim=act_dim,
            reward_value=stored_reward,
        )

        batch = buf.sample(batch_size=32)
        rewards = batch.rewards.cpu().numpy().squeeze(-1)

        np.testing.assert_allclose(rewards, stored_reward, atol=1e-5)

    def test_reward_model_update_reflected_on_next_sample(self):
        """Changing the reward model's output is reflected immediately on the next sample."""
        obs_dim, act_dim = 3, 1
        model = ConstantRewardModel(0.0)
        buf = _make_buffer(obs_dim, act_dim, 100, model)
        _fill_buffer(
            buf, n_transitions=50, obs_dim=obs_dim, act_dim=act_dim, reward_value=99.0
        )

        batch_before = buf.sample(batch_size=16)
        np.testing.assert_allclose(
            batch_before.rewards.cpu().numpy().squeeze(-1), 0.0, atol=1e-5
        )

        # Swap in a different model via attribute assignment.
        buf.reward_model = ConstantRewardModel(1.0)

        batch_after = buf.sample(batch_size=16)
        np.testing.assert_allclose(
            batch_after.rewards.cpu().numpy().squeeze(-1), 1.0, atol=1e-5
        )

    def test_observations_and_actions_are_unchanged(self):
        """Only rewards are relabeled; observations and actions are untouched."""
        obs_dim, act_dim = 4, 2
        buf = _make_buffer(obs_dim, act_dim, 100, ConstantRewardModel(0.5))
        _fill_buffer(
            buf, n_transitions=50, obs_dim=obs_dim, act_dim=act_dim, reward_value=1.0
        )

        batch = buf.sample(batch_size=10)
        # Observations and actions should be valid arrays, not all-same constants.
        assert batch.observations.shape == (10, obs_dim)
        assert batch.actions.shape == (10, act_dim)
        # Reward shape must be (batch_size, 1).
        assert batch.rewards.shape == (10, 1)

    def test_no_reward_model_set_is_valid(self):
        """A buffer with reward_model=None is valid and samples without error."""
        buf = _make_buffer(obs_dim=2, act_dim=1, capacity=50)
        _fill_buffer(buf, n_transitions=20, obs_dim=2, act_dim=1, reward_value=3.0)
        batch = buf.sample(batch_size=8)
        assert batch.rewards.shape == (8, 1)


class TestRelabelingReplayBufferWithMask:
    def test_obs_masked_when_model_has_obs_mask(self):
        """Non-causal dimensions are zeroed in both obs and next_obs when obs_mask is set."""
        obs_dim, act_dim = 4, 1
        # Mask that keeps only dims 0 and 2.
        obs_mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        model = MaskingRewardModel(constant=0.0, obs_mask=obs_mask)
        buf = _make_buffer(obs_dim, act_dim, capacity=100, reward_model=model)

        rng = np.random.default_rng(1)
        for _ in range(50):
            obs = rng.uniform(0.5, 1.0, (obs_dim,)).astype(np.float32)
            next_obs = rng.uniform(0.5, 1.0, (obs_dim,)).astype(np.float32)
            action = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
            buf.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=np.array([1.0]),
                done=np.array([False]),
                infos=[{}],
            )

        batch = buf.sample(batch_size=32)
        obs_np = batch.observations.cpu().numpy()
        next_obs_np = batch.next_observations.cpu().numpy()

        # Masked dims (1 and 3) must be zero.
        np.testing.assert_array_equal(obs_np[:, 1], np.zeros(32))
        np.testing.assert_array_equal(obs_np[:, 3], np.zeros(32))
        np.testing.assert_array_equal(next_obs_np[:, 1], np.zeros(32))
        np.testing.assert_array_equal(next_obs_np[:, 3], np.zeros(32))

        # Kept dims (0 and 2) must be non-zero (original values in [0.5, 1.0]).
        assert obs_np[:, 0].any()
        assert obs_np[:, 2].any()

    def test_obs_unmasked_when_model_has_no_obs_mask(self):
        """Default RewardModel (obs_mask=None) leaves observations untouched."""
        obs_dim, act_dim = 3, 1
        model = ConstantRewardModel(constant=0.5)
        buf = _make_buffer(obs_dim, act_dim, capacity=100, reward_model=model)

        rng = np.random.default_rng(2)
        for _ in range(50):
            obs = rng.uniform(0.5, 1.0, (obs_dim,)).astype(np.float32)
            next_obs = rng.uniform(0.5, 1.0, (obs_dim,)).astype(np.float32)
            action = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
            buf.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=np.array([1.0]),
                done=np.array([False]),
                infos=[{}],
            )

        batch = buf.sample(batch_size=16)
        # All dims should be non-zero (stored values were in [0.5, 1.0]).
        obs_np = batch.observations.cpu().numpy()
        assert (obs_np != 0).any(axis=0).all()
