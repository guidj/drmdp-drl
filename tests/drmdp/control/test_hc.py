"""
Tests for drmdp.control.hc: IntervalPositionWrapper, HCReplayBuffer, HCSACPolicy, HCSAC.
"""

import copy
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from drmdp import rewdelay
from drmdp.control import hc

# ---------------------------------------------------------------------------
# TestIntervalPositionWrapper
# ---------------------------------------------------------------------------


class TestIntervalPositionWrapper:
    """Tests for IntervalPositionWrapper."""

    def _make_wrapped_env(self, max_delay: int = 4) -> hc.IntervalPositionWrapper:
        """Return env wrapped with delayed reward wrappers and IntervalPositionWrapper."""
        env = gym.make("Pendulum-v1")
        env = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(max_delay))
        env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)
        return hc.IntervalPositionWrapper(env, max_delay=max_delay)

    def test_observation_space_dim_augmented(self):
        """obs_space.shape[-1] is one larger than the underlying env's obs dim."""
        max_delay = 4
        env = self._make_wrapped_env(max_delay)
        base_env = gym.make("Pendulum-v1")
        assert env.observation_space.shape[0] == base_env.observation_space.shape[0] + 1

    def test_position_zero_at_reset(self):
        """First obs after reset() has position 0.0 appended."""
        env = self._make_wrapped_env(max_delay=4)
        obs, _ = env.reset()
        assert obs[-1] == 0.0

    def test_position_increments_within_interval(self):
        """Position grows step by step within an interval (no reward emitted)."""
        max_delay = 5
        env = self._make_wrapped_env(max_delay)
        env.reset()
        for expected_pos in range(1, max_delay):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            position = obs[-1]
            if position == 0.0:
                # Interval boundary occurred; next tests not meaningful in this step.
                break
            np.testing.assert_allclose(position, expected_pos / max_delay, atol=1e-6)

    def test_position_resets_after_interval_end(self):
        """Position of the returned obs is 0 immediately after an interval ends."""
        max_delay = 3
        env = self._make_wrapped_env(max_delay)
        env.reset()
        # Drive exactly to interval end (steps 1 .. max_delay).
        for step_idx in range(1, max_delay):
            env.step(env.action_space.sample())
        # Final step of the interval: returned obs belongs to next interval.
        obs, _, _, _, _ = env.step(env.action_space.sample())
        np.testing.assert_allclose(obs[-1], 0.0, atol=1e-6)

    def test_position_resets_after_done(self):
        """Position of the first obs in a new episode is 0."""
        env = self._make_wrapped_env(max_delay=4)
        env.reset()
        obs, _ = env.reset()
        assert obs[-1] == 0.0


# ---------------------------------------------------------------------------
# TestHCReplayBuffer
# ---------------------------------------------------------------------------


class TestHCReplayBuffer:
    """Tests for HCReplayBuffer."""

    def test_history_zeros_at_first_transition(self):
        """The very first transition has all-zero history."""
        obs_dim, act_dim, max_delay = 3, 2, 4
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        _add_transition(buf, obs_dim, act_dim)
        history = buf._history_sa[0, 0]
        np.testing.assert_array_equal(history, 0.0)

    def test_history_accumulates_within_interval(self):
        """History grows step by step during an interval (no boundary)."""
        obs_dim, act_dim, max_delay = 2, 1, 4
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        buf = hc.HCReplayBuffer(
            buffer_size=200,
            observation_space=obs_space,
            action_space=act_space,
            max_delay=max_delay,
        )
        rng = np.random.default_rng(0)
        for _ in range(3):
            obs = rng.uniform(-1, 1, (1, obs_dim)).astype(np.float32)
            act = rng.uniform(-1, 1, (1, act_dim)).astype(np.float32)
            buf.add(
                obs,
                obs,
                act,
                np.array([0.0]),
                np.array([False]),
                [{"interval_end": False}],
            )
        # At position 2 the history should contain exactly 2 non-zero rows.
        history_step2 = buf._history_sa[2, 0]  # (max_delay, sa_dim)
        n_nonzero_rows = int(np.any(history_step2 != 0.0, axis=-1).sum())
        assert n_nonzero_rows == 2

    def test_history_resets_after_done(self):
        """After done=True the next transition has all-zero history."""
        obs_dim, act_dim, max_delay = 3, 2, 4
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        _add_transition(buf, obs_dim, act_dim, interval_end=False)
        _add_transition(buf, obs_dim, act_dim, done=True)
        _add_transition(buf, obs_dim, act_dim, interval_end=False)
        history_after_done = buf._history_sa[2, 0]
        np.testing.assert_array_equal(history_after_done, 0.0)

    def test_history_resets_after_interval_end(self):
        """After interval_end=True the next transition has all-zero history."""
        obs_dim, act_dim, max_delay = 3, 2, 4
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        _add_transition(buf, obs_dim, act_dim, interval_end=False)
        _add_transition(buf, obs_dim, act_dim, interval_end=True)
        _add_transition(buf, obs_dim, act_dim, interval_end=False)
        history_after_end = buf._history_sa[2, 0]
        np.testing.assert_array_equal(history_after_end, 0.0)

    def test_interval_ends_stored_correctly(self):
        """Stored interval_end flags match what was passed to add()."""
        obs_dim, act_dim, max_delay = 3, 2, 4
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        _add_transition(buf, obs_dim, act_dim, interval_end=False)
        _add_transition(buf, obs_dim, act_dim, interval_end=True)
        _add_transition(buf, obs_dim, act_dim, interval_end=False)
        assert buf._interval_ends[0, 0, 0] == 0.0
        assert buf._interval_ends[1, 0, 0] == 1.0
        assert buf._interval_ends[2, 0, 0] == 0.0

    def test_history_shape(self):
        """Sampled history has shape (batch, max_delay, obs_dim + act_dim)."""
        obs_dim, act_dim, max_delay, batch_size = 3, 2, 4, 8
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        for _ in range(20):
            _add_transition(buf, obs_dim, act_dim)
        samples = buf.sample(batch_size)
        assert samples.history.shape == (batch_size, max_delay, obs_dim + act_dim)

    def test_interval_ends_shape(self):
        """Sampled interval_ends has shape (batch, 1)."""
        obs_dim, act_dim, max_delay, batch_size = 3, 2, 4, 8
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        for _ in range(20):
            _add_transition(buf, obs_dim, act_dim)
        samples = buf.sample(batch_size)
        assert samples.interval_ends.shape == (batch_size, 1)

    def test_get_samples_returns_hc_replay_buffer_samples(self):
        """sample() returns HCReplayBufferSamples."""
        obs_dim, act_dim, max_delay = 3, 2, 4
        buf = _make_hc_buffer(obs_dim, act_dim, max_delay)
        for _ in range(20):
            _add_transition(buf, obs_dim, act_dim)
        samples = buf.sample(4)
        assert isinstance(samples, hc.HCReplayBufferSamples)


# ---------------------------------------------------------------------------
# TestHCSACPolicy
# ---------------------------------------------------------------------------


class TestHCSACPolicy:
    """Tests for HCSACPolicy attribute existence after _setup_model."""

    def _make_agent(self) -> hc.HCSAC:
        env = _make_box_env()
        return hc.HCSAC(env, max_delay=3, verbose=0)

    def test_policy_has_history_encoder(self):
        agent = self._make_agent()
        assert hasattr(agent.policy, "history_encoder")
        assert isinstance(agent.policy.history_encoder, torch.nn.Module)

    def test_policy_has_head_net(self):
        agent = self._make_agent()
        assert hasattr(agent.policy, "head_net")
        assert isinstance(agent.policy.head_net, torch.nn.Module)

    def test_head_optimizer_exists(self):
        agent = self._make_agent()
        assert hasattr(agent.policy, "head_optimizer")
        assert isinstance(agent.policy.head_optimizer, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# TestHCSAC
# ---------------------------------------------------------------------------


class TestHCSAC:
    """Integration tests for HCSAC."""

    def _make_hc_env(self, delay: int = 3) -> hc.IntervalPositionWrapper:
        env = gym.make("Pendulum-v1")
        env = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay))
        env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)
        return hc.IntervalPositionWrapper(env, max_delay=delay)

    def test_learn_completes_without_error(self):
        """Short training run completes without raising."""
        env = self._make_hc_env(delay=3)
        agent = hc.HCSAC(
            env,
            max_delay=3,
            reg_lambda=5.0,
            verbose=0,
            buffer_size=500,
            batch_size=32,
            learning_starts=50,
        )
        agent.learn(total_timesteps=200)

    def test_target_networks_soft_updated_after_train(self):
        """head_net_target parameters differ from a pre-train deepcopy."""
        env = self._make_hc_env(delay=3)
        agent = hc.HCSAC(
            env,
            max_delay=3,
            reg_lambda=5.0,
            verbose=0,
            buffer_size=500,
            batch_size=32,
            learning_starts=50,
        )
        # Snapshot targets before any training.
        head_target_before = copy.deepcopy(list(agent.head_net_target.parameters()))
        agent.learn(total_timesteps=300)
        # At least one parameter must have changed.
        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(
                head_target_before, agent.head_net_target.parameters()
            )
        )
        assert any_changed, "head_net_target parameters were not updated"

    def test_reg_lambda_zero_does_not_crash(self):
        """Training with reg_lambda=0.0 (no regularisation) completes."""
        env = self._make_hc_env(delay=3)
        agent = hc.HCSAC(
            env,
            max_delay=3,
            reg_lambda=0.0,
            verbose=0,
            buffer_size=500,
            batch_size=32,
            learning_starts=50,
        )
        agent.learn(total_timesteps=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_box_env(obs_dim: int = 3, act_dim: int = 2) -> gym.Env:
    """Minimal Box env — used for buffer / policy construction, not rollouts."""

    class _BoxEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
            )

        def step(self, action):
            del action
            return (
                self.observation_space.sample(),
                0.0,
                False,
                False,
                {"interval_end": False},
            )

        def reset(self, seed=None, options=None):
            del seed, options
            return self.observation_space.sample(), {}

    return _BoxEnv()


def _make_hc_buffer(
    obs_dim: int = 3,
    act_dim: int = 2,
    max_delay: int = 4,
    capacity: int = 200,
) -> hc.HCReplayBuffer:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    return hc.HCReplayBuffer(
        buffer_size=capacity,
        observation_space=obs_space,
        action_space=act_space,
        max_delay=max_delay,
    )


def _add_transition(
    buf: hc.HCReplayBuffer,
    obs_dim: int,
    act_dim: int,
    interval_end: bool = False,
    done: bool = False,
    reward: float = 0.0,
) -> None:
    obs = np.zeros((1, obs_dim), dtype=np.float32)
    act = np.zeros((1, act_dim), dtype=np.float32)
    infos: List[Dict[str, Any]] = [{"interval_end": interval_end}]
    buf.add(obs, obs, act, np.array([reward]), np.array([done]), infos)
