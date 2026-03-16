import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from drmdp import rewdelay



class DummyEnv(gym.Env):
    """
    Terminates on `term_steps`.
    The observation is a vector with the
    step count, with possible values {-1, 1, 2, ... term_steps}
    """

    def __init__(self, term_steps: int = 2):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = spaces.Discrete(3)
        self.step_count = 0
        self.term_steps = term_steps

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.ones(3) * self.step_count
        reward = 1.0

        terminated = self.step_count >= self.term_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        del seed
        del options
        self.step_count = 0
        return np.ones(3) * -1, {}



def test_delayed_reward_wrapper_with_fixed_delay_init():
    env = DummyEnv()
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

    assert isinstance(wrapped.reward_delay, rewdelay.FixedDelay)
    assert wrapped.reward_delay.delay == 2
    assert wrapped.observation_space == env.observation_space
    assert wrapped.action_space == env.action_space
    assert wrapped.segment is None
    assert wrapped.segment_step is None
    assert wrapped.delay is None
    assert wrapped.op(range(10)) == sum(range(10))


def test_delayed_reward_wrapper_with_fixed_delay_step():
    env = DummyEnv(term_steps=4)
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

    obs, info = wrapped.reset()
    np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
    assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}

    # Seg 1, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 0, "next_delay": None}

    # Ep 1, Seg 1, Step 2
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([2, 2, 2]))
    assert (reward, term, trunc) == (2.0, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 1, "next_delay": 2}

    # Ep 1, Seg 2, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([3, 3, 3]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 1, "segment_step": 0, "next_delay": None}

    # Ep 1, Seg 2, Step 2
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([4, 4, 4]))
    assert (reward, term, trunc) == (2.0, True, False)
    assert info == {"delay": 2, "segment": 1, "segment_step": 1, "next_delay": 2}

    # Reset after termination
    obs, info = wrapped.reset()
    np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
    assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}

    # Ep 2, Seg 1, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 0, "next_delay": None}


def test_delayed_reward_wrapper_with_fixed_delay_reset():
    env = DummyEnv()
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

    wrapped.reset()
    # Seg 1, Step 1
    wrapped.step(0)
    assert wrapped.segment == 0
    assert wrapped.segment_step == 0
    assert wrapped.rewards == [1]

    # Seg 1, Step 2
    # sement info has been reset
    # for the next step
    wrapped.step(0)
    assert wrapped.segment == 1
    assert wrapped.segment_step == -1
    assert not wrapped.rewards

    # Seg 2, Step 1
    wrapped.step(0)
    assert wrapped.segment == 1
    assert wrapped.segment_step == 0
    assert wrapped.rewards == [1]

    # Reset
    wrapped.reset()
    assert wrapped.segment == 0
    assert wrapped.segment_step == -1
    assert not wrapped.rewards


def test_delayed_reward_wrapper_with_poisson_delay_init():
    env = DummyEnv()
    wrapped = rewdelay.DelayedRewardWrapper(
        env, reward_delay=rewdelay.ClippedPoissonDelay(2, min_delay=2, max_delay=5)
    )

    assert isinstance(wrapped.reward_delay, rewdelay.ClippedPoissonDelay)
    assert wrapped.reward_delay.lam == 2
    assert wrapped.observation_space == env.observation_space
    assert wrapped.action_space == env.action_space
    assert wrapped.segment is None
    assert wrapped.segment_step is None
    assert wrapped.delay is None
    assert wrapped.op(range(10)) == sum(range(10))


def test_delayed_reward_wrapper_with_poisson_delay_step(monkeypatch):
    class MockPoissonRng:
        def __init__(self, return_value: int):
            self.return_value = return_value

        def poisson(self, lam: int):
            del lam
            return self.return_value

    env = DummyEnv(term_steps=4)
    reward_delay = rewdelay.ClippedPoissonDelay(2, min_delay=2, max_delay=5)
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

    # Delay of 3
    monkeypatch.setattr(reward_delay, "rng", MockPoissonRng(3))

    wrapped.reset()
    # Ep 1, Seg 1, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 3, "segment": 0, "segment_step": 0, "next_delay": None}

    # # Ep 1, Seg 1, Step 2
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([2, 2, 2]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 3, "segment": 0, "segment_step": 1, "next_delay": None}

    # Override the sampler for the coming step
    monkeypatch.setattr(reward_delay, "rng", MockPoissonRng(2))

    # Ep 1, Seg 1, Step 3
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([3, 3, 3]))
    assert (reward, term, trunc) == (3, False, False)
    assert info == {"delay": 3, "segment": 0, "segment_step": 2, "next_delay": 2}

    # Ep 1, Seg 2, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([4, 4, 4]))
    assert (reward, term, trunc) == (None, True, False)
    assert info == {"delay": 2, "segment": 1, "segment_step": 0, "next_delay": None}

    # Reset after termination
    wrapped.reset()

    # Ep 2, Seg 1, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 0, "next_delay": None}

    # Ep 2, Seg 1, Step 2
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([2, 2, 2]))
    assert (reward, term, trunc) == (2.0, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 1, "next_delay": 2}

    # Ep 2, Seg 2, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([3, 3, 3]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 1, "segment_step": 0, "next_delay": None}


def test_data_buffer():
    buffer = rewdelay.DataBuffer()
    assert buffer.max_capacity is None
    assert buffer.max_size_bytes is None
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32

    buffer.add(1)
    buffer.add(3)
    buffer.add(5)
    buffer.add(7)
    assert buffer.buffer == [1, 1, 3, 5, 7]
    assert buffer.size() == 5
    assert buffer.size_bytes() == 64

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32


def test_data_buffer_max_capacity_with_latest_acc_mode():
    buffer = rewdelay.DataBuffer(
        max_capacity=10,
    )

    for value in range(100):
        buffer.add(value)
        assert buffer.size() <= 10
        # latest
        assert buffer.buffer == list(range(max(0, value - 10 + 1), value + 1))

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32


def test_data_buffer_max_capacity_with_first_acc_mode():
    buffer = rewdelay.DataBuffer(max_capacity=10, acc_mode="FIRST")

    for value in range(100):
        buffer.add(value)
        assert buffer.size() <= 10
        # first
        assert buffer.buffer == list(range(0, min(value + 1, 10)))

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32


def test_data_buffer_max_size_bytes_with_latest_acc_mode():
    buffer = rewdelay.DataBuffer(max_size_bytes=128)

    for value in range(100):
        buffer.add(value)
        assert buffer.size() > 0
        assert buffer.size_bytes() <= 128
        # latest
        assert buffer.buffer[-1] == value

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32


def test_data_buffer_max_size_bytes_with_first_acc_mode():
    buffer = rewdelay.DataBuffer(max_size_bytes=128, acc_mode="FIRST")

    for value in range(100):
        buffer.add(value)
        assert buffer.size() > 0
        assert buffer.size_bytes() <= 128
        # latest
        assert buffer.buffer[0] == 0

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32


def test_data_buffer_max_capacity_max_size_bytes_with_latest_acc_mode():
    buffer = rewdelay.DataBuffer(
        max_capacity=2,
        max_size_bytes=128,
    )

    for value in range(100):
        buffer.add(value)
        assert 0 < buffer.size() <= 2
        assert buffer.size_bytes() <= 128
        # latest
        assert buffer.buffer[-1] == value

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32
