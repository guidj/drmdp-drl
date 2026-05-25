from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
import pytest

from drmdp import core


class FakeEnv(gym.Env):
    """Minimal gym environment for testing EnvMonitorWrapper."""

    def __init__(self, obs_dim: int = 2):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._reward: float = 1.0
        self._term: bool = False
        self._trunc: bool = False

    def step(self, action: Any):
        obs = self.observation_space.sample()
        return obs, self._reward, self._term, self._trunc, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping] = None):
        obs = self.observation_space.sample()
        return obs, {}


class TestSeeder:
    def test_get_seed_returns_none_when_instance_is_none(self):
        seeder = core.Seeder(instance=None)
        assert seeder.get_seed(episode=0) is None

    def test_get_seed_deterministic_for_same_inputs(self):
        seeder = core.Seeder(instance=42)
        seed_first = seeder.get_seed(episode=5)
        seed_second = seeder.get_seed(episode=5)
        assert seed_first == seed_second

    def test_get_seed_different_episodes_produce_different_seeds(self):
        seeder = core.Seeder(instance=1)
        seed_ep0 = seeder.get_seed(episode=0)
        seed_ep1 = seeder.get_seed(episode=1)
        assert seed_ep0 != seed_ep1

    def test_get_seed_different_instances_produce_different_seeds(self):
        seeder_a = core.Seeder(instance=0)
        seeder_b = core.Seeder(instance=1)
        assert seeder_a.get_seed(episode=0) != seeder_b.get_seed(episode=0)

    def test_get_seed_cantor_pairing_formula(self):
        seeder = core.Seeder(instance=3)
        episode = 7
        k1, k2 = 3, episode
        expected = int(((k1 + k2) * (k1 + k2 + 1)) / 2 + k2)
        assert seeder.get_seed(episode=episode) == expected


class TestEnvMonitor:
    def test_initial_state(self):
        mon = core.EnvMonitor()
        assert mon.returns == []
        assert mon.steps == []
        assert mon.rewards == 0
        assert mon.step == 0

    def test_reset_with_no_steps_does_not_append(self):
        mon = core.EnvMonitor()
        mon.reset()
        assert mon.returns == []
        assert mon.steps == []

    def test_reset_appends_rewards_and_steps(self):
        mon = core.EnvMonitor()
        mon.rewards = 5.0
        mon.step = 3
        mon.reset()
        assert mon.returns == [5.0]
        assert mon.steps == [3]

    def test_reset_clears_accumulator(self):
        mon = core.EnvMonitor()
        mon.rewards = 5.0
        mon.step = 3
        mon.reset()
        assert mon.rewards == 0.0
        assert mon.step == 0

    def test_clear_empties_all_state(self):
        mon = core.EnvMonitor()
        mon.returns = [1.0, 2.0]
        mon.steps = [10, 20]
        mon.rewards = 3.0
        mon.step = 5
        mon.clear()
        assert mon.returns == []
        assert mon.steps == []
        assert mon.rewards == 0.0
        assert mon.step == 0

    def test_multiple_episodes_tracked_independently(self):
        mon = core.EnvMonitor()
        mon.rewards = 10.0
        mon.step = 5
        mon.reset()
        mon.rewards = 20.0
        mon.step = 8
        mon.reset()
        assert mon.returns == [10.0, 20.0]
        assert mon.steps == [5, 8]


class TestEnvMonitorWrapper:
    @pytest.fixture()
    def wrapped_env(self):
        return core.EnvMonitorWrapper(FakeEnv())

    def test_step_accumulates_reward(self, wrapped_env):
        wrapped_env.reset()
        _, reward, _, _, _ = wrapped_env.step(wrapped_env.action_space.sample())
        assert wrapped_env.mon.rewards == reward

    def test_step_injects_info_on_termination(self, wrapped_env):
        wrapped_env.reset()
        wrapped_env.env._reward = 3.0
        wrapped_env.env._term = True
        _, _, term, _, info = wrapped_env.step(wrapped_env.action_space.sample())
        assert term is True
        assert "true_episode_return" in info
        assert "true_episode_steps" in info
        assert info["true_episode_return"] == 3.0
        assert info["true_episode_steps"] == 1

    def test_step_injects_info_on_truncation(self, wrapped_env):
        wrapped_env.reset()
        wrapped_env.env._trunc = True
        _, _, _, trunc, info = wrapped_env.step(wrapped_env.action_space.sample())
        assert trunc is True
        assert "true_episode_return" in info
        assert "true_episode_steps" in info

    def test_reset_clears_monitor_accumulator(self, wrapped_env):
        wrapped_env.reset()
        wrapped_env.env._reward = 5.0
        wrapped_env.step(wrapped_env.action_space.sample())
        assert wrapped_env.mon.rewards == 5.0
        wrapped_env.reset()
        assert wrapped_env.mon.rewards == 0.0
        assert wrapped_env.mon.step == 0


class TestArgChain:
    def test_empty_layers_returns_default(self):
        chain = core.ArgChain([])
        assert chain.get("k", 7) == 7

    def test_single_layer_found(self):
        chain = core.ArgChain([{"k": 42}])
        assert chain.get("k") == 42

    def test_single_layer_not_found_returns_default(self):
        chain = core.ArgChain([{"a": 1}])
        assert chain.get("b", 99) == 99

    def test_default_is_none_when_omitted(self):
        chain = core.ArgChain([{"a": 1}])
        assert chain.get("missing") is None

    def test_first_layer_wins(self):
        chain = core.ArgChain([{"k": "first"}, {"k": "second"}])
        assert chain.get("k") == "first"

    def test_falls_through_to_later_layer(self):
        chain = core.ArgChain([{"a": 1}, {"b": 2}])
        assert chain.get("b") == 2

    def test_none_value_is_valid(self):
        chain = core.ArgChain([{"k": None}, {"k": "fallback"}])
        assert chain.get("k") is None

    def test_prepend_gives_higher_priority(self):
        chain = core.ArgChain([{"k": "base"}])
        new_chain = chain.prepend([{"k": "override"}])
        assert new_chain.get("k") == "override"

    def test_prepend_does_not_mutate_original(self):
        chain = core.ArgChain([{"k": "base"}])
        chain.prepend([{"k": "override"}])
        assert chain.get("k") == "base"

    def test_extend_appends_lower_priority(self):
        chain = core.ArgChain([{"k": "original"}])
        new_chain = chain.extend([{"k": "extended"}])
        assert new_chain.get("k") == "original"

    def test_extend_does_not_mutate_original(self):
        chain = core.ArgChain([{"k": "base"}])
        chain.extend([{"k": "extended"}])
        assert chain.get("k") == "base"

    def test_pinned_cli_wins_over_prepended_layer(self):
        cli = {"k": "cli"}
        base = {"k": "base"}
        override = {"k": "override"}
        chain = core.ArgChain.pinned([cli], [base])
        new_chain = chain.prepend([override])
        assert new_chain.get("k") == "cli"

    def test_prepend_with_pin_inserts_below_pinned(self):
        cli = {"k": "cli"}
        base = {"k": "base"}
        new_layer = {"other": "new"}
        chain = core.ArgChain.pinned([cli], [base])
        new_chain = chain.prepend([new_layer])
        assert new_chain.get("other") == "new"
        assert new_chain.get("k") == "cli"

    def test_extend_preserves_pin(self):
        chain = core.ArgChain.pinned([{"a": 1}], [{"b": 2}])
        extended = chain.extend([{"c": 3}])
        assert extended._pin == chain._pin

    def test_prepend_preserves_pin(self):
        chain = core.ArgChain.pinned([{"a": 1}], [{"b": 2}])
        prepended = chain.prepend([{"c": 3}])
        assert prepended._pin == chain._pin
