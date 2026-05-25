import re
from typing import Any, Dict, Sequence
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from drmdp import envs

# ---------------------------------------------------------------------------
# TestWrapperInit
# ---------------------------------------------------------------------------


class TestWrapperInit:
    def test_mujoco_env_accepted(self) -> None:
        base_env = gym.make("InvertedPendulum-v5")
        wrapper = envs.UniversalNonStationaryMuJoCoWrapper(base_env)
        assert wrapper.mujoco_model is not None
        assert wrapper.original_mass is not None
        assert wrapper.original_friction is not None
        wrapper.close()

    def test_non_mujoco_env_raises_value_error(self) -> None:
        base_env = gym.make("Pendulum-v1")
        with pytest.raises(ValueError, match="model"):
            envs.UniversalNonStationaryMuJoCoWrapper(base_env)
        base_env.close()

    def test_original_values_are_copies(self) -> None:
        wrapper = _make_wrapped_env()
        assert wrapper.original_mass is not wrapper.mujoco_model.body_mass
        np.testing.assert_array_equal(
            wrapper.original_mass, wrapper.mujoco_model.body_mass
        )
        assert wrapper.original_friction is not wrapper.mujoco_model.geom_friction
        np.testing.assert_array_equal(
            wrapper.original_friction, wrapper.mujoco_model.geom_friction
        )
        wrapper.close()

    def test_gravity_disabled_stores_none(self) -> None:
        wrapper = _make_wrapped_env(change_gravity=False)
        assert wrapper.original_gravity is None
        wrapper.close()

    def test_mass_disabled_stores_none(self) -> None:
        wrapper = _make_wrapped_env(change_mass=False)
        assert wrapper.original_mass is None
        wrapper.close()

    def test_step_count_starts_at_zero(self) -> None:
        wrapper = _make_wrapped_env()
        assert wrapper.step_count == 0
        wrapper.close()


# ---------------------------------------------------------------------------
# TestParameterModification
# ---------------------------------------------------------------------------


class TestParameterModification:
    def test_reset_modifies_body_mass(self) -> None:
        wrapper = _make_wrapped_env(mass_variance=0.3)
        original = np.copy(wrapper.original_mass)
        wrapper.reset(seed=42)
        nonzero_mask = original > 0
        assert not np.allclose(
            wrapper.mujoco_model.body_mass[nonzero_mask],
            original[nonzero_mask],
        )
        wrapper.close()

    def test_reset_modifies_geom_friction(self) -> None:
        wrapper = _make_wrapped_env(friction_variance=0.3)
        original = np.copy(wrapper.original_friction)
        wrapper.reset(seed=42)
        nonzero_mask = original > 0
        assert not np.allclose(
            wrapper.mujoco_model.geom_friction[nonzero_mask],
            original[nonzero_mask],
        )
        wrapper.close()

    def test_step_modifies_gravity_x_axis(self) -> None:
        wrapper = _make_wrapped_env(
            change_gravity=True,
            gravity_drift_rate=0.1,
            max_gravity_shift=2.0,
        )
        wrapper.reset()
        original_gravity = np.copy(wrapper.original_gravity)
        action = wrapper.action_space.sample()
        wrapper.step(action)
        assert wrapper.mujoco_model.opt.gravity[0] != original_gravity[0]
        np.testing.assert_allclose(
            wrapper.mujoco_model.opt.gravity[1], original_gravity[1]
        )
        np.testing.assert_allclose(
            wrapper.mujoco_model.opt.gravity[2], original_gravity[2]
        )
        wrapper.close()

    def test_gravity_drift_follows_sinusoidal_formula(self) -> None:
        drift_rate = 0.05
        max_shift = 1.5
        wrapper = _make_wrapped_env(
            change_gravity=True,
            gravity_drift_rate=drift_rate,
            max_gravity_shift=max_shift,
        )
        wrapper.reset()
        original_gx = wrapper.original_gravity[0]
        action = wrapper.action_space.sample()
        for step_idx in range(1, 4):
            wrapper.step(action)
            expected_gx = original_gx + np.sin(step_idx * drift_rate) * max_shift
            np.testing.assert_allclose(
                wrapper.mujoco_model.opt.gravity[0], expected_gx, atol=1e-10
            )
        wrapper.close()

    def test_step_count_resets_on_reset(self) -> None:
        wrapper = _make_wrapped_env()
        wrapper.reset()
        action = wrapper.action_space.sample()
        for _ in range(3):
            wrapper.step(action)
        assert wrapper.step_count == 3
        wrapper.reset()
        assert wrapper.step_count == 0
        wrapper.close()

    def test_mass_noise_within_variance_bounds(self) -> None:
        variance = 0.2
        wrapper = _make_wrapped_env(mass_variance=variance)
        original = np.copy(wrapper.original_mass)
        wrapper.reset(seed=99)
        nonzero_mask = original > 0
        lower = original[nonzero_mask] * (1.0 - variance)
        upper = original[nonzero_mask] * (1.0 + variance)
        actual = wrapper.mujoco_model.body_mass[nonzero_mask]
        assert np.all(actual >= lower - 1e-12)
        assert np.all(actual <= upper + 1e-12)
        wrapper.close()


# ---------------------------------------------------------------------------
# TestTransitionsChange
# ---------------------------------------------------------------------------


class TestTransitionsChange:
    def test_different_mass_different_next_state(self) -> None:
        wrapper = _make_wrapped_env(
            env_id="InvertedPendulum-v5",
            change_mass=False,
            change_friction=False,
            change_gravity=False,
        )
        wrapper.reset(seed=0)
        qpos = np.copy(wrapper.unwrapped.data.qpos)
        qvel = np.copy(wrapper.unwrapped.data.qvel)
        action = np.zeros(wrapper.action_space.shape)

        obs_a, _, _, _, _ = wrapper.step(action)

        wrapper.unwrapped.set_state(qpos, qvel)
        wrapper.mujoco_model.body_mass[1:] *= 2.0
        obs_b, _, _, _, _ = wrapper.step(action)

        assert not np.allclose(obs_a, obs_b), (
            "Doubling body mass should produce a different next state"
        )
        wrapper.close()

    def test_different_friction_different_next_state(self) -> None:
        wrapper = _make_wrapped_env(
            env_id="HalfCheetah-v5",
            change_mass=False,
            change_friction=False,
            change_gravity=False,
        )
        wrapper.reset(seed=0)
        qpos = np.copy(wrapper.unwrapped.data.qpos)
        qvel = np.copy(wrapper.unwrapped.data.qvel)
        action = np.ones(wrapper.action_space.shape) * 0.5

        obs_a, _, _, _, _ = wrapper.step(action)

        wrapper.unwrapped.set_state(qpos, qvel)
        wrapper.mujoco_model.geom_friction[:] *= 0.1
        obs_b, _, _, _, _ = wrapper.step(action)

        assert not np.allclose(obs_a, obs_b), (
            "Reducing friction by 10x should produce a different next state"
        )
        wrapper.close()

    def test_different_variance_different_dynamics(self) -> None:
        """Same seed, same initial state, but different mass_variance configs
        produce different masses and therefore different next states.
        """
        env_id = "InvertedPendulum-v5"
        seed = 42
        action = np.array([0.0])

        wrapper_a = _make_wrapped_env(env_id=env_id, mass_variance=0.0)
        wrapper_a.reset(seed=seed)
        obs_a, _, _, _, _ = wrapper_a.step(action)
        mass_a = np.copy(wrapper_a.mujoco_model.body_mass)
        wrapper_a.close()

        wrapper_b = _make_wrapped_env(env_id=env_id, mass_variance=0.4)
        wrapper_b.reset(seed=seed)
        obs_b, _, _, _, _ = wrapper_b.step(action)
        mass_b = np.copy(wrapper_b.mujoco_model.body_mass)
        wrapper_b.close()

        assert not np.allclose(mass_a[1:], mass_b[1:]), (
            "variance=0 vs 0.4 should produce different mass configurations"
        )
        assert not np.allclose(obs_a, obs_b), (
            "Different mass configs should yield different next states"
        )


# ---------------------------------------------------------------------------
# TestRewardBehavior
# ---------------------------------------------------------------------------


class TestRewardBehavior:
    """Characterise how rewards change when dynamics parameters change.

    The wrapper never overrides the reward computation.  Rewards change
    *indirectly* because altered dynamics produce different states and
    velocities, which the underlying env's reward formula evaluates.
    """

    def test_halfcheetah_reward_changes_with_mass(self) -> None:
        """HalfCheetah reward = fwd_vel * w - ctrl_cost.
        Heavier mass -> less velocity for the same action -> different reward.
        """
        rewards = _collect_rewards_under_mass_multipliers(
            env_id="HalfCheetah-v5",
            mass_multipliers=[1.0, 3.0],
            action_value=1.0,
            num_steps=5,
        )
        assert not np.allclose(rewards[0], rewards[1]), (
            "HalfCheetah rewards should differ when mass triples"
        )

    def test_halfcheetah_ctrl_cost_invariant_to_mass(self) -> None:
        """ctrl_cost = weight * ||action||^2 -- depends only on action."""
        infos = _collect_infos_under_mass_multipliers(
            env_id="HalfCheetah-v5",
            mass_multipliers=[1.0, 3.0],
            action_value=0.7,
            num_steps=3,
        )
        for step_idx in range(3):
            np.testing.assert_allclose(
                infos[0][step_idx]["reward_ctrl"],
                infos[1][step_idx]["reward_ctrl"],
                atol=1e-10,
            )

    def test_halfcheetah_forward_reward_changes_with_mass(self) -> None:
        """forward_reward = weight * x_velocity -- velocity depends on dynamics."""
        infos = _collect_infos_under_mass_multipliers(
            env_id="HalfCheetah-v5",
            mass_multipliers=[1.0, 3.0],
            action_value=1.0,
            num_steps=5,
        )
        fwd_a = [info["reward_forward"] for info in infos[0]]
        fwd_b = [info["reward_forward"] for info in infos[1]]
        assert not np.allclose(fwd_a, fwd_b), (
            "Forward reward should change with mass (velocity differs)"
        )

    def test_inverted_pendulum_step_reward_always_one_while_upright(self) -> None:
        """InvertedPendulum reward = 1.0 per step while upright.
        The per-step reward is invariant to dynamics; only episode duration
        changes (the pole falls sooner or later).
        """
        wrapper = _make_wrapped_env(env_id="InvertedPendulum-v5", mass_variance=0.3)
        np.random.seed(77)
        wrapper.reset(seed=0)
        action = np.array([0.0])
        for _ in range(3):
            _, reward, terminated, _, _ = wrapper.step(action)
            if terminated:
                break
            assert reward == 1.0
        wrapper.close()

    def test_hopper_healthy_reward_constant_per_step(self) -> None:
        """Hopper healthy_reward = 1.0 per step while alive, regardless of
        dynamics parameters.
        """
        infos_list = _collect_infos_under_mass_multipliers(
            env_id="Hopper-v5",
            mass_multipliers=[1.0, 2.0],
            action_value=0.0,
            num_steps=3,
        )
        for infos in infos_list:
            for info in infos:
                if "reward_survive" in info:
                    np.testing.assert_allclose(info["reward_survive"], 1.0)

    def test_wrapper_is_transparent_to_reward(self) -> None:
        """Definitive proof: patch the inner env to return a sentinel reward
        and verify the wrapper passes it through unchanged.
        """
        wrapper = _make_wrapped_env(
            change_gravity=True, gravity_drift_rate=0.1, max_gravity_shift=2.0
        )
        wrapper.reset(seed=0)
        sentinel_reward = -12345.6789
        inner_step_return = (
            np.zeros(wrapper.observation_space.shape),
            sentinel_reward,
            False,
            False,
            {"sentinel": True},
        )
        with mock.patch.object(wrapper.env, "step", return_value=inner_step_return):
            _, reward, _, _, info = wrapper.step(wrapper.action_space.sample())
        assert reward == sentinel_reward
        assert info["sentinel"] is True
        wrapper.close()


# ---------------------------------------------------------------------------
# TestFactoryFunction
# ---------------------------------------------------------------------------


class TestFactoryFunction:
    @pytest.mark.parametrize(
        "env_id",
        ["InvertedPendulum-v5", "HalfCheetah-v5", "Reacher-v5"],
    )
    def test_factory_returns_wrapped_env(self, env_id: str) -> None:
        wrapped = envs.make_non_stationary_mujoco_env(env_id)
        assert isinstance(wrapped, envs.UniversalNonStationaryMuJoCoWrapper)
        wrapped.close()

    def test_factory_halfcheetah_config(self) -> None:
        wrapped = envs.make_non_stationary_mujoco_env("HalfCheetah-v5")
        assert wrapped.mass_variance == 0.2
        assert wrapped.friction_variance == 0.2
        assert wrapped.change_gravity is True
        assert wrapped.max_gravity_shift == 1.0
        wrapped.close()

    def test_factory_inverted_pendulum_config(self) -> None:
        wrapped = envs.make_non_stationary_mujoco_env("InvertedPendulum-v5")
        assert wrapped.mass_variance == 0.1
        assert wrapped.friction_variance == 0.1
        assert wrapped.change_gravity is False
        wrapped.close()

    def test_factory_non_mujoco_raises(self) -> None:
        with pytest.raises(ValueError, match="model"):
            envs.make_non_stationary_mujoco_env("Pendulum-v1")

    def test_version_stripping_regex(self) -> None:
        assert re.sub(r"-v\d+", "", "HalfCheetah-v5") == "HalfCheetah"
        assert re.sub(r"-v\d+", "", "InvertedPendulum-v5") == "InvertedPendulum"
        assert re.sub(r"-v\d+", "", "Hopper-v5") == "Hopper"


# ---------------------------------------------------------------------------
# TestClassicControlWrapper
# ---------------------------------------------------------------------------


class TestClassicControlWrapper:
    def test_non_classic_env_raises(self) -> None:
        base_env = gym.make("Pendulum-v1")
        with pytest.raises(ValueError, match="no_such_attr"):
            envs.ClassicControlNonStationaryWrapper(
                base_env, param_variances={"no_such_attr": 0.1}
            )
        base_env.close()

    def test_originals_stored(self) -> None:
        wrapped = envs.make_non_stationary_classic_env("Pendulum-v1")
        assert wrapped.originals["g"] == 10.0
        assert wrapped.originals["m"] == 1.0
        assert wrapped.originals["l"] == 1.0
        wrapped.close()

    @pytest.mark.parametrize(
        "env_id",
        [
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Acrobot-v1",
        ],
    )
    def test_factory_returns_wrapped_env(self, env_id: str) -> None:
        wrapped = envs.make_non_stationary_classic_env(env_id)
        assert isinstance(wrapped, envs.ClassicControlNonStationaryWrapper)
        wrapped.close()

    def test_reset_perturbs_parameters(self) -> None:
        wrapped = envs.make_non_stationary_classic_env("Pendulum-v1")
        wrapped.reset(seed=42)
        unwrapped = wrapped.env.unwrapped
        assert unwrapped.g != wrapped.originals["g"]
        assert unwrapped.m != wrapped.originals["m"]
        assert unwrapped.l != wrapped.originals["l"]
        wrapped.close()

    def test_different_seeds_different_params(self) -> None:
        wrapped = envs.make_non_stationary_classic_env("Pendulum-v1")
        wrapped.reset(seed=1)
        params_a = (wrapped.env.unwrapped.g, wrapped.env.unwrapped.m)
        wrapped.reset(seed=2)
        params_b = (wrapped.env.unwrapped.g, wrapped.env.unwrapped.m)
        assert params_a != params_b
        wrapped.close()

    def test_cartpole_derived_quantities_recomputed(self) -> None:
        wrapped = envs.make_non_stationary_classic_env("CartPole-v1")
        wrapped.reset(seed=42)
        unwrapped = wrapped.env.unwrapped
        np.testing.assert_allclose(
            unwrapped.total_mass,
            unwrapped.masscart + unwrapped.masspole,
        )
        np.testing.assert_allclose(
            unwrapped.polemass_length,
            unwrapped.masspole * unwrapped.length,
        )
        wrapped.close()

    def test_perturbation_within_bounds(self) -> None:
        wrapped = envs.make_non_stationary_classic_env("Pendulum-v1")
        wrapped.reset(seed=99)
        for attr_name, variance in wrapped.param_variances.items():
            original = wrapped.originals[attr_name]
            actual = getattr(wrapped.env.unwrapped, attr_name)
            lower = original * (1.0 - variance)
            upper = original * (1.0 + variance)
            assert lower - 1e-12 <= actual <= upper + 1e-12, (
                f"{attr_name}: {actual} not in [{lower}, {upper}]"
            )
        wrapped.close()

    def test_transitions_change_with_different_variance(self) -> None:
        env_id = "Pendulum-v1"
        seed = 42
        action = np.array([0.0])

        no_perturb = envs.ClassicControlNonStationaryWrapper(
            gym.make(env_id), param_variances={"g": 0.0, "m": 0.0, "l": 0.0}
        )
        no_perturb.reset(seed=seed)
        obs_a, _, _, _, _ = no_perturb.step(action)
        no_perturb.close()

        high_perturb = envs.ClassicControlNonStationaryWrapper(
            gym.make(env_id), param_variances={"g": 0.4, "m": 0.4, "l": 0.4}
        )
        high_perturb.reset(seed=seed)
        obs_b, _, _, _, _ = high_perturb.step(action)
        high_perturb.close()

        assert not np.allclose(obs_a, obs_b)

    def test_factory_unknown_env_raises(self) -> None:
        with pytest.raises(ValueError, match="No classic control config"):
            envs.make_non_stationary_classic_env("HalfCheetah-v5")

    def test_wrapper_transparent_to_reward(self) -> None:
        wrapped = envs.make_non_stationary_classic_env("Pendulum-v1")
        wrapped.reset(seed=0)
        sentinel = -99999.0
        mock_return: Any = (np.zeros(3), sentinel, False, False, {})
        with mock.patch.object(wrapped.env, "step", return_value=mock_return):
            _, reward, _, _, _ = wrapped.step(np.array([0.0]))
        assert reward == sentinel
        wrapped.close()


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_wrapped_env(
    env_id: str = "InvertedPendulum-v5",
    **wrapper_kwargs: Any,
) -> envs.UniversalNonStationaryMuJoCoWrapper:
    base_env = gym.make(env_id)
    defaults: Dict[str, Any] = {
        "change_mass": True,
        "mass_variance": 0.15,
        "change_friction": True,
        "friction_variance": 0.15,
        "change_gravity": False,
    }
    defaults.update(wrapper_kwargs)
    return envs.UniversalNonStationaryMuJoCoWrapper(base_env, **defaults)


def _collect_rewards_under_mass_multipliers(
    env_id: str,
    mass_multipliers: Sequence[float],
    action_value: float,
    num_steps: int,
) -> Sequence[Sequence[float]]:
    all_rewards: list[list[float]] = []
    for multiplier in mass_multipliers:
        wrapper = _make_wrapped_env(
            env_id=env_id,
            change_mass=False,
            change_friction=False,
            change_gravity=False,
        )
        wrapper.reset(seed=0)
        qpos = np.copy(wrapper.unwrapped.data.qpos)
        qvel = np.copy(wrapper.unwrapped.data.qvel)

        wrapper.mujoco_model.body_mass[1:] *= multiplier
        wrapper.unwrapped.set_state(qpos, qvel)

        action = np.full(wrapper.action_space.shape, action_value)
        rewards: list[float] = []
        for _ in range(num_steps):
            _, reward, terminated, truncated, _ = wrapper.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        all_rewards.append(rewards)
        wrapper.close()
    return all_rewards


def _collect_infos_under_mass_multipliers(
    env_id: str,
    mass_multipliers: Sequence[float],
    action_value: float,
    num_steps: int,
) -> Sequence[Sequence[Dict[str, Any]]]:
    all_infos: list[list[Dict[str, Any]]] = []
    for multiplier in mass_multipliers:
        wrapper = _make_wrapped_env(
            env_id=env_id,
            change_mass=False,
            change_friction=False,
            change_gravity=False,
        )
        wrapper.reset(seed=0)
        qpos = np.copy(wrapper.unwrapped.data.qpos)
        qvel = np.copy(wrapper.unwrapped.data.qvel)

        wrapper.mujoco_model.body_mass[1:] *= multiplier
        wrapper.unwrapped.set_state(qpos, qvel)

        action = np.full(wrapper.action_space.shape, action_value)
        infos: list[Dict[str, Any]] = []
        for _ in range(num_steps):
            _, _, terminated, truncated, info = wrapper.step(action)
            infos.append(info)
            if terminated or truncated:
                break
        all_infos.append(infos)
        wrapper.close()
    return all_infos
