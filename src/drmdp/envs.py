import re
from typing import Any, Callable, Mapping, Optional

import gymnasium as gym
import numpy as np


class UniversalNonStationaryMuJoCoWrapper(gym.Wrapper):
    """
    A universal Gymnasium wrapper for injecting non-stationarity into any MuJoCo v5 environment.
    Dynamically adapts to the number of bodies and geometries in the underlying model.
    """

    def __init__(
        self,
        env,
        change_mass=True,
        mass_variance=0.15,
        change_friction=True,
        friction_variance=0.15,
        change_gravity=True,
        gravity_drift_rate=0.01,
        max_gravity_shift=1.5,
    ):
        super().__init__(env)

        # Safety Check: Ensure the environment has the MuJoCo model attached
        if not hasattr(self.env.unwrapped, "model"):
            raise ValueError(
                "Environment missing 'model' attribute. Ensure it is a MuJoCo environment."
            )

        self.mujoco_model = self.env.unwrapped.model
        self.rng = np.random.default_rng()

        # Configuration Parameters
        self.change_mass = change_mass
        self.mass_variance = mass_variance

        self.change_friction = change_friction
        self.friction_variance = friction_variance

        self.change_gravity = change_gravity
        self.gravity_drift_rate = gravity_drift_rate
        self.max_gravity_shift = max_gravity_shift

        self.step_count = 0

        # Dynamically store original values based on the specific v5 model's dimensions
        self.original_mass = (
            np.copy(self.mujoco_model.body_mass) if change_mass else None
        )
        self.original_friction = (
            np.copy(self.mujoco_model.geom_friction) if change_friction else None
        )
        self.original_gravity = (
            np.copy(self.mujoco_model.opt.gravity) if change_gravity else None
        )

    def reset(self, *, seed=None, options=None):
        """Applies episodic (abrupt) non-stationarity at the start of each episode."""

        self.rng = np.random.default_rng(seed=seed)
        # Randomize Mass
        if self.change_mass:
            mass_noise = self.rng.uniform(
                1.0 - self.mass_variance,
                1.0 + self.mass_variance,
                size=self.original_mass.shape,
            )
            self.mujoco_model.body_mass[:] = self.original_mass * mass_noise

        # Randomize Friction (v5 features perfectly symmetric default frictions)
        if self.change_friction:
            friction_noise = self.rng.uniform(
                1.0 - self.friction_variance,
                1.0 + self.friction_variance,
                size=self.original_friction.shape,
            )
            self.mujoco_model.geom_friction[:] = self.original_friction * friction_noise

        self.step_count = 0

        # Pass seed and options correctly for Gymnasium v1.0+ compatibility
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Applies continuous (smooth) non-stationarity at every time step."""
        self.step_count += 1

        if self.change_gravity:
            # Oscillate horizontal gravity (X-axis) around its original base value
            drift = (
                np.sin(self.step_count * self.gravity_drift_rate)
                * self.max_gravity_shift
            )
            self.mujoco_model.opt.gravity[0] = self.original_gravity[0] + drift

        # Returns the standard Gymnasium tuple: obs, reward, terminated, truncated, info
        return self.env.step(action)


def make_non_stationary_mujoco_env(env_id, render_mode=None, **kwargs):
    """
    Creates a Gymnasium MuJoCo v5 environment and wraps it with environment-specific
    non-stationary parameters. Fully supports v5 custom kwargs (e.g., xml_file).

    Args:
        env_id (str): The Gymnasium environment ID (e.g., 'HalfCheetah-v5').
        render_mode (str): Render mode for the environment.
        **kwargs: Additional arguments for gym.make() (perfect for v5's xml_file or reset_noise_scale).

    Returns:
        gym.Env: The wrapped, non-stationary environment.
    """

    # Strip the version number to match the base environment name
    base_env_name = re.sub(r"-v\d+", "", env_id)

    # Define heuristic configurations based on physical stability limits for v5
    configs = {
        # Highly stable crawlers/quadrupeds
        "Ant": {
            "mass_variance": 0.2,
            "friction_variance": 0.2,
            "change_gravity": True,
            "max_gravity_shift": 1.0,
        },
        "HalfCheetah": {
            "mass_variance": 0.2,
            "friction_variance": 0.2,
            "change_gravity": True,
            "max_gravity_shift": 1.0,
        },
        "Swimmer": {
            "mass_variance": 0.2,
            "friction_variance": 0.2,
            "change_gravity": True,
            "max_gravity_shift": 0.5,
        },
        # Bipedal/Monopedal hoppers
        "Hopper": {
            "mass_variance": 0.1,
            "friction_variance": 0.1,
            "change_gravity": True,
            "max_gravity_shift": 0.3,
        },
        "Walker2d": {
            "mass_variance": 0.1,
            "friction_variance": 0.1,
            "change_gravity": True,
            "max_gravity_shift": 0.3,
        },
        # Highly complex/unstable balance
        "Humanoid": {
            "mass_variance": 0.05,
            "friction_variance": 0.05,
            "change_gravity": False,
        },
        "HumanoidStandup": {
            "mass_variance": 0.05,
            "friction_variance": 0.05,
            "change_gravity": False,
        },
        # Strict balancing tasks
        "InvertedPendulum": {
            "mass_variance": 0.1,
            "friction_variance": 0.1,
            "change_gravity": False,
        },
        "InvertedDoublePendulum": {
            "mass_variance": 0.05,
            "friction_variance": 0.05,
            "change_gravity": False,
        },
        # Pinned manipulation tasks (Pusher-v5 has corrected mass values, making it much more stable)
        "Pusher": {
            "mass_variance": 0.15,
            "friction_variance": 0.15,
            "change_gravity": True,
            "max_gravity_shift": 0.3,
        },
        "Reacher": {
            "mass_variance": 0.15,
            "friction_variance": 0.15,
            "change_gravity": True,
            "max_gravity_shift": 0.3,
        },
    }

    # Default fallback for unrecognized or custom MuJoCo environments
    default_config = {
        "mass_variance": 0.1,
        "friction_variance": 0.1,
        "change_gravity": False,
    }

    env_config = configs.get(base_env_name, default_config)

    # Create the v5 base environment, passing through any specific kwargs
    base_env = gym.make(env_id, render_mode=render_mode, **kwargs)

    # Wrap it
    wrapped_env = UniversalNonStationaryMuJoCoWrapper(
        base_env,
        change_mass=True,
        mass_variance=env_config.get("mass_variance"),
        change_friction=True,
        friction_variance=env_config.get("friction_variance"),
        change_gravity=env_config.get("change_gravity"),
        gravity_drift_rate=0.01,
        max_gravity_shift=env_config.get("max_gravity_shift", 0.0),
    )

    return wrapped_env


# ---------------------------------------------------------------------------
# Classic control non-stationarity
# ---------------------------------------------------------------------------


class ClassicControlNonStationaryWrapper(gym.Wrapper):
    """Gym wrapper that injects episodic non-stationarity into classic control envs.

    On each ``reset``, physics constants on the unwrapped env are perturbed
    with multiplicative uniform noise around their original values.  An
    optional ``recompute`` callback updates derived quantities (e.g.
    CartPole's ``total_mass``) after the primary attributes change.
    """

    def __init__(
        self,
        env: gym.Env,
        param_variances: Mapping[str, float],
        recompute: Optional[Callable[[gym.Env], None]] = None,
    ):
        super().__init__(env)
        self.param_variances = dict(param_variances)
        self._recompute = recompute
        self.rng = np.random.default_rng()

        unwrapped = self.env.unwrapped
        self.originals: dict[str, float] = {}
        for attr_name in self.param_variances:
            if not hasattr(unwrapped, attr_name):
                raise ValueError(
                    f"{type(unwrapped).__name__} has no attribute {attr_name!r}"
                )
            self.originals[attr_name] = float(getattr(unwrapped, attr_name))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ):
        self.rng = np.random.default_rng(seed=seed)
        unwrapped = self.env.unwrapped
        for attr_name, variance in self.param_variances.items():
            noise = self.rng.uniform(1.0 - variance, 1.0 + variance)
            setattr(unwrapped, attr_name, self.originals[attr_name] * noise)
        if self._recompute is not None:
            self._recompute(unwrapped)
        return self.env.reset(seed=seed, options=options)


def _recompute_cartpole(unwrapped: Any) -> None:
    unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
    unwrapped.polemass_length = unwrapped.masspole * unwrapped.length


_CLASSIC_CONTROL_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "CartPole": {
        "param_variances": {
            "gravity": 0.1,
            "masscart": 0.15,
            "masspole": 0.15,
            "length": 0.1,
            "force_mag": 0.1,
        },
        "recompute": _recompute_cartpole,
    },
    "Pendulum": {
        "param_variances": {
            "g": 0.1,
            "m": 0.15,
            "l": 0.1,
        },
    },
    "MountainCar": {
        "param_variances": {
            "force": 0.15,
            "gravity": 0.15,
        },
    },
    "MountainCarContinuous": {
        "param_variances": {
            "power": 0.15,
        },
    },
    "Acrobot": {
        "param_variances": {
            "LINK_MASS_1": 0.15,
            "LINK_MASS_2": 0.15,
            "LINK_LENGTH_1": 0.1,
            "LINK_LENGTH_2": 0.1,
            "LINK_COM_POS_1": 0.1,
            "LINK_COM_POS_2": 0.1,
        },
    },
}


def make_non_stationary_classic_env(
    env_id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> ClassicControlNonStationaryWrapper:
    """Create a classic control environment with episodic non-stationarity.

    Args:
        env_id: Gymnasium environment ID (e.g. ``'Pendulum-v1'``).
        render_mode: Render mode forwarded to ``gym.make``.
        **kwargs: Additional keyword arguments forwarded to ``gym.make``.

    Returns:
        The wrapped environment.

    Raises:
        ValueError: If the environment is not a recognised classic control env.
    """
    base_env_name = re.sub(r"-v\d+", "", env_id)
    config = _CLASSIC_CONTROL_CONFIGS.get(base_env_name)
    if config is None:
        raise ValueError(
            f"No classic control config for {env_id!r}. "
            f"Known: {sorted(_CLASSIC_CONTROL_CONFIGS)}"
        )
    base_env = gym.make(env_id, render_mode=render_mode, **kwargs)
    return ClassicControlNonStationaryWrapper(
        base_env,
        param_variances=config["param_variances"],
        recompute=config.get("recompute"),
    )
