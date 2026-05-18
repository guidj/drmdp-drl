"""
This module defines core abstractions and types.
"""

import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

NestedArray = Union[Mapping, np.ndarray]
TimeStep = Tuple[ObsType, SupportsFloat, bool, bool, Mapping[str, Any]]
InitState = Tuple[ObsType, Mapping[str, Any]]
RenderType = Optional[Union[RenderFrame, Sequence[RenderFrame]]]
StateTransition = Mapping[int, Sequence[Tuple[float, int, float, bool]]]
# Type: Mapping[state, Mapping[action, Sequence[Tuple[prob, next_state, reward, terminated]]]]
EnvTransition = Mapping[int, StateTransition]
MutableStateTransition = Dict[int, List[Tuple[float, int, float, bool]]]
MutableEnvTransition = Dict[int, MutableStateTransition]
MapsToIntId = Callable[[Any], int]


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    """
    Output of a policy's action function.
    Encapsulates the chosen action and policy state.
    """

    action: ActType
    state: Any
    info: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Configuration parameters for an experiment.
    """

    name: str
    args: Optional[Mapping[str, Any]]
    feats_spec: Sequence[Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class ProblemSpec:
    """
    Configuration for delayed, aggregate (and anonymous) reward experiments.
    """

    policy_type: str
    reward_mapper: Mapping[str, Any]
    delay_config: Optional[Mapping[str, Any]]
    epsilon: float
    gamma: float
    learning_rate_config: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """
    Configuration for experiment run.
    """

    num_runs: int
    episodes_per_run: int
    log_episode_frequency: int
    use_seed: bool
    output_dir: str


@dataclasses.dataclass(frozen=True)
class Experiment:
    """
    Experiments definition.
    """

    env_spec: EnvSpec
    problem_spec: ProblemSpec
    epochs: int


@dataclasses.dataclass(frozen=True)
class ExperimentInstance:
    """
    A single experiment task.
    """

    exp_id: str
    instance_id: int
    experiment: Experiment
    run_config: RunConfig
    context: Optional[Mapping[str, Any]]
    export_model: bool


class EnvMonitorWrapper(gym.Wrapper):
    """
    Tracks the returns and steps for an environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self.mon = EnvMonitor()

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self.mon.rewards += reward
        self.mon.step += 1
        if term or trunc:
            info["true_episode_return"] = self.mon.rewards
            info["true_episode_steps"] = self.mon.step
        return obs, reward, term, trunc, info

    def reset(self, *, seed=None, options=None):
        self.mon.reset()
        return super().reset(seed=seed, options=options)


class EnvMonitor:
    """
    Monitors episode returns and steps.
    """

    def __init__(self):
        self.returns: List[float] = []
        self.steps: List[int] = []
        self.rewards: float = 0
        self.step: int = 0

    def reset(self):
        """
        Stack values to track new episode.
        """
        if self.step > 0:
            self.returns.append(self.rewards)
            self.steps.append(self.step)
        self.rewards = 0.0
        self.step = 0

    def clear(self):
        """
        Clear monitored data.
        """
        self.returns = []
        self.steps = []
        self.step = 0
        self.rewards = 0.0


class Seeder:
    """
    Use's Cantor's pairing function to turn a pair
    of integers into a unique integer.
    """

    def __init__(self, instance: Optional[int] = None):
        self.instance = instance

    def get_seed(self, episode: int) -> Optional[int]:
        """
        For a given instance (seed), generated
        episode specific seeds consistently.
        """
        if self.instance is not None:
            k1 = self.instance
            k2 = episode
            return int(((k1 + k2) * (k1 + k2 + 1)) / 2 + k2)
        return self.instance


@dataclasses.dataclass
class ProxiedEnv:
    """
    An env and its proxy.
    """

    env: gym.Env
    proxy: gym.Env


class ArgChain:
    """Priority-ordered lookup over a sequence of argument dicts.

    Earlier layers take precedence; later layers are fallbacks.
    Layers 0.._pin-1 are "pinned" (e.g. CLI overrides); prepend() inserts
    new layers at position _pin, keeping the pinned prefix at the top.
    """

    def __init__(self, layers: Sequence[Mapping[str, Any]], _pin: int = 0) -> None:
        self._layers: Tuple[Mapping[str, Any], ...] = tuple(layers)
        self._pin: int = _pin

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        for layer in self._layers:
            if key in layer:
                return layer[key]
        return default

    def prepend(self, layers: Sequence[Mapping[str, Any]]) -> "ArgChain":
        """Insert layers just below the pinned prefix."""
        idx = self._pin
        return ArgChain(
            [*self._layers[:idx], *layers, *self._layers[idx:]],
            _pin=idx,
        )

    def extend(self, layers: Sequence[Mapping[str, Any]]) -> "ArgChain":
        """Append layers at the lowest priority."""
        return ArgChain([*self._layers, *layers], _pin=self._pin)

    @classmethod
    def pinned(
        cls,
        pinned: Sequence[Mapping[str, Any]],
        rest: Sequence[Mapping[str, Any]] = (),
    ) -> "ArgChain":
        """Create a chain with the given layers permanently anchored at the top."""
        return cls([*pinned, *rest], _pin=len(pinned))
