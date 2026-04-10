"""Reward delay mechanisms and replay buffer for delayed feedback RL."""

import abc
import random
import sys
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import gymnasium as gym
import numpy as np

from drmdp import mathutils


class OptState(str, Enum):
    """
    Optimisation solver states.
    """

    UNSOLVED = "unsolved"
    SOLVED = "solved"


class RewardDelay(abc.ABC):
    """
    Abstract class for delayed reward config.
    """

    @abc.abstractmethod
    def sample(self) -> int:
        """Sample a delay value."""

    @abc.abstractmethod
    def range(self) -> Tuple[int, int]:
        """Return the (min, max) range of possible delay values."""

    @classmethod
    @abc.abstractmethod
    def id(cls) -> str:
        """Return a string identifier for this delay type."""


class FixedDelay(RewardDelay):
    """
    Fixed window delays.
    """

    def __init__(self, delay: int):
        super().__init__()
        self.delay = delay

    def sample(self) -> int:
        return self.delay

    def range(self) -> Tuple[int, int]:
        return self.delay, self.delay

    @classmethod
    def id(cls):
        return "fixed"


class UniformDelay(RewardDelay):
    """
    Delays are sampled uniformly at random
    from a range of values.
    """

    def __init__(self, min_delay: int, max_delay: int):
        super().__init__()
        self.min_delay = min_delay
        self.max_delay = max_delay

    def sample(self):
        """Sample a delay value uniformly at random from the configured range."""
        return random.randint(self.min_delay, self.max_delay)

    def range(self) -> Tuple[int, int]:
        return self.min_delay, self.max_delay

    @classmethod
    def id(cls):
        return "uniform"


class ClippedPoissonDelay(RewardDelay):
    """
    Delays are sampled from a clipped Poisson distribution
    """

    def __init__(
        self, lam: int, min_delay: Optional[int] = None, max_delay: Optional[int] = None
    ):
        """
        Calculate upper and lower bounds if not provided.
        """
        super().__init__()
        lower, upper = mathutils.poisson_exact_confidence_interval(lam)
        self.lam = lam
        self.min_delay = min_delay or lower
        self.max_delay = max_delay or upper
        self.rng = np.random.default_rng()

    def sample(self):
        return np.clip(self.rng.poisson(self.lam), self.min_delay, self.max_delay)

    def range(self) -> Tuple[int, int]:
        return self.min_delay, self.max_delay

    @classmethod
    def id(cls):
        return "clipped-poisson"


class DataBuffer:
    """
    A fixed-capacity buffer for storing data samples.

    Capacity is controlled by `max_capacity` (number of elements) and/or
    `max_size_bytes` (total memory). When a limit is reached, behaviour
    depends on `acc_mode`:

    - ACC_FIRST: keeps the earliest samples; new elements are silently
      dropped once the buffer is full.
    - ACC_LASTEST: keeps the most recent samples; the oldest element is
      evicted to make room before each new element is added.
    """

    ACC_FIRST = "FIRST"
    ACC_LASTEST = "LASTEST"

    def __init__(
        self,
        max_capacity: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        acc_mode: str = ACC_LASTEST,
    ):
        """
        Init.
        """
        self.max_capacity = max_capacity
        self.max_size_bytes = max_size_bytes
        self.acc_mode = acc_mode
        self.buffer: List[Any] = []

    def add(self, element: Any):
        """
        Adds data to buffer.
        """
        if self._within_byte_limit(element) and self._within_capacity_limit():
            self._append(element)

    def clear(self):
        """
        Empties buffer.
        """
        self.buffer = []

    def size(self):
        """
        Current buffer size.
        """
        return len(self.buffer)

    def size_bytes(self):
        """
        Current buffer size in bytes.
        """
        return list_size(self.buffer)

    def _within_byte_limit(self, element: Any) -> bool:
        """Evict oldest entries if needed; return True if element can be added."""
        if self.max_size_bytes is None:
            return True
        if list_size(self.buffer + [element]) < self.max_size_bytes:
            return True
        if self.acc_mode == self.ACC_LASTEST:
            while list_size(self.buffer + [element]) >= self.max_size_bytes:
                self._pop_earliest()
            return True
        return False

    def _within_capacity_limit(self) -> bool:
        """Evict oldest entry if needed; return True if element can be added."""
        if self.max_capacity is None:
            return True
        if self.size() < self.max_capacity:
            return True
        if self.acc_mode == self.ACC_LASTEST:
            self._pop_earliest()
            return True
        return False

    def _pop_earliest(self):
        """
        Remove the First-In element in the list.
        """
        self.buffer.pop(0)

    def _append(self, element: Any):
        """
        Appends values to buffers.
        """
        self.buffer.append(element)


class WindowedTaskSchedule:
    """
    Sets schedule for updates, using two types of schedules:
    1. Fixed interval (fixed)
    2. Doubling size (double)
    """

    FIXED = "fixed"
    DOUBLE = "double"

    def __init__(self, mode: str, init_update_ep: int):
        """
        Instatiates the class for a given update schedule.
        """
        if mode not in (self.FIXED, self.DOUBLE):
            raise ValueError(
                f"Unsupported mode: {mode}. Must be ({self.FIXED}, {self.DOUBLE})"
            )

        self.mode = mode
        self.init_update_ep = init_update_ep
        self.curr_update_ep = init_update_ep
        self._done = False

        self.next_update_ep = (
            self.curr_update_ep * 2
            if self.mode == self.DOUBLE
            else self.curr_update_ep + self.init_update_ep
        )

    def step(self, episode: int) -> None:
        """
        Updates the estimation window.
        """
        if episode == self.next_update_ep:
            # New window
            self.curr_update_ep = self.next_update_ep
            self.next_update_ep = (
                self.curr_update_ep * 2
                if self.mode == self.DOUBLE
                else self.curr_update_ep + self.init_update_ep
            )
            # Reset window state.
            self._done = False

    def set_state(self, succ: bool) -> None:
        """
        Sets state for the current cycle.
        """

        self._done = succ

    @property
    def current_window_done(self) -> bool:
        """
        Returns true if the current cycle state is `False`.
        """
        return self._done


class SupportsName(Protocol):
    """
    Provides methods to get the name of the
    class and underlying (`unwrapped`) env.
    """

    env: gym.Env
    unwrapped: gym.Env

    def get_name(self):
        """
        Name and id of the class.
        """
        cls_name = type(self).__name__
        env_id = id(self)
        return f"{cls_name}(id={env_id})"

    def get_env_name(self):
        """
        Name and id of the underlying (`unwrapped`) environment.
        """
        cls_name = type(self.env.unwrapped).__name__
        env_id = id(self.unwrapped)
        return f"{cls_name}(id={env_id})"


class DelayedRewardWrapper(gym.Wrapper, SupportsName):
    """
    Emits rewards following a delayed aggregation schedule.
    Rewards at the end of the reward window correspond
    to the sum of rewards in the window.
    In the remaining steps, no reward is emitted (`None`).
    """

    def __init__(
        self,
        env: gym.Env,
        reward_delay: RewardDelay,
        op: Callable[[Sequence[float]], float] = sum,
    ):
        super().__init__(env)
        self.reward_delay = reward_delay
        self.segment: Optional[int] = None
        self.segment_step: Optional[int] = None
        self.delay: Optional[int] = None
        self.rewards: List[float] = []
        self.op = op

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self.segment_step += 1
        self.rewards.append(reward)

        segment = self.segment
        segment_step = self.segment_step
        delay = self.delay
        final_segment_step = self.segment_step == self.delay - 1
        if final_segment_step:
            # reset segment
            self.segment += 1
            self.segment_step = -1
            reward = self.op(self.rewards)
            # new delay
            self.delay = self.reward_delay.sample()
            self.rewards = []
        else:
            reward = None
        return (
            obs,
            reward,
            term,
            trunc,
            {
                **info,
                "delay": delay,
                "segment": segment,
                "segment_step": segment_step,
                # Provide the next delay on the final step
                # and omit otherwise
                "next_delay": self.delay if final_segment_step else None,
            },
        )

    def reset(self, *, seed=None, options=None):
        self.segment = 0
        self.segment_step = -1
        self.delay = self.reward_delay.sample()
        self.rewards = []
        obs, info = super().reset(seed=seed, options=options)
        return obs, {
            **info,
            "delay": self.delay,
            "segment": self.segment,
            "segment_step": self.segment_step,
            # Provide the next delay on the final step
            # and omit otherwise
            "next_delay": self.delay,
        }


class ImputeMissingRewardWrapper(gym.RewardWrapper, SupportsName):
    """
    Missing rewards (`None`) are replaced with zero.
    """

    def __init__(self, env: gym.Env, impute_value: float):
        super().__init__(env)
        self.impute_value = float(impute_value)

    def reward(self, reward):
        if reward is None:
            return self.impute_value
        return reward


def list_size(xs: List[Any]) -> int:
    """
    Gets the size of a list in bytes.
    """
    return sys.getsizeof(xs) - sys.getsizeof([])
