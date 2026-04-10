"""
Base abstractions for off-policy control with reward model interleaving.
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A completed episode trajectory.

    Attributes:
        observations: Observations before each action, shape (T, obs_dim).
        actions: Actions taken at each step, shape (T, action_dim).
        env_rewards: Raw environment rewards (delayed aggregate or 0-imputed), shape (T,).
        terminals: Episode-end flags per step, shape (T,).
        episode_return: Sum of env_rewards over the episode.
    """

    observations: np.ndarray
    actions: np.ndarray
    env_rewards: np.ndarray
    terminals: np.ndarray
    episode_return: float


class RewardModel(ABC):
    """Abstract base class for reward estimation/redistribution models."""

    @abstractmethod
    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Predict per-step reward estimates for a batch of transitions.

        Args:
            observations: Observations before each action, shape (T, obs_dim).
            actions: Actions taken, shape (T, action_dim).
            terminals: Episode-end flags, shape (T,). Sequence-aware models
                use this to respect episode boundaries.

        Returns:
            Per-step reward estimates, shape (T,).
        """

    @abstractmethod
    def update(
        self,
        trajectories: Sequence[Trajectory],
    ) -> Mapping[str, float]:
        """Update the model from completed episode trajectories.

        Args:
            trajectories: Completed episodes to learn from.

        Returns:
            Mapping of metric names to scalar values (e.g. buffer_size, loss).
        """


class RelabelingReplayBuffer(ReplayBuffer):
    """Replay buffer that relabels rewards at sample time via a reward model.

    Rewards stored in the buffer are never modified. The reward model is
    invoked on every `sample()` call so the policy always trains on
    up-to-date estimates as the model improves.
    """

    def __init__(
        self,
        *args: Any,
        reward_model: Optional[RewardModel] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model

    def sample(
        self,
        batch_size: int,
        env: Optional[Any] = None,
    ) -> ReplayBufferSamples:
        """Sample a batch, relabeling rewards with the current reward model.

        If no reward model is set, returns stored rewards unchanged.
        """
        batch = super().sample(batch_size, env)
        if self.reward_model is None:
            return batch
        new_rewards = self.reward_model.predict(
            observations=batch.observations.cpu().numpy(),
            actions=batch.actions.cpu().numpy(),
            terminals=batch.dones.cpu().numpy().squeeze(-1),
        )
        relabeled = torch.as_tensor(
            new_rewards[:, np.newaxis],
            dtype=batch.rewards.dtype,
            device=batch.rewards.device,
        )
        return batch._replace(rewards=relabeled)
