"""
DGRA: Deep return-Grounded Reward Approximation.

A parametric reward model that trains a small MLP with a two-term MSE loss:

    reward_loss = MSE(sum(predicted_per_step), aggregate_reward)
    regu_loss   = MSE(start_return + sum(predicted_per_step), end_return)
    total_loss  = reward_loss + regu_lam * regu_loss

The regularisation term grounds per-step predictions to the observed
progression of cumulative episode returns, encouraging predictions that are
globally consistent with the return trajectory rather than just locally
consistent with each delayed window.

Windows are extracted from completed episode trajectories by scanning the
imputed reward signal: a non-zero env_reward marks the end of a delay
interval and provides the aggregate reward label for that window. Partial
windows at episode termination (where no aggregate reward was received) are
discarded.

Note: delay intervals whose true aggregate reward is exactly 0 are not
detectable from the imputed reward signal alone and will be skipped. This
is a known limitation of the Trajectory schema.
"""

import dataclasses
from typing import List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drmdp.control import base


class _RNetwork(nn.Module):
    """Feedforward MLP predicting per-step reward from (state, action, terminal).

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer width.
        num_hidden_layers: Number of hidden layers (0 = linear model).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        # +1 for the terminal flag so the model can distinguish episode ends
        # from mid-episode transitions without a separate embedding lookup.
        input_dim = obs_dim + action_dim + 1
        layers: List[nn.Module] = []
        layer_in = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(layer_in, hidden_dim))
            layers.append(nn.ReLU())
            layer_in = hidden_dim
        self._layers = nn.Sequential(*layers)
        self._final = nn.Linear(layer_in, 1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        terminals: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-step reward for a batch of (s, a, term) tuples.

        Args:
            obs: Shape (B, obs_dim).
            actions: Shape (B, action_dim).
            terminals: Shape (B, 1), float terminal flags.

        Returns:
            Shape (B, 1) reward predictions.
        """
        combined = torch.cat([obs, actions, terminals], dim=-1)
        return self._final(self._layers(combined))


@dataclasses.dataclass(frozen=True)
class _Window:
    """A single delay-interval window extracted from an episode trajectory.

    Attributes:
        observations: Shape (W, obs_dim).
        actions: Shape (W, action_dim).
        terminals: Shape (W,), episode-end flags.
        aggregate_reward: Sum of per-step rewards over the interval.
        start_return: Cumulative episode return before this window.
        end_return: Cumulative episode return after this window
            (always equal to start_return + aggregate_reward).
    """

    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    aggregate_reward: float
    start_return: float
    end_return: float


class DGRARewardModel(base.RewardModel):
    """Return-grounded MLP reward model for delayed-feedback environments.

    Trains a feedforward MLP to predict per-step rewards from (state, action,
    terminal) inputs, supervised by delayed aggregate rewards and regularised
    by episode return consistency.

    Attributes:
        max_buffer_size: Maximum number of windows retained across updates.
        train_epochs: Training epochs run on the window buffer each update.
        batch_size: Mini-batch size (in windows) for each training step.
        regu_lam: Weight of the return-consistency regularisation term.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        learning_rate: float = 1e-3,
        train_epochs: int = 10,
        batch_size: int = 64,
        regu_lam: float = 1.0,
        max_buffer_size: int = 500,
    ) -> None:
        self._net = _RNetwork(obs_dim, action_dim, hidden_dim, num_hidden_layers)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        self._train_epochs = train_epochs
        self._batch_size = batch_size
        self._regu_lam = regu_lam
        self._max_buffer_size = max_buffer_size
        self._buffer: List[_Window] = []

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Return per-step reward estimates for a batch of transitions.

        Each (state, action, terminal) tuple is processed independently
        (Markovian prediction).

        Args:
            observations: Shape (T, obs_dim).
            actions: Shape (T, action_dim).
            terminals: Shape (T,), episode-end flags.

        Returns:
            Per-step reward estimates, shape (T,), dtype float32.
        """
        self._net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(observations, dtype=torch.float32)
            act_t = torch.as_tensor(actions, dtype=torch.float32)
            term_t = torch.as_tensor(
                terminals[:, np.newaxis].astype(np.float32), dtype=torch.float32
            )
            preds = self._net(obs_t, act_t, term_t)
        numpy_preds: np.ndarray = preds.squeeze(-1).cpu().numpy()
        return numpy_preds.astype(np.float32)

    def update(
        self,
        trajectories: Sequence[base.Trajectory],
    ) -> Mapping[str, float]:
        """Update the model from completed episode trajectories.

        Extracts delay windows from each trajectory, adds them to the
        rolling window buffer, then trains the MLP for train_epochs epochs.

        Args:
            trajectories: Completed episodes to learn from.

        Returns:
            Metrics: buffer_size, training_steps, reward_loss, regu_loss
            (losses are averages over the final training epoch).
        """
        for trajectory in trajectories:
            self._buffer.extend(_extract_windows(trajectory))
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer[-self._max_buffer_size :]

        if not self._buffer:
            return {
                "buffer_size": 0.0,
                "training_steps": 0.0,
                "reward_loss": 0.0,
                "regu_loss": 0.0,
            }

        total_training_steps = 0
        last_epoch_reward_losses: List[float] = []
        last_epoch_regu_losses: List[float] = []

        self._net.train()
        for epoch_idx in range(self._train_epochs):
            indices = np.random.permutation(len(self._buffer))
            epoch_reward_losses: List[float] = []
            epoch_regu_losses: List[float] = []

            for batch_start in range(0, len(self._buffer), self._batch_size):
                batch_indices = indices[batch_start : batch_start + self._batch_size]
                mini_batch = [self._buffer[idx] for idx in batch_indices]

                sum_preds: List[torch.Tensor] = []
                agg_rewards: List[float] = []
                start_returns: List[float] = []
                end_returns: List[float] = []

                for window in mini_batch:
                    obs_t = torch.as_tensor(window.observations, dtype=torch.float32)
                    act_t = torch.as_tensor(window.actions, dtype=torch.float32)
                    term_t = torch.as_tensor(
                        window.terminals[:, np.newaxis].astype(np.float32),
                        dtype=torch.float32,
                    )
                    preds = self._net(obs_t, act_t, term_t)  # (W, 1)
                    sum_preds.append(preds.sum())
                    agg_rewards.append(window.aggregate_reward)
                    start_returns.append(window.start_return)
                    end_returns.append(window.end_return)
                    if epoch_idx == self._train_epochs - 1:
                        total_training_steps += len(window.observations)

                sum_preds_t = torch.stack(sum_preds)
                agg_t = torch.tensor(agg_rewards, dtype=torch.float32)
                start_t = torch.tensor(start_returns, dtype=torch.float32)
                end_t = torch.tensor(end_returns, dtype=torch.float32)

                reward_loss = F.mse_loss(sum_preds_t, agg_t)
                regu_loss = F.mse_loss(start_t + sum_preds_t, end_t)
                total_loss = reward_loss + self._regu_lam * regu_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                epoch_reward_losses.append(reward_loss.item())
                epoch_regu_losses.append(regu_loss.item())

            if epoch_idx == self._train_epochs - 1:
                last_epoch_reward_losses = epoch_reward_losses
                last_epoch_regu_losses = epoch_regu_losses

        return {
            "buffer_size": float(len(self._buffer)),
            "training_steps": float(total_training_steps),
            "reward_loss": float(np.mean(last_epoch_reward_losses)),
            "regu_loss": float(np.mean(last_epoch_regu_losses)),
        }


def _extract_windows(trajectory: base.Trajectory) -> List[_Window]:
    """Extract delay-interval windows from a completed episode trajectory.

    A window spans the steps between consecutive non-zero env_rewards. The
    non-zero env_reward marks the interval end and provides the aggregate
    reward label. A partial tail at episode termination (env_reward == 0 at
    the terminal step) is discarded because no aggregate signal was received.

    Args:
        trajectory: A completed episode trajectory from the control loop.

    Returns:
        List of windows in chronological order. Empty if no complete
        intervals exist in the trajectory.
    """
    windows: List[_Window] = []
    window_start = 0
    cumulative_return = 0.0

    for step_idx, reward in enumerate(trajectory.env_rewards):
        if reward == 0.0:
            continue
        aggregate_reward = float(reward)
        start_return = cumulative_return
        end_return = cumulative_return + aggregate_reward
        windows.append(
            _Window(
                observations=trajectory.observations[window_start : step_idx + 1],
                actions=trajectory.actions[window_start : step_idx + 1],
                terminals=trajectory.terminals[window_start : step_idx + 1],
                aggregate_reward=aggregate_reward,
                start_return=start_return,
                end_return=end_return,
            )
        )
        cumulative_return = end_return
        window_start = step_idx + 1

    return windows
