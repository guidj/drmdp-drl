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

Windows are extracted from completed episode trajectories using the
per-step ``infos[t]["interval_end"]`` flag injected by
``ImputeMissingRewardWrapper``.  A ``True`` flag marks the end of a delay
interval; the corresponding ``env_rewards[t]`` provides the aggregate
reward label.  Partial windows at episode termination (where no aggregate
reward was received) are discarded.
"""

import dataclasses
from typing import List, Mapping, Optional, Sequence

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
        train_epochs_decay: float = 1.0,
        batch_size: int = 64,
        regu_lam: float = 1.0,
        max_buffer_size: int = 500,
    ) -> None:
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._net = _RNetwork(obs_dim, action_dim, hidden_dim, num_hidden_layers)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        self._train_epochs = train_epochs
        self._train_epochs_decay = train_epochs_decay
        self._update_idx = 0
        self._batch_size = batch_size
        self._regu_lam = regu_lam
        self._max_buffer_size = max_buffer_size
        self._buffer: List[_Window] = []
        # Stacked tensor view of self._buffer, rebuilt lazily inside
        # update() when _stacked_dirty is True. Storing them lets the
        # training loop run one batched forward+backward per mini-batch
        # instead of one per window.
        self._stacked_obs: Optional[torch.Tensor] = None
        self._stacked_acts: Optional[torch.Tensor] = None
        self._stacked_terms: Optional[torch.Tensor] = None
        self._stacked_mask: Optional[torch.Tensor] = None
        self._stacked_agg: Optional[torch.Tensor] = None
        self._stacked_start: Optional[torch.Tensor] = None
        self._stacked_end: Optional[torch.Tensor] = None
        self._stacked_dirty = True

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
        if self._net.training:
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
            extracted = _extract_windows(trajectory)
            if extracted:
                self._buffer.extend(extracted)
                self._stacked_dirty = True
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer[-self._max_buffer_size :]
            self._stacked_dirty = True

        if not self._buffer:
            return {
                "buffer_size": 0.0,
                "training_steps": 0.0,
                "reward_loss": 0.0,
                "regu_loss": 0.0,
            }

        if self._stacked_dirty:
            self._rebuild_stacked()

        assert self._stacked_obs is not None
        assert self._stacked_acts is not None
        assert self._stacked_terms is not None
        assert self._stacked_mask is not None
        assert self._stacked_agg is not None
        assert self._stacked_start is not None
        assert self._stacked_end is not None

        last_epoch_reward_losses: List[float] = []
        last_epoch_regu_losses: List[float] = []

        effective_epochs = max(
            int(self._train_epochs * self._train_epochs_decay**self._update_idx),
            1,
        )

        self._net.train()
        for epoch_idx in range(effective_epochs):
            indices = np.random.permutation(len(self._buffer))
            epoch_reward_losses: List[float] = []
            epoch_regu_losses: List[float] = []

            for batch_start in range(0, len(self._buffer), self._batch_size):
                batch_indices = indices[batch_start : batch_start + self._batch_size]
                idx_t = torch.as_tensor(batch_indices, dtype=torch.long)

                obs_b = self._stacked_obs.index_select(0, idx_t)
                act_b = self._stacked_acts.index_select(0, idx_t)
                term_b = self._stacked_terms.index_select(0, idx_t)
                mask_b = self._stacked_mask.index_select(0, idx_t)
                agg_b = self._stacked_agg.index_select(0, idx_t)
                start_b = self._stacked_start.index_select(0, idx_t)
                end_b = self._stacked_end.index_select(0, idx_t)

                # Flatten (B, W) into (B*W) for a single batched forward,
                # then reshape back and zero-out padded positions before
                # the per-window sum.
                batch_n, window_w, _ = obs_b.shape
                flat_obs = obs_b.reshape(batch_n * window_w, self._obs_dim)
                flat_act = act_b.reshape(batch_n * window_w, self._action_dim)
                flat_term = term_b.reshape(batch_n * window_w, 1)
                flat_preds = self._net(flat_obs, flat_act, flat_term)
                preds = flat_preds.reshape(batch_n, window_w, 1)
                sum_preds = (preds * mask_b).sum(dim=1).squeeze(-1)

                reward_loss = F.mse_loss(sum_preds, agg_b)
                regu_loss = F.mse_loss(start_b + sum_preds, end_b)
                total_loss = reward_loss + self._regu_lam * regu_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                epoch_reward_losses.append(reward_loss.item())
                epoch_regu_losses.append(regu_loss.item())

            if epoch_idx == effective_epochs - 1:
                last_epoch_reward_losses = epoch_reward_losses
                last_epoch_regu_losses = epoch_regu_losses

        # Total training step count = number of real (non-padded) timesteps
        # touched in the final epoch — equals one full pass over the buffer.
        total_training_steps = int(self._stacked_mask.sum().item())

        self._update_idx += 1

        return {
            "buffer_size": float(len(self._buffer)),
            "training_steps": float(total_training_steps),
            "reward_loss": float(np.mean(last_epoch_reward_losses)),
            "regu_loss": float(np.mean(last_epoch_regu_losses)),
        }

    def _rebuild_stacked(self) -> None:
        """Materialise self._buffer as (N, W_max, ...) padded torch tensors.

        Window length varies under ClippedPoissonDelay/UniformDelay; pad each
        window to W_max with zeros and track real-vs-padded positions in a
        binary mask. The stacked tensors are reused across epochs.
        """
        windows = self._buffer
        num_windows = len(windows)
        window_max = max(len(w.observations) for w in windows)

        obs = np.zeros((num_windows, window_max, self._obs_dim), dtype=np.float32)
        acts = np.zeros((num_windows, window_max, self._action_dim), dtype=np.float32)
        terms = np.zeros((num_windows, window_max, 1), dtype=np.float32)
        mask = np.zeros((num_windows, window_max, 1), dtype=np.float32)
        agg = np.empty(num_windows, dtype=np.float32)
        start = np.empty(num_windows, dtype=np.float32)
        end = np.empty(num_windows, dtype=np.float32)

        for idx, window in enumerate(windows):
            length = len(window.observations)
            obs[idx, :length] = window.observations
            acts[idx, :length] = window.actions
            terms[idx, :length, 0] = window.terminals
            mask[idx, :length, 0] = 1.0
            agg[idx] = window.aggregate_reward
            start[idx] = window.start_return
            end[idx] = window.end_return

        self._stacked_obs = torch.from_numpy(obs)
        self._stacked_acts = torch.from_numpy(acts)
        self._stacked_terms = torch.from_numpy(terms)
        self._stacked_mask = torch.from_numpy(mask)
        self._stacked_agg = torch.from_numpy(agg)
        self._stacked_start = torch.from_numpy(start)
        self._stacked_end = torch.from_numpy(end)
        self._stacked_dirty = False


def _extract_windows(trajectory: base.Trajectory) -> List[_Window]:
    """Extract delay-interval windows from a completed episode trajectory.

    A window spans the steps between consecutive ``interval_end`` flags
    (from ``trajectory.infos``).  The flag marks the interval end; the
    corresponding ``env_rewards`` entry provides the aggregate reward
    label.  A partial tail without an ``interval_end`` flag is discarded
    because no aggregate signal was received.

    Args:
        trajectory: A completed episode trajectory from the control loop.

    Returns:
        List of windows in chronological order. Empty if no complete
        intervals exist in the trajectory.
    """
    windows: List[_Window] = []
    window_start = 0
    cumulative_return = 0.0

    for step_idx in range(len(trajectory.env_rewards)):
        if not trajectory.infos[step_idx].get("interval_end", False):
            continue
        aggregate_reward = float(trajectory.env_rewards[step_idx])
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
