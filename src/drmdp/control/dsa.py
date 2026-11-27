"""
DSA: Delayed State Attention for off-policy RL with delayed rewards.

Combines three components:

1. A transformer encoder that compresses recent (s, a) history into a
   latent vector z, trained via next-state prediction:
       f(z, s_t, a_t) → ŝ_{t+1}
   This forces z to capture causally relevant dynamics — the information
   from history that actually matters for predicting what happens next.

2. A DGRA reward model that relabels per-step rewards at sample time
   from delayed aggregate feedback.

3. Standard SAC with Q([obs_t, z], a_t).  Since z is detached from the
   RL computation graph, actor and critic gradients never flow through
   the encoder — variance reduction without Q-decomposition.

The encoder runs at sample time in the replay buffer (Python/PyTorch),
so the policy can train via sbx.SAC (JAX).  Two history modes are
supported: ``per_interval`` (reset at signal boundaries) and
``sliding_window`` (last K steps regardless of intervals).
"""

import collections
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common import buffers, callbacks
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from drmdp.control import base

# ---------------------------------------------------------------------------
# Private network modules
# ---------------------------------------------------------------------------


class _LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for short sequences."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self._embedding = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional encoding for the first ``seq_len`` positions.

        Returns:
            Shape (1, seq_len, d_model).
        """
        positions = torch.arange(seq_len, device=self._embedding.weight.device)
        return self._embedding(positions).unsqueeze(0)


class _TransformerHistoryEncoder(nn.Module):
    """Transformer encoder over a padded (s, a) sequence.

    Projects each (s, a) pair into d_model dimensions, adds learned
    positional encoding, passes through a TransformerEncoder, then
    mean-pools over non-padded positions and projects to latent_dim.

    Args:
        sa_dim: Dimension of concatenated (state, action) vectors.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoderLayer blocks.
        latent_dim: Output latent dimension.
        max_len: Maximum sequence length (used for positional encoding).
    """

    def __init__(
        self,
        sa_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        latent_dim: int = 32,
        max_len: int = 10,
    ) -> None:
        super().__init__()
        self._input_proj = nn.Linear(sa_dim, d_model)
        self._pos_enc = _LearnedPositionalEncoding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self._transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self._pool_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self._to_latent = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        history: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history sequence into a latent vector.

        Args:
            history: (B, K, sa_dim) — left-zero-padded (s, a) pairs.
            padding_mask: (B, K) bool — True for padded positions.

        Returns:
            z: (B, latent_dim).
        """
        all_padded = padding_mask.all(dim=1)
        if all_padded.all():
            return torch.zeros(
                history.shape[0],
                self._to_latent.out_features,
                device=history.device,
                dtype=history.dtype,
            )

        safe_mask = padding_mask.clone()
        if all_padded.any():
            safe_mask[all_padded, 0] = False

        projected = F.gelu(self._input_proj(history))
        projected = projected + self._pos_enc(history.shape[1])
        encoded = self._transformer(projected, src_key_padding_mask=safe_mask)

        query = self._pool_query.expand(history.shape[0], -1, -1)
        pooled, _ = self._pool_attn(query, encoded, encoded, key_padding_mask=safe_mask)
        pooled = pooled.squeeze(1)

        result = self._to_latent(pooled)

        if all_padded.any():
            result[all_padded] = 0.0

        return result


class _NextStatePredictor(nn.Module):
    """MLP predicting the next state from (z, s_t, a_t).

    Args:
        latent_dim: Dimension of the history latent vector z.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + obs_dim + action_dim
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state.

        Args:
            z: (B, latent_dim) — history encoding.
            obs: (B, obs_dim) — current observation.
            actions: (B, action_dim) — current action.

        Returns:
            Predicted next observation, shape (B, obs_dim).
        """
        combined = torch.cat([z, obs, actions], dim=-1)
        return self._net(combined)


class HistoryEncoder(nn.Module):
    """Transformer history encoder trained via next-state prediction.

    Wraps a transformer encoder and a next-state predictor.  The encoder
    compresses (s, a) history into a latent z; the predictor uses z plus
    the current (s, a) to predict the next state.  The prediction loss
    trains both networks end-to-end.

    At inference (sample time in the replay buffer), only ``encode()``
    is called — it returns a detached z with no gradient.

    Args:
        sa_dim: Dimension of concatenated (state, action) vectors.
        obs_dim: Observation dimension (for the predictor output).
        action_dim: Action dimension.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of TransformerEncoderLayer blocks.
        latent_dim: Latent dimension for z.
        max_len: Maximum history sequence length.
        predictor_hidden_dim: Hidden layer width for the predictor MLP.
    """

    def __init__(
        self,
        sa_dim: int,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        latent_dim: int = 32,
        max_len: int = 10,
        predictor_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = _TransformerHistoryEncoder(
            sa_dim=sa_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            latent_dim=latent_dim,
            max_len=max_len,
        )
        self.predictor = _NextStatePredictor(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
        )

    def encode(
        self,
        history: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history into a detached latent vector.

        Args:
            history: (B, K, sa_dim) — left-zero-padded.
            padding_mask: (B, K) bool — True for padded positions.

        Returns:
            z: (B, latent_dim), detached from computation graph.
        """
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(history, padding_mask)

    def prediction_loss(
        self,
        history: torch.Tensor,
        padding_mask: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        terminals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next-state prediction loss (MSE) for training.

        Non-terminal transitions only: terminal transitions have no
        meaningful next state.

        Args:
            history: (B, K, sa_dim).
            padding_mask: (B, K) bool.
            obs: (B, obs_dim) — current observation.
            actions: (B, action_dim) — current action.
            next_obs: (B, obs_dim) — actual next observation.
            terminals: (B,) bool or float — True/1.0 for terminal steps.

        Returns:
            Scalar MSE loss over non-terminal transitions.
        """
        self.encoder.train()
        self.predictor.train()
        z = self.encoder(history, padding_mask)
        predicted = self.predictor(z, obs, actions)
        non_terminal = (
            (~terminals.bool()) if terminals.dtype == torch.bool else (terminals < 0.5)
        )
        if not non_terminal.any():
            return torch.tensor(0.0, device=history.device, requires_grad=True)
        return F.mse_loss(predicted[non_terminal], next_obs[non_terminal])


# ---------------------------------------------------------------------------
# History window extraction
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _HistoryWindow:
    """A single history window paired with a transition for encoder training.

    Attributes:
        history: Left-zero-padded (s, a) pairs, shape (max_delay, sa_dim).
        obs: Current observation at this step, shape (obs_dim,).
        action: Action taken at this step, shape (action_dim,).
        next_obs: Next observation, shape (obs_dim,).
        terminal: Whether this step ended the episode.
    """

    history: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    terminal: bool


def _extract_history_windows(
    trajectory: base.Trajectory,
    max_delay: int,
    mode: str = "per_interval",
) -> List[_HistoryWindow]:
    """Extract (s, a) history windows from a completed trajectory.

    Each window captures the history *before* step t (excluding step t),
    paired with the transition at step t for next-state prediction training.

    Args:
        trajectory: A completed episode trajectory.
        max_delay: Maximum history window length.
        mode: ``"per_interval"`` resets history at ``interval_end`` and done;
            ``"sliding_window"`` resets only at done.

    Returns:
        List of history windows.
    """
    windows: List[_HistoryWindow] = []
    obs_dim = trajectory.observations.shape[1]
    action_dim = trajectory.actions.shape[1]
    sa_dim = obs_dim + action_dim

    recent: List[np.ndarray] = []
    num_steps = len(trajectory.observations)

    for step_idx in range(num_steps):
        if step_idx < num_steps - 1:
            next_obs_val = trajectory.observations[step_idx + 1]
        else:
            next_obs_val = trajectory.observations[step_idx]

        history_window = np.zeros((max_delay, sa_dim), dtype=np.float32)
        n_recent = min(len(recent), max_delay)
        if n_recent > 0:
            history_window[-n_recent:] = np.stack(recent[-n_recent:])

        windows.append(
            _HistoryWindow(
                history=history_window,
                obs=trajectory.observations[step_idx],
                action=trajectory.actions[step_idx],
                next_obs=next_obs_val,
                terminal=bool(trajectory.terminals[step_idx]),
            )
        )

        is_terminal = bool(trajectory.terminals[step_idx])
        is_interval_end = trajectory.infos[step_idx].get("interval_end", False)

        if is_terminal:
            recent = []
        elif mode == "per_interval" and is_interval_end:
            recent = []
        else:
            sa_pair = np.concatenate(
                [
                    trajectory.observations[step_idx],
                    trajectory.actions[step_idx],
                ]
            )
            recent.append(sa_pair)
            if len(recent) > max_delay:
                recent = recent[-max_delay:]

    return windows


# ---------------------------------------------------------------------------
# Observation wrappers
# ---------------------------------------------------------------------------


class LatentAugmentedObsWrapper(gym.Wrapper):
    """Expands the observation space by ``latent_dim`` zero-padded dimensions.

    At collection time, observations are padded with zeros.  The actual
    latent values are injected at sample time by ``DSAReplayBuffer``.
    This wrapper exists so that SAC constructs networks with the correct
    input dimension ``(obs_dim + latent_dim)``.
    """

    def __init__(self, env: gym.Env, latent_dim: int) -> None:
        super().__init__(env)
        self._latent_dim = latent_dim
        obs_space: gym.spaces.Box = self.observation_space  # type: ignore[has-type]
        low = np.concatenate([obs_space.low, np.full(latent_dim, -np.inf)])
        high = np.concatenate([obs_space.high, np.full(latent_dim, np.inf)])
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [obs.astype(np.float32), np.zeros(self._latent_dim, dtype=np.float32)]
        )

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._pad(obs), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pad(obs), reward, terminated, truncated, info


class DSAEvalWrapper(gym.Wrapper):
    """Stateful evaluation wrapper that encodes history on the fly.

    Maintains a deque of recent (s, a) pairs and encodes them via the
    transformer encoder at each step, returning ``[obs, z]`` instead
    of ``[obs, 0...0]``.

    Args:
        env: The wrapped environment (should already have
            ``LatentAugmentedObsWrapper`` applied upstream, whose
            zero-padding this wrapper overwrites with real z values).
        history_encoder: The trained ``HistoryEncoder`` module.
        max_delay: Maximum history window length.
        latent_dim: Latent vector dimension.
        obs_dim_before_augmentation: Observation dimension before the
            ``LatentAugmentedObsWrapper`` (i.e. including
            ``IntervalPositionWrapper`` but not latent padding).
        action_dim: Action dimension.
        history_mode: ``"per_interval"`` or ``"sliding_window"``.
    """

    def __init__(
        self,
        env: gym.Env,
        history_encoder: "HistoryEncoder",
        max_delay: int,
        latent_dim: int,
        obs_dim_before_augmentation: int,
        action_dim: int,
        history_mode: str = "per_interval",
    ) -> None:
        super().__init__(env)
        self._history_encoder = history_encoder
        self._max_delay = max_delay
        self._latent_dim = latent_dim
        self._obs_dim = obs_dim_before_augmentation
        self._action_dim = action_dim
        self._history_mode = history_mode
        self._sa_dim = obs_dim_before_augmentation + action_dim
        self._recent_sa: collections.deque = collections.deque(maxlen=max_delay)
        self._prev_obs: Optional[np.ndarray] = None

    def _encode_history(self) -> np.ndarray:
        history_window = np.zeros((self._max_delay, self._sa_dim), dtype=np.float32)
        recent = list(self._recent_sa)
        n_recent = min(len(recent), self._max_delay)
        if n_recent > 0:
            history_window[-n_recent:] = np.stack(recent[-n_recent:])
        history_t = torch.as_tensor(history_window, dtype=torch.float32).unsqueeze(0)
        padding_mask = history_t.abs().sum(dim=-1) == 0
        z = self._history_encoder.encode(history_t, padding_mask)
        return z.squeeze(0).cpu().numpy()  # type: ignore[no-any-return]

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        z = self._encode_history()
        raw_obs = obs[: self._obs_dim]
        return np.concatenate([raw_obs, z]).astype(np.float32)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._recent_sa.clear()
        self._prev_obs = obs
        return self._augment(obs), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented = self._augment(obs)

        if self._prev_obs is not None:
            sa_pair = np.concatenate(
                [self._prev_obs[: self._obs_dim], np.asarray(action).flatten()]
            )
            self._recent_sa.append(sa_pair)

        is_interval_end = info.get("interval_end", False)
        done = terminated or truncated
        if done:
            self._recent_sa.clear()
            self._prev_obs = None
        elif self._history_mode == "per_interval" and is_interval_end:
            self._recent_sa.clear()
            self._prev_obs = obs
        else:
            self._prev_obs = obs

        return augmented, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class DSAReplayBuffer(buffers.ReplayBuffer):
    """Replay buffer with history tracking, latent encoding, and reward relabeling.

    Stores (s, a) history per transition (adapted from ``HCReplayBuffer``).
    At sample time:
    1. Encodes history → detached z via the transformer encoder.
    2. Augments observations: [obs, z] and [next_obs, z_next].
    3. Relabels rewards via a reward model (e.g. DGRA).
    4. Returns standard ``ReplayBufferSamples`` compatible with sbx.SAC.

    Args:
        *args: Forwarded to ``ReplayBuffer``.
        max_delay: Maximum history window length.
        history_encoder: The ``HistoryEncoder`` module (or None before setup).
        reward_model: The reward model for relabeling (or None).
        raw_obs_dim: Observation dim before ``LatentAugmentedObsWrapper``
            (includes IntervalPosition dim).
        history_mode: ``"per_interval"`` or ``"sliding_window"``.
        **kwargs: Forwarded to ``ReplayBuffer``.
    """

    def __init__(
        self,
        *args: Any,
        max_delay: int = 3,
        history_encoder: Optional[HistoryEncoder] = None,
        reward_model: Optional[base.RewardModel] = None,
        raw_obs_dim: Optional[int] = None,
        history_mode: str = "per_interval",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_delay = max_delay
        self._history_encoder = history_encoder
        self._reward_model = reward_model
        self._history_mode = history_mode
        self._raw_obs_dim = raw_obs_dim

        if raw_obs_dim is not None:
            sa_dim = raw_obs_dim + self.action_dim
        else:
            sa_dim = int(np.prod(self.obs_shape)) + self.action_dim
        self._sa_dim = sa_dim

        self._history_sa = np.zeros(
            (self.buffer_size, self.n_envs, max_delay, sa_dim),
            dtype=np.float32,
        )
        self._interval_ends = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        self._recent_sa: List[collections.deque] = [
            collections.deque(maxlen=max_delay) for _ in range(self.n_envs)
        ]

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        pos = self.pos
        done_arr = np.asarray(done).reshape(self.n_envs)
        obs_arr = np.asarray(obs).reshape(self.n_envs, -1)
        act_arr = np.asarray(action).reshape(self.n_envs, self.action_dim)

        for env_idx in range(self.n_envs):
            recent = list(self._recent_sa[env_idx])
            n_recent = len(recent)
            history_window = np.zeros((self._max_delay, self._sa_dim), dtype=np.float32)
            if n_recent > 0:
                history_window[-n_recent:] = np.stack(recent)
            self._history_sa[pos, env_idx] = history_window
            self._interval_ends[pos, env_idx, 0] = float(
                infos[env_idx].get("interval_end", False)
            )

        super().add(obs, next_obs, action, reward, done, infos)

        for env_idx in range(self.n_envs):
            is_done = bool(done_arr[env_idx])
            is_interval_end = infos[env_idx].get("interval_end", False)

            if is_done:
                self._recent_sa[env_idx].clear()
            elif self._history_mode == "per_interval" and is_interval_end:
                self._recent_sa[env_idx].clear()
            else:
                raw_obs = obs_arr[env_idx]
                if self._raw_obs_dim is not None:
                    raw_obs = raw_obs[: self._raw_obs_dim]
                sa_pair = np.concatenate([raw_obs, act_arr[env_idx]])
                self._recent_sa[env_idx].append(sa_pair)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[Any] = None,
    ) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs_np = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs_np = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        obs_np = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions_np = self.actions[batch_inds, env_indices, :]
        dones_np = (
            self.dones[batch_inds, env_indices]
            * (1 - self.timeouts[batch_inds, env_indices])
        ).reshape(-1, 1)
        rewards_np = self._normalize_reward(
            self.rewards[batch_inds, env_indices].reshape(-1, 1), env
        )

        obs_t = self.to_torch(obs_np)
        actions_t = self.to_torch(actions_np)
        next_obs_t = self.to_torch(next_obs_np)
        dones_t = self.to_torch(dones_np)
        rewards_t = self.to_torch(rewards_np)

        history_t = self.to_torch(self._history_sa[batch_inds, env_indices])
        interval_ends_t = self.to_torch(self._interval_ends[batch_inds, env_indices])

        if self._history_encoder is not None and self._raw_obs_dim is not None:
            raw_obs_dim = self._raw_obs_dim
            padding_mask = history_t.abs().sum(dim=-1) == 0
            z = self._history_encoder.encode(history_t, padding_mask)

            raw_obs_t = obs_t[:, :raw_obs_dim]
            raw_act_t = actions_t
            current_sa = torch.cat([raw_obs_t, raw_act_t], dim=-1).unsqueeze(1)
            shifted = torch.cat([history_t[:, 1:, :], current_sa], dim=1)
            reset_mask = dones_t.bool()
            if self._history_mode == "per_interval":
                reset_mask = reset_mask | interval_ends_t.bool()
            next_history = torch.where(
                reset_mask.unsqueeze(-1).expand_as(shifted),
                torch.zeros_like(shifted),
                shifted,
            )
            next_padding_mask = next_history.abs().sum(dim=-1) == 0
            z_next = self._history_encoder.encode(next_history, next_padding_mask)

            # Replace the zero-padded latent dims with actual z values.
            obs_t = torch.cat([obs_t[:, :raw_obs_dim], z], dim=-1)
            next_obs_t = torch.cat([next_obs_t[:, :raw_obs_dim], z_next], dim=-1)

        if self._reward_model is not None and self._raw_obs_dim is not None:
            raw_obs_for_rm = obs_np[:, : self._raw_obs_dim]
            new_rewards = self._reward_model.predict(
                observations=raw_obs_for_rm,
                actions=actions_np,
                terminals=dones_np.squeeze(-1),
            )
            rewards_t = torch.as_tensor(
                new_rewards[:, np.newaxis],
                dtype=rewards_t.dtype,
                device=rewards_t.device,
            )

        return ReplayBufferSamples(
            observations=obs_t,
            actions=actions_t,
            next_observations=next_obs_t,
            dones=dones_t,
            rewards=rewards_t,
        )


# ---------------------------------------------------------------------------
# Training callback
# ---------------------------------------------------------------------------


class DSACallback(callbacks.BaseCallback):
    """SB3 callback for DSA: trains encoder and updates DGRA reward model.

    Collects completed episode trajectories and periodically:
    - Extracts history windows and trains the encoder via prediction loss.
    - Passes trajectories to the DGRA reward model for its own update.

    Args:
        history_encoder: The ``HistoryEncoder`` module to train.
        reward_model: The DGRA reward model.
        max_delay: History window length.
        history_mode: ``"per_interval"`` or ``"sliding_window"``.
        encoder_update_every_n_steps: Train encoder every N env steps.
        encoder_train_epochs: Epochs per encoder update.
        encoder_batch_size: Mini-batch size for encoder training.
        encoder_learning_rate: Learning rate for encoder optimizer.
        reward_model_update_every_n_steps: Update DGRA every N env steps.
        clear_buffer_on_update: Reset SAC replay buffer after DGRA update.
        log_episode_frequency: Log episode stats every N episodes.
        train_logger: Logger for experiment tracking.
        max_history_buffer_size: Max history windows retained for training.
    """

    def __init__(
        self,
        history_encoder: HistoryEncoder,
        reward_model: Optional[base.RewardModel],
        max_delay: int,
        raw_obs_dim: int,
        history_mode: str = "per_interval",
        encoder_update_every_n_steps: int = 2000,
        encoder_train_epochs: int = 5,
        encoder_batch_size: int = 64,
        encoder_learning_rate: float = 1e-3,
        reward_model_update_every_n_steps: int = 2000,
        clear_buffer_on_update: bool = False,
        log_episode_frequency: int = 1,
        train_logger: Optional[Any] = None,
        max_history_buffer_size: int = 10000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._history_encoder = history_encoder
        self._reward_model = reward_model
        self._max_delay = max_delay
        self._raw_obs_dim = raw_obs_dim
        self._history_mode = history_mode
        self._encoder_update_every_n_steps = encoder_update_every_n_steps
        self._encoder_train_epochs = encoder_train_epochs
        self._encoder_batch_size = encoder_batch_size
        self._reward_model_update_every_n_steps = reward_model_update_every_n_steps
        self._clear_buffer_on_update = clear_buffer_on_update
        self._log_episode_frequency = log_episode_frequency
        self._train_logger = train_logger
        self._max_history_buffer_size = max_history_buffer_size

        self._encoder_optimizer = torch.optim.Adam(
            history_encoder.parameters(), lr=encoder_learning_rate
        )
        self._history_buffer: List[_HistoryWindow] = []
        self._pending_trajectories: List[base.Trajectory] = []

        self._episode_obs: List[np.ndarray] = []
        self._episode_actions: List[np.ndarray] = []
        self._episode_rewards: List[float] = []
        self._episode_terminals: List[bool] = []
        self._episode_infos: List[Dict[str, Any]] = []

        self._last_encoder_update_step = 0
        self._last_rm_update_step = 0
        self._episode_count = 0
        self._last_encoder_metrics: Mapping[str, float] = {}
        self._last_rm_metrics: Mapping[str, float] = {}

    def _on_training_start(self) -> None:
        self._last_encoder_update_step = 0
        self._last_rm_update_step = 0

    def _on_step(self) -> bool:
        obs = self.model._last_obs[0].copy()
        obs = obs[: self._raw_obs_dim]
        action = np.asarray(self.locals["actions"]).reshape(-1)
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self._episode_obs.append(obs)
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)
        self._episode_terminals.append(done)
        self._episode_infos.append(dict(info))

        if done:
            self._on_episode_end()

        if (
            self.num_timesteps
            >= self._last_encoder_update_step + self._encoder_update_every_n_steps
            and self._history_buffer
        ):
            self._last_encoder_metrics = self._update_encoder()
            self._last_encoder_update_step = self.num_timesteps

        if (
            self._reward_model is not None
            and self.num_timesteps
            >= self._last_rm_update_step + self._reward_model_update_every_n_steps
            and self._pending_trajectories
        ):
            self._last_rm_metrics = self._reward_model.update(
                self._pending_trajectories
            )
            self._pending_trajectories = []
            self._last_rm_update_step = self.num_timesteps
            if self._clear_buffer_on_update:
                self.model.replay_buffer.reset()

        return True

    def _on_episode_end(self) -> None:
        if not self._episode_obs:
            return

        trajectory = base.Trajectory(
            observations=np.array(self._episode_obs),
            actions=np.array(self._episode_actions),
            env_rewards=np.array(self._episode_rewards),
            terminals=np.array(self._episode_terminals, dtype=np.float32),
            infos=tuple(self._episode_infos),
            episode_return=sum(self._episode_rewards),
        )
        self._pending_trajectories.append(trajectory)

        windows = _extract_history_windows(
            trajectory, self._max_delay, self._history_mode
        )
        self._history_buffer.extend(windows)
        if len(self._history_buffer) > self._max_history_buffer_size:
            self._history_buffer = self._history_buffer[
                -self._max_history_buffer_size :
            ]

        self._episode_count += 1
        if (
            self._train_logger is not None
            and self._episode_count % self._log_episode_frequency == 0
        ):
            self._train_logger.log(
                episode=self._episode_count,
                steps=len(trajectory.observations),
                global_steps=self.num_timesteps,
                returns=trajectory.episode_return,
                info={
                    **{
                        f"encoder/{key}": val
                        for key, val in self._last_encoder_metrics.items()
                    },
                    **{
                        f"reward_model/{key}": val
                        for key, val in self._last_rm_metrics.items()
                    },
                },
            )

        self._episode_obs = []
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_terminals = []
        self._episode_infos = []

    def _update_encoder(self) -> Mapping[str, float]:
        if not self._history_buffer:
            return {"encoder_loss": 0.0, "encoder_buffer_size": 0}

        losses: List[float] = []
        for _epoch in range(self._encoder_train_epochs):
            indices = np.random.permutation(len(self._history_buffer))
            for batch_start in range(
                0, len(self._history_buffer), self._encoder_batch_size
            ):
                batch_idx = indices[
                    batch_start : batch_start + self._encoder_batch_size
                ]
                batch_windows = [self._history_buffer[idx] for idx in batch_idx]

                history_batch = torch.as_tensor(
                    np.stack([win.history for win in batch_windows]),
                    dtype=torch.float32,
                )
                obs_batch = torch.as_tensor(
                    np.stack([win.obs for win in batch_windows]),
                    dtype=torch.float32,
                )
                action_batch = torch.as_tensor(
                    np.stack([win.action for win in batch_windows]),
                    dtype=torch.float32,
                )
                next_obs_batch = torch.as_tensor(
                    np.stack([win.next_obs for win in batch_windows]),
                    dtype=torch.float32,
                )
                terminal_batch = torch.as_tensor(
                    np.array(
                        [win.terminal for win in batch_windows],
                        dtype=np.float32,
                    ),
                )

                padding_mask = history_batch.abs().sum(dim=-1) == 0
                loss = self._history_encoder.prediction_loss(
                    history_batch,
                    padding_mask,
                    obs_batch,
                    action_batch,
                    next_obs_batch,
                    terminal_batch,
                )
                self._encoder_optimizer.zero_grad()
                loss.backward()
                self._encoder_optimizer.step()
                losses.append(loss.item())

        return {
            "encoder_loss": float(np.mean(losses)) if losses else 0.0,
            "encoder_buffer_size": float(len(self._history_buffer)),
        }
