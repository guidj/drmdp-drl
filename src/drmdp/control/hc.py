"""
HC-decomposition SAC agent for off-policy RL with delayed rewards.

The Q-function is decomposed into:

    Q^π(h_t, a_t) = Q^H(h_t) + Q^C(s_t, a_t)

where h_t = H_φ(τ_{t_i:t}) is a GRU encoding of every (s, a) pair in the
current signal interval *before* step t.  Signal intervals reset at each
interval boundary (identified via ``info["interval_end"]`` injected by
``ImputeMissingRewardWrapper``) and at episode termination.

The actor gradient flows through Q^C only, removing historical reward
variance from policy updates.

A regularisation loss (Eq. 9, λ·MSE(H_φ, R_t) at interval-end steps)
prevents H_φ from absorbing too much credit from C_φ.

The policy class Π_s from the paper is implemented by wrapping the
environment with ``IntervalPositionWrapper``, which appends the normalised
interval position (t - t_i) / max_delay to every observation.

Reference: Han et al., "Off-Policy Reinforcement Learning with Delayed
Rewards", ICML 2022 (§4, Eq. 6–9).
"""

import collections
import copy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common import buffers, utils
from stable_baselines3.sac.policies import SACPolicy

# ---------------------------------------------------------------------------
# IntervalPositionWrapper
# ---------------------------------------------------------------------------


class IntervalPositionWrapper(gym.Wrapper):
    """Appends normalised interval position (t - t_i) / max_delay to observations.

    Implements the Π_s policy class from the paper (§3.2): the actor and
    critic receive the current state augmented with the step's position within
    the current signal interval.

    Interval boundaries are detected from ``info["interval_end"]`` (injected
    by ``ImputeMissingRewardWrapper``), which is robust to signal intervals
    whose aggregate reward sums to zero.

    Must be applied **after** ``ImputeMissingRewardWrapper``.

    Args:
        env: Wrapped gymnasium environment.
        max_delay: Maximum signal interval length; used to normalise position.
    """

    def __init__(self, env: gym.Env, max_delay: int) -> None:
        super().__init__(env)
        self._max_delay = max_delay
        self._position: int = 0
        low = np.concatenate([self.observation_space.low, [0.0]])  # type: ignore[has-type]
        high = np.concatenate([self.observation_space.high, [1.0]])  # type: ignore[has-type]
        self.observation_space = gym.spaces.Box(  # type: ignore[assignment]
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._position = 0
        return self._augment(obs), info  # type: ignore[no-any-return]

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("interval_end", False) or terminated or truncated:
            self._position = 0
        else:
            self._position = min(self._position + 1, self._max_delay)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        pos = np.array([self._position / self._max_delay], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), pos])


# ---------------------------------------------------------------------------
# Named tuple for HC replay buffer samples
# ---------------------------------------------------------------------------


class HCReplayBufferSamples(NamedTuple):
    """Replay buffer samples extended with signal-interval history.

    Fields mirror ``ReplayBufferSamples`` with two additional tensors:
    - ``history``: (batch, max_delay, obs_dim + act_dim) — left-zero-padded
      (s, a) sequence from the start of the current signal interval
    - ``interval_ends``: (batch, 1) float32 — 1.0 if the step was an
      interval boundary (reward was emitted), 0.0 otherwise
    """

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    history: torch.Tensor  # (batch, max_delay, obs_dim + act_dim)
    interval_ends: torch.Tensor  # (batch, 1) float32
    discounts: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# HCSAC: top-level agent
# ---------------------------------------------------------------------------


class HCSAC(SAC):
    """SAC with HC Q-function decomposition for delayed reward environments.

    Replaces the standard SAC critic Q(s, a) with:

        Q^π(h_t, a_t) = Q^H(h_t) + Q^C(s_t, a_t)

    Critic and head are trained with a shared TD target that includes both
    components.  The actor gradient is computed through Q^C only, which
    reduces variance from historical accumulated rewards.

    Args:
        env: Gymnasium environment or vector environment.
        max_delay: Maximum signal interval length (= maximum reward delay).
            Determines the GRU input window size.  Should match the delay
            parameter used in ``DelayedRewardWrapper``.
        history_hidden_size: Hidden size of the GRU history encoder.
        **kwargs: Forwarded to ``stable_baselines3.SAC``.
    """

    def __init__(
        self,
        env: Any,
        max_delay: int = 3,
        history_hidden_size: int = 128,
        reg_lambda: float = 5.0,
        **kwargs: Any,
    ) -> None:
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs["max_delay"] = max_delay
        policy_kwargs["history_hidden_size"] = history_hidden_size

        rb_kwargs = kwargs.pop("replay_buffer_kwargs", {})
        rb_kwargs["max_delay"] = max_delay

        super().__init__(
            HCSACPolicy,
            env,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=kwargs.pop("replay_buffer_class", HCReplayBuffer),
            replay_buffer_kwargs=rb_kwargs,
            **kwargs,
        )
        self._max_delay = max_delay
        # λ for H_φ regularisation loss (Eq. 9). Default 5.0 per paper for RNN variant.
        self._reg_lambda = reg_lambda

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:  # noqa: PLR0915
        """HC training loop.

        Differences from standard SAC.train():
        - TD target includes Q^H_target(h_{t+1}) + Q^C_target(s', a').
        - Head network trained to predict the full target.
        - Critic trained to predict target minus the head component.
        - Actor gradient flows through Q^C only (not Q^H).
        - Separate soft-updates for head/encoder target networks.
        """
        self.policy.set_training_mode(True)
        optimizers = [
            self.actor.optimizer,
            self.critic.optimizer,
            self.policy.head_optimizer,
        ]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        head_losses: List[float] = []
        critic_losses: List[float] = []
        actor_losses: List[float] = []
        ent_coef_losses: List[float] = []
        ent_coefs: List[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Use interval_ends stored in the replay buffer rather than reward != 0
            # so that zero-sum signal intervals are handled correctly.
            current_sa = torch.cat(
                [replay_data.observations, replay_data.actions], dim=-1
            ).unsqueeze(1)  # (B, 1, sa_dim)
            history = replay_data.history  # (B, max_delay, sa_dim)
            shifted = torch.cat([history[:, 1:, :], current_sa], dim=1)
            interval_end = replay_data.interval_ends.bool()  # (B, 1)
            next_history = torch.where(
                interval_end.unsqueeze(-1).expand_as(shifted),
                torch.zeros_like(shifted),
                shifted,
            )  # (B, max_delay, sa_dim)

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )

                _, h_next_enc = self.history_encoder_target(next_history)
                h_next = h_next_enc.squeeze(0)  # (B, hidden)
                qh_next = self.head_net_target(h_next)  # (B, 1)

                next_qc = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_qc_min, _ = torch.min(next_qc, dim=1, keepdim=True)
                next_qc_min = next_qc_min - ent_coef * next_log_prob.reshape(-1, 1)

                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * (
                    qh_next + next_qc_min
                )

            _, h_t_enc = self.history_encoder(history)
            h_t = h_t_enc.squeeze(0)  # (B, hidden)
            qh_pred = self.head_net(h_t)  # (B, 1)
            head_loss: torch.Tensor = F.mse_loss(qh_pred, target_q.detach())

            # Regularisation (Eq. 9): at interval-end steps force H_φ ≈ R_t.
            # Prevents H_φ from absorbing credit that C_φ needs for the actor.
            if self._reg_lambda > 0.0 and interval_end.any():
                reg_mask = interval_end.squeeze(-1)  # (B,)
                reg_loss: torch.Tensor = F.mse_loss(
                    qh_pred[reg_mask],
                    replay_data.rewards[reg_mask].detach(),
                )
                head_loss = head_loss + self._reg_lambda * reg_loss

            # Critic update: Q^C(s, a) vs. target minus head component so the
            # critic learns only the action-specific value Q^C.
            current_qc = self.critic(replay_data.observations, replay_data.actions)
            critic_target_val = (target_q - qh_next).detach()
            critic_loss: torch.Tensor = 0.5 * sum(
                F.mse_loss(qc_val, critic_target_val) for qc_val in current_qc
            )

            self.policy.head_optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            (head_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.history_encoder.parameters())
                + list(self.head_net.parameters()),
                max_norm=1.0,
            )
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.policy.head_optimizer.step()
            self.critic.optimizer.step()

            head_losses.append(head_loss.item())
            critic_losses.append(critic_loss.item())

            # Actor update: gradient through Q^C only (not Q^H).
            qc_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qc_pi, _ = torch.min(qc_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qc_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                utils.polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                utils.polyak_update(
                    self.head_net.parameters(),
                    self.head_net_target.parameters(),
                    self.tau,
                )
                utils.polyak_update(
                    self.history_encoder.parameters(),
                    self.history_encoder_target.parameters(),
                    self.tau,
                )
                utils.polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/head_loss", np.mean(head_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _setup_model(self) -> None:
        super()._setup_model()
        # Convenience aliases mirroring SAC's actor/critic aliases.
        self.head_net = self.policy.head_net
        self.head_net_target = self.policy.head_net_target
        self.history_encoder = self.policy.history_encoder
        self.history_encoder_target = self.policy.history_encoder_target


# ---------------------------------------------------------------------------
# HCSACPolicy
# ---------------------------------------------------------------------------


class HCSACPolicy(SACPolicy):
    """SAC policy with HC head networks appended to the standard critic.

    Adds a GRU history encoder, a head network Q^H, and their target copies
    on top of the standard SACPolicy critic.  The standard critic serves as
    Q^C; the actor uses only Q^C for gradient computation.
    """

    def __init__(
        self,
        *args: Any,
        max_delay: int = 3,
        history_hidden_size: int = 128,
        **kwargs: Any,
    ) -> None:
        # Store before super().__init__() because _build() is called inside it.
        self._max_delay = max_delay
        self._history_hidden_size = history_hidden_size
        super().__init__(*args, **kwargs)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        if hasattr(self, "history_encoder"):
            self.history_encoder.train(mode)
        if hasattr(self, "head_net"):
            self.head_net.train(mode)
        # Target networks must always be in eval mode.
        if hasattr(self, "history_encoder_target"):
            self.history_encoder_target.eval()
        if hasattr(self, "head_net_target"):
            self.head_net_target.eval()

    def _build(self, lr_schedule: Any) -> None:  # type: ignore[override]
        super()._build(lr_schedule)
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        sa_dim = obs_dim + act_dim

        self.history_encoder = _HistoryEncoder(sa_dim, self._history_hidden_size).to(
            self.device
        )
        self.history_encoder_target = copy.deepcopy(self.history_encoder)
        self.history_encoder_target.eval()

        self.head_net = _HeadNetwork(self._history_hidden_size).to(self.device)
        self.head_net_target = copy.deepcopy(self.head_net)
        self.head_net_target.eval()

        # Share an optimizer so both networks train together on the head loss.
        self.head_optimizer = self.optimizer_class(
            list(self.history_encoder.parameters()) + list(self.head_net.parameters()),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


# ---------------------------------------------------------------------------
# HCReplayBuffer
# ---------------------------------------------------------------------------


class HCReplayBuffer(buffers.ReplayBuffer):
    """Replay buffer that stores signal-interval history alongside transitions.

    For each stored transition (s_t, a_t, r_t, s_{t+1}), the buffer records:
    - ``_history_sa``: the (s, a) pairs from the start of the current signal
      interval up to but not including t, left-zero-padded to ``max_delay``
    - ``_interval_ends``: whether this step was an interval boundary, read
      from ``info["interval_end"]`` (set by ``ImputeMissingRewardWrapper``)

    History resets at interval boundaries and at episode termination.

    Args:
        *args: Forwarded to ``ReplayBuffer``.
        max_delay: Maximum signal interval lengtorch.  Determines history window
            size.  Should match the delay in ``DelayedRewardWrapper``.
        **kwargs: Forwarded to ``ReplayBuffer``.
    """

    def __init__(
        self,
        *args: Any,
        max_delay: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_delay = max_delay
        obs_dim = int(np.prod(self.obs_shape))
        self._sa_dim = obs_dim + self.action_dim

        # Per-position history: (buffer_size, n_envs, max_delay, sa_dim).
        self._history_sa = np.zeros(
            (self.buffer_size, self.n_envs, max_delay, self._sa_dim),
            dtype=np.float32,
        )
        # Per-position interval-end flags: (buffer_size, n_envs, 1).
        self._interval_ends = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        # Rolling deque per env; tracks (s ‖ a) pairs since the last interval
        # boundary.  maxlen ensures the window never exceeds max_delay.
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
        # Snapshot history and interval_end flag BEFORE calling super() so
        # the stored entry does not include the current (obs_t, action_t).
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

        # Use info["interval_end"] rather than reward != 0 so that
        # zero-sum signal intervals reset the deques correctly.
        for env_idx in range(self.n_envs):
            if infos[env_idx].get("interval_end", False) or bool(done_arr[env_idx]):
                self._recent_sa[env_idx].clear()
            else:
                sa = np.concatenate([obs_arr[env_idx], act_arr[env_idx]])
                self._recent_sa[env_idx].append(sa)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[Any] = None,
    ) -> HCReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
        )
        obs_t, act_t, next_obs_t, dones_t, rewards_t = tuple(map(self.to_torch, data))
        history_tensor = self.to_torch(self._history_sa[batch_inds, env_indices])
        interval_ends_tensor = self.to_torch(
            self._interval_ends[batch_inds, env_indices]
        )
        return HCReplayBufferSamples(
            observations=obs_t,
            actions=act_t,
            next_observations=next_obs_t,
            dones=dones_t,
            rewards=rewards_t,
            history=history_tensor,
            interval_ends=interval_ends_tensor,
        )


# ---------------------------------------------------------------------------
# Private network modules
# ---------------------------------------------------------------------------


class _HistoryEncoder(nn.Module):
    """GRU that encodes a padded (s, a) sequence into a history embedding.

    Processes the full padded window (left-zero-padded); the final hidden
    state summarises the actual signal-interval history.
    """

    def __init__(self, sa_dim: int, hidden_size: int) -> None:
        super().__init__()
        self._gru = nn.GRU(input_size=sa_dim, hidden_size=hidden_size, batch_first=True)

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history: (batch, max_delay, sa_dim) — left-zero-padded.

        Returns:
            output: (batch, max_delay, hidden_size)
            h_n: (1, batch, hidden_size) — final hidden state.
        """
        return self._gru(history)  # type: ignore[no-any-return]


class _HeadNetwork(nn.Module):
    """Shallow MLP that maps a GRU hidden state to a scalar Q^H value."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_size)

        Returns:
            q_h: (batch, 1)
        """
        return self._net(hidden)
