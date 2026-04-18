"""
GRD: Generative Return Decomposition.

A parametric reward model that learns explicit causal structure between
environment variables to decompose the total episode return into per-step
Markovian rewards (Zhang et al., NeurIPS 2023).

Training uses three loss terms:

    reward_loss  = MSE(sum_t r̂(C^{s→r}⊙s_t, C^{a→r}⊙a_t), R)
    dynamics_nll = MDN NLL for p(s_{t+1}[i] | C^{s→s}_{:,i}⊙s_t, C^{a→s}_{:,i}⊙a_t)
    sparsity_reg = lam_offdiag * mean(off-diagonal edges) + lam_diag * mean(diag C^{s→s})
    total_loss   = reward_loss + dyn_weight * dynamics_nll + sparsity_reg

Causal masks are sampled via Gumbel-Softmax during training (continuous
relaxation) and computed as hard argmax for prediction and the compact
observation mask.

The compact observation mask C^{s→π} is derived as the reachability closure
of reward-relevant state dimensions through the learned C^{s→s}: any state
dimension that can influence a reward-relevant dimension (directly or
transitively) is included. This mask is applied to observations at sample
time via RelabelingReplayBuffer.obs_mask so the policy trains on only the
causally relevant dimensions.
"""

import dataclasses
import math
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drmdp.control import base

# Number of MDN components for the dynamics network, per the paper (K=3).
_MDN_COMPONENTS = 3


class GRDRewardModel(base.RewardModel):
    """Causal reward model via generative return decomposition.

    Learns four binary causal masks (C^{s→r}, C^{a→r}, C^{s→s}, C^{a→s})
    jointly with a reward network and a per-dim MDN dynamics network.  The
    reward network predicts per-step rewards from causally masked (state,
    action) inputs; its predictions are supervised to sum to the total episode
    return R.

    At inference time, greedy masks select the causal dimensions; the compact
    observation mask (reachability closure of reward-relevant dims) is exposed
    via obs_mask so the policy sees only causally relevant state features.

    Attributes:
        max_buffer_size: Maximum full trajectories and transitions retained.
        train_epochs: Training epochs run on the buffer each update call.
        batch_size: Number of full trajectories per reward-loss mini-batch.
        trans_batch_size: Number of transitions per dynamics mini-batch.
        dyn_weight: Weight of the dynamics NLL term in the total loss.
        sparsity_lam_diag: Sparsity penalty for diagonal of C^{s→s}.
        sparsity_lam_offdiag: Sparsity penalty for all other causal edges.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        learning_rate: float = 3e-4,
        train_epochs: int = 10,
        batch_size: int = 4,
        trans_batch_size: int = 256,
        dyn_weight: float = 1.0,
        sparsity_lam_diag: float = 1e-4,
        sparsity_lam_offdiag: float = 1e-5,
        max_buffer_size: int = 300,
    ) -> None:
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._train_epochs = train_epochs
        self._batch_size = batch_size
        self._trans_batch_size = trans_batch_size
        self._dyn_weight = dyn_weight
        self._sparsity_lam_diag = sparsity_lam_diag
        self._sparsity_lam_offdiag = sparsity_lam_offdiag
        self._max_buffer_size = max_buffer_size

        self._causal = _CausalStructure(obs_dim, action_dim)
        self._reward_net = _RewardNetwork(
            obs_dim, action_dim, hidden_dim, num_hidden_layers
        )
        self._dyn_net = _DynamicsNetwork(
            obs_dim, action_dim, hidden_dim, num_hidden_layers
        )

        self._optimizer = torch.optim.Adam(
            list(self._causal.parameters())
            + list(self._reward_net.parameters())
            + list(self._dyn_net.parameters()),
            lr=learning_rate,
        )

        self._traj_buffer: List[base.Trajectory] = []
        self._trans_buffer: List[_Transition] = []

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Return per-step reward estimates using greedy causal masks.

        Args:
            observations: Shape (T, obs_dim).
            actions: Shape (T, action_dim).
            terminals: Shape (T,), episode-end flags.

        Returns:
            Per-step reward estimates, shape (T,), dtype float32.
        """
        self._reward_net.eval()
        with torch.no_grad():
            mask_sr, mask_ar, _, _ = self._causal.greedy_masks()
            obs_t = torch.as_tensor(observations, dtype=torch.float32)
            act_t = torch.as_tensor(actions, dtype=torch.float32)
            term_t = torch.as_tensor(
                terminals[:, np.newaxis].astype(np.float32), dtype=torch.float32
            )
            masked_obs = obs_t * mask_sr.unsqueeze(0)
            masked_act = act_t * mask_ar.unsqueeze(0)
            preds = self._reward_net(masked_obs, masked_act, term_t)
        result: np.ndarray = preds.squeeze(-1).cpu().numpy().astype(np.float32)
        return result

    def update(
        self,
        trajectories: Sequence[base.Trajectory],
    ) -> Mapping[str, float]:
        """Update from completed episode trajectories.

        Adds trajectories and pre-extracted transitions to rolling buffers,
        evicts the oldest if either buffer exceeds max_buffer_size, then
        trains for train_epochs epochs.

        Args:
            trajectories: Completed episodes to learn from.

        Returns:
            Metrics: buffer_size, training_steps, reward_loss, dyn_loss,
            sparsity_reg (averages over the final training epoch).
        """
        for trajectory in trajectories:
            self._traj_buffer.append(trajectory)
            self._trans_buffer.extend(_extract_transitions(trajectory))

        if len(self._traj_buffer) > self._max_buffer_size:
            self._traj_buffer = self._traj_buffer[-self._max_buffer_size :]
        if len(self._trans_buffer) > self._max_buffer_size:
            self._trans_buffer = self._trans_buffer[-self._max_buffer_size :]

        if not self._traj_buffer:
            return {
                "buffer_size": 0.0,
                "training_steps": 0.0,
                "reward_loss": 0.0,
                "dyn_loss": 0.0,
                "sparsity_reg": 0.0,
            }

        total_training_steps = 0
        last_reward_losses: List[float] = []
        last_dyn_losses: List[float] = []
        last_sparsity_regs: List[float] = []

        self._reward_net.train()
        self._dyn_net.train()

        for epoch_idx in range(self._train_epochs):
            traj_indices = np.random.permutation(len(self._traj_buffer))
            epoch_reward_losses: List[float] = []
            epoch_dyn_losses: List[float] = []
            epoch_sparsity_regs: List[float] = []

            for batch_start in range(0, len(self._traj_buffer), self._batch_size):
                batch_traj_indices = traj_indices[
                    batch_start : batch_start + self._batch_size
                ]
                traj_batch = [self._traj_buffer[idx] for idx in batch_traj_indices]

                # --- reward loss (L_rew) ---
                mask_sr, mask_ar, _, _ = self._causal.sample_gumbel()
                sum_preds: List[torch.Tensor] = []
                episode_returns: List[float] = []

                for trajectory in traj_batch:
                    obs_t = torch.as_tensor(
                        trajectory.observations, dtype=torch.float32
                    )
                    act_t = torch.as_tensor(trajectory.actions, dtype=torch.float32)
                    term_t = torch.as_tensor(
                        trajectory.terminals[:, np.newaxis].astype(np.float32),
                        dtype=torch.float32,
                    )
                    masked_obs = obs_t * mask_sr.unsqueeze(0)
                    masked_act = act_t * mask_ar.unsqueeze(0)
                    preds = self._reward_net(masked_obs, masked_act, term_t)  # (T, 1)
                    sum_preds.append(preds.sum())
                    episode_returns.append(trajectory.episode_return)
                    if epoch_idx == self._train_epochs - 1:
                        total_training_steps += len(trajectory.observations)

                sum_preds_t = torch.stack(sum_preds)
                returns_t = torch.tensor(episode_returns, dtype=torch.float32)
                reward_loss = F.mse_loss(sum_preds_t, returns_t)

                # --- dynamics loss (L_dyn) ---
                trans_count = min(self._trans_batch_size, len(self._trans_buffer))
                trans_indices = np.random.choice(
                    len(self._trans_buffer), size=trans_count, replace=False
                )
                trans_batch = [self._trans_buffer[idx] for idx in trans_indices]

                obs_batch = torch.as_tensor(
                    np.stack([tr.obs for tr in trans_batch]), dtype=torch.float32
                )
                act_batch = torch.as_tensor(
                    np.stack([tr.action for tr in trans_batch]), dtype=torch.float32
                )
                next_obs_batch = torch.as_tensor(
                    np.stack([tr.next_obs for tr in trans_batch]), dtype=torch.float32
                )

                _, _, mask_ss, mask_as = self._causal.sample_gumbel()
                dyn_loss = self._dyn_net.nll(
                    obs_batch, act_batch, next_obs_batch, mask_ss, mask_as
                )

                # --- sparsity regularisation (L_reg) ---
                sparsity_reg = _sparsity_reg(
                    self._causal, self._sparsity_lam_diag, self._sparsity_lam_offdiag
                )

                total_loss = reward_loss + self._dyn_weight * dyn_loss + sparsity_reg

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                epoch_reward_losses.append(reward_loss.item())
                epoch_dyn_losses.append(dyn_loss.item())
                epoch_sparsity_regs.append(sparsity_reg.item())

            if epoch_idx == self._train_epochs - 1:
                last_reward_losses = epoch_reward_losses
                last_dyn_losses = epoch_dyn_losses
                last_sparsity_regs = epoch_sparsity_regs

        return {
            "buffer_size": float(len(self._traj_buffer)),
            "training_steps": float(total_training_steps),
            "reward_loss": float(np.mean(last_reward_losses)),
            "dyn_loss": float(np.mean(last_dyn_losses)),
            "sparsity_reg": float(np.mean(last_sparsity_regs)),
        }

    @property
    def compact_obs_mask(self) -> np.ndarray:
        """Binary mask of causally reward-relevant observation dimensions.

        Computes the reachability closure: starts with dimensions directly
        selected by C^{s→r}, then transitively adds source dimensions via
        C^{s→s} until convergence. Shape (obs_dim,).
        """
        with torch.no_grad():
            mask_sr, _, mask_ss, _ = self._causal.greedy_masks()
        mask_sr_np = mask_sr.cpu().numpy()
        mask_ss_np = mask_ss.cpu().numpy()
        return _compute_compact_mask(mask_sr_np, mask_ss_np)

    @property
    def obs_mask(self) -> Optional[np.ndarray]:
        """Compact observation mask forwarded to RelabelingReplayBuffer."""
        return self.compact_obs_mask


class _CausalStructure(nn.Module):
    """Learnable causal edge probabilities as unconstrained logit parameters.

    Each potential edge is parameterised by a 2-element logit vector whose
    softmax gives the Bernoulli probability of that edge being active.
    Gumbel-Softmax provides a differentiable relaxation during training;
    hard argmax gives binary masks at inference time.

    Attributes:
        phi_sr: State→reward edge logits, shape (obs_dim, 2).
        phi_ar: Action→reward edge logits, shape (action_dim, 2).
        phi_ss: State→state edge logits, shape (obs_dim, obs_dim, 2).
            Entry [j, i] = edge s_j → s_i.
        phi_as: Action→state edge logits, shape (action_dim, obs_dim, 2).
            Entry [j, i] = edge a_j → s_i.
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.phi_sr = nn.Parameter(torch.zeros(obs_dim, 2))
        self.phi_ar = nn.Parameter(torch.zeros(action_dim, 2))
        self.phi_ss = nn.Parameter(torch.zeros(obs_dim, obs_dim, 2))
        self.phi_as = nn.Parameter(torch.zeros(action_dim, obs_dim, 2))

    def sample_gumbel(
        self, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample soft binary masks via Gumbel-Softmax.

        Args:
            temperature: Gumbel-Softmax temperature; lower → harder.

        Returns:
            (mask_sr, mask_ar, mask_ss, mask_as) — soft masks in [0, 1] with
            the same shapes as the corresponding phi tensors minus the last dim.
        """
        mask_sr = F.gumbel_softmax(self.phi_sr, tau=temperature, hard=False)[..., 1]
        mask_ar = F.gumbel_softmax(self.phi_ar, tau=temperature, hard=False)[..., 1]
        mask_ss = F.gumbel_softmax(self.phi_ss, tau=temperature, hard=False)[..., 1]
        mask_as = F.gumbel_softmax(self.phi_as, tau=temperature, hard=False)[..., 1]
        return mask_sr, mask_ar, mask_ss, mask_as

    def greedy_masks(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return hard binary masks using greedy argmax (no gradient).

        Returns:
            (mask_sr, mask_ar, mask_ss, mask_as) — binary {0, 1} tensors.
        """
        with torch.no_grad():
            mask_sr = (F.softmax(self.phi_sr, dim=-1)[..., 1] >= 0.5).float()
            mask_ar = (F.softmax(self.phi_ar, dim=-1)[..., 1] >= 0.5).float()
            mask_ss = (F.softmax(self.phi_ss, dim=-1)[..., 1] >= 0.5).float()
            mask_as = (F.softmax(self.phi_as, dim=-1)[..., 1] >= 0.5).float()
        return mask_sr, mask_ar, mask_ss, mask_as


class _RewardNetwork(nn.Module):
    """Feedforward MLP predicting per-step reward from masked (state, action, terminal).

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


class _DynamicsNetwork(nn.Module):
    """Per-dimension MDN dynamics network with causal masking.

    Predicts each target state dimension s_{t+1}[i] from its own column of
    causal masks: the source dimensions selected by C^{s→s}_{:,i} and
    C^{a→s}_{:,i}. A 3-component mixture density network is used per dim,
    matching the paper exactly.

    A single shared MLP maps (obs_dim + action_dim) → 9 (3 components ×
    (π-logit, μ, log_σ)). All target dimensions are batched together for
    efficiency: the B-example batch is reshaped to (B × obs_dim) before the
    forward pass, then folded back to (B, obs_dim, 3).

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer width.
        num_hidden_layers: Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        input_dim = obs_dim + action_dim
        layers: List[nn.Module] = []
        layer_in = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(layer_in, hidden_dim))
            layers.append(nn.ReLU())
            layer_in = hidden_dim
        self._layers = nn.Sequential(*layers)
        # 3 components × (π logit, μ, log_σ) = 9 outputs
        self._final = nn.Linear(layer_in, _MDN_COMPONENTS * 3)

    def nll(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        mask_ss: torch.Tensor,
        mask_as: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MDN negative log-likelihood with per-dim causal masking.

        For each target dimension i, the input is
        [mask_ss[:, i] ⊙ obs, mask_as[:, i] ⊙ action], so the gradient of
        the NLL with respect to phi_ss[j, i] directly measures whether source
        dim j helps predict target dim i.

        Args:
            obs: Shape (B, obs_dim).
            action: Shape (B, action_dim).
            next_obs: Shape (B, obs_dim).
            mask_ss: Shape (obs_dim, obs_dim); entry [j, i] = edge s_j → s_i.
            mask_as: Shape (action_dim, obs_dim); entry [j, i] = edge a_j → s_i.

        Returns:
            Scalar NLL averaged over batch examples and state dimensions.
        """
        batch_size = obs.shape[0]
        obs_dim = self._obs_dim

        # Per-target-dim masked inputs:
        # obs_expanded[b, i, j] = mask_ss[j, i] * obs[b, j]
        # mask_ss.T has shape (obs_dim, obs_dim); [i, j] = mask_ss[j, i]
        obs_expanded = obs.unsqueeze(1) * mask_ss.T.unsqueeze(
            0
        )  # (B, obs_dim, obs_dim)
        act_expanded = action.unsqueeze(1) * mask_as.T.unsqueeze(
            0
        )  # (B, obs_dim, action_dim)
        inputs = torch.cat(
            [obs_expanded, act_expanded], dim=-1
        )  # (B, obs_dim, obs_dim+action_dim)

        flat_inputs = inputs.reshape(
            batch_size * obs_dim, self._obs_dim + self._action_dim
        )
        raw = self._final(self._layers(flat_inputs))  # (B*obs_dim, 9)
        raw = raw.reshape(batch_size, obs_dim, _MDN_COMPONENTS, 3)

        pi_logits = raw[..., 0]  # (B, obs_dim, K)
        mu = raw[..., 1]  # (B, obs_dim, K)
        log_sigma = raw[..., 2]  # (B, obs_dim, K)

        pi = F.softmax(pi_logits, dim=-1)
        sigma = log_sigma.clamp(-5.0, 2.0).exp()

        target = next_obs.unsqueeze(-1)  # (B, obs_dim, 1)
        log_normal = (
            -0.5 * ((target - mu) / sigma).pow(2)
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        )  # (B, obs_dim, K)
        log_mix = torch.log(pi + 1e-8) + log_normal  # (B, obs_dim, K)
        return -torch.logsumexp(log_mix, dim=-1).mean()


@dataclasses.dataclass(frozen=True)
class _Transition:
    """A single (s_t, a_t, s_{t+1}) transition.

    Attributes:
        obs: Observation at step t, shape (obs_dim,).
        action: Action taken at step t, shape (action_dim,).
        next_obs: Observation at step t+1, shape (obs_dim,).
    """

    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray


def _extract_transitions(trajectory: base.Trajectory) -> List[_Transition]:
    """Extract (s_t, a_t, s_{t+1}) triples from a completed trajectory.

    The terminal step has no valid next observation, so it is skipped.

    Args:
        trajectory: A completed episode trajectory.

    Returns:
        List of transitions in chronological order.
    """
    transitions: List[_Transition] = []
    num_steps = len(trajectory.observations)
    for step_idx in range(num_steps):
        if trajectory.terminals[step_idx]:
            continue
        if step_idx + 1 >= num_steps:
            continue
        transitions.append(
            _Transition(
                obs=trajectory.observations[step_idx],
                action=trajectory.actions[step_idx],
                next_obs=trajectory.observations[step_idx + 1],
            )
        )
    return transitions


def _compute_compact_mask(
    mask_sr: np.ndarray,
    mask_ss: np.ndarray,
) -> np.ndarray:
    """Compute the reachability closure of reward-relevant observation dimensions.

    Starting from dimensions directly selected by C^{s→r}, iteratively adds
    source dimensions via columns of C^{s→s}: if dim i is already in the
    compact set and mask_ss[j, i] == 1, then dim j is also added.  Repeats
    until convergence (at most obs_dim iterations).

    Args:
        mask_sr: Binary vector of shape (obs_dim,).
        mask_ss: Binary matrix of shape (obs_dim, obs_dim); entry [j, i] = s_j→s_i.

    Returns:
        Binary compact mask of shape (obs_dim,).
    """
    compact = mask_sr.astype(bool)
    mask_bool = mask_ss.astype(bool)
    for _ in range(len(compact)):
        updated = compact | (mask_bool @ compact)
        if np.array_equal(updated, compact):
            break
        compact = updated
    return compact.astype(np.float32)


def _sparsity_reg(
    causal: _CausalStructure,
    lam_diag: float,
    lam_offdiag: float,
) -> torch.Tensor:
    """Compute the sparsity regularisation term.

    Diagonal entries of C^{s→s} correspond to state persistence connections
    (s_i → s_i), which are nearly always present and warrant a weaker penalty.
    All other edge types use the off-diagonal lambda.

    Args:
        causal: The causal structure module.
        lam_diag: Penalty weight for diagonal of C^{s→s}.
        lam_offdiag: Penalty weight for C^{s→r}, C^{a→r}, off-diag C^{s→s}, C^{a→s}.

    Returns:
        Scalar regularisation loss.
    """
    sr_prob = F.softmax(causal.phi_sr, dim=-1)[..., 1].mean()
    ar_prob = F.softmax(causal.phi_ar, dim=-1)[..., 1].mean()
    as_prob = F.softmax(causal.phi_as, dim=-1)[..., 1].mean()

    ss_probs = F.softmax(causal.phi_ss, dim=-1)[..., 1]  # (obs_dim, obs_dim)
    obs_dim = ss_probs.shape[0]
    diag_mask = torch.eye(obs_dim, device=ss_probs.device, dtype=torch.bool)

    ss_diag_prob = ss_probs[diag_mask].mean()
    if obs_dim > 1:
        ss_offdiag_prob = ss_probs[~diag_mask].mean()
    else:
        ss_offdiag_prob = torch.zeros(1, device=ss_probs.device)[0]

    offdiag_term = lam_offdiag * (sr_prob + ar_prob + as_prob + ss_offdiag_prob)
    diag_term = lam_diag * ss_diag_prob
    return offdiag_term + diag_term
