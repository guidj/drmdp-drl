"""
IRCR: Iterative Relative Credit Refinement reward model.

Guidance rewards via dual-buffer credit assignment: a FIFO ring buffer of
recent transitions and a min-heap buffer of the best trajectories by total
return.  Each transition carries a credit equal to the mean per-step return
of its source trajectory.  Guidance rewards are the min-max normalised
credits drawn 50/50 from both buffers.

Reference: Gangwani & Peng, "Learning Guidance Rewards with
Trajectory-space Smoothing", NeurIPS 2020.
"""

import heapq
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from stable_baselines3.common import buffers
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from drmdp.control import base


class FIFOTransitionBuffer:
    """Fixed-capacity ring buffer of individual transitions.

    Each transition stores (obs, next_obs, action, done, credit) where
    credit = episode_return / episode_length for the source trajectory.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_steps: int):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_steps = max_steps

        self.obs = np.zeros((max_steps, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_steps, action_dim), dtype=np.float32)
        self.dones = np.zeros((max_steps, 1), dtype=np.uint8)
        self.credits = np.zeros((max_steps, 1), dtype=np.float32)

        self.min_credit_val: Optional[float] = None
        self.max_credit_val: Optional[float] = None

        self._filled_i = 0
        self._curr_i = 0

    def __len__(self) -> int:
        return self._filled_i

    def add_trajectories(self, trajectories: Sequence[base.Trajectory]) -> None:
        """Flatten trajectories to transitions and append to the ring buffer.

        Each transition receives credit = episode_return / episode_length.
        When the new batch exceeds remaining capacity the buffer rolls so
        the write cursor resets to zero and the oldest data is overwritten.
        """
        all_obs: List[np.ndarray] = []
        all_next_obs: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        all_dones: List[np.ndarray] = []
        all_credits: List[np.ndarray] = []

        for traj in trajectories:
            num_steps = len(traj.observations)
            next_obs = np.concatenate(
                [traj.observations[1:], traj.observations[-1:]], axis=0
            )
            dones = np.zeros((num_steps, 1), dtype=np.uint8)
            if traj.terminals[-1]:
                dones[-1] = 1

            credit = traj.episode_return / max(num_steps, 1)
            credits = np.full((num_steps, 1), credit, dtype=np.float32)

            all_obs.append(traj.observations)
            all_next_obs.append(next_obs)
            all_actions.append(traj.actions)
            all_dones.append(dones)
            all_credits.append(credits)

        if len(all_obs) == 0:
            return

        concat_obs = np.concatenate(all_obs, axis=0)
        concat_next_obs = np.concatenate(all_next_obs, axis=0)
        concat_actions = np.concatenate(all_actions, axis=0)
        concat_dones = np.concatenate(all_dones, axis=0)
        concat_credits = np.concatenate(all_credits, axis=0)

        nentries = concat_obs.shape[0]
        if self._curr_i + nentries > self._max_steps:
            rollover = self._max_steps - self._curr_i
            self.obs = np.roll(self.obs, rollover, axis=0)
            self.next_obs = np.roll(self.next_obs, rollover, axis=0)
            self.actions = np.roll(self.actions, rollover, axis=0)
            self.dones = np.roll(self.dones, rollover, axis=0)
            self.credits = np.roll(self.credits, rollover, axis=0)
            self._curr_i = 0
            self._filled_i = self._max_steps

        self.obs[self._curr_i : self._curr_i + nentries] = concat_obs
        self.next_obs[self._curr_i : self._curr_i + nentries] = concat_next_obs
        self.actions[self._curr_i : self._curr_i + nentries] = concat_actions
        self.dones[self._curr_i : self._curr_i + nentries] = concat_dones
        self.credits[self._curr_i : self._curr_i + nentries] = concat_credits

        self._curr_i += nentries
        if self._filled_i < self._max_steps:
            self._filled_i = min(self._filled_i + nentries, self._max_steps)
        if self._curr_i >= self._max_steps:
            self._curr_i = 0

        self.min_credit_val = float(self.credits[: len(self)].min())
        self.max_credit_val = float(self.credits[: len(self)].max())

    def sample(self, batch_size: int) -> Optional[Mapping[str, np.ndarray]]:
        """Sample transitions uniformly. Returns None if too few stored."""
        if len(self) < batch_size:
            return None
        inds = np.random.choice(np.arange(len(self)), size=batch_size, replace=False)
        return {
            "observations": self.obs[inds],
            "next_observations": self.next_obs[inds],
            "actions": self.actions[inds],
            "dones": self.dones[inds],
            "credits": self.credits[inds],
        }


class MinHeapTrajectoryBuffer:
    """Priority buffer retaining the best trajectories by total return.

    Uses a min-heap: incoming trajectories replace the lowest-return stored
    trajectory when their return exceeds it.  Flattened to transitions for
    sampling.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_trajs: int):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_trajs = max_trajs

        self._heap: List[List[Any]] = []
        self._traj_data: Dict[int, base.Trajectory] = {}
        self._num_trajs = 0

        self.obs: Optional[np.ndarray] = None
        self.next_obs: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.dones: Optional[np.ndarray] = None
        self.credits: Optional[np.ndarray] = None

        self.min_credit_val: Optional[float] = None
        self.max_credit_val: Optional[float] = None

    def __len__(self) -> int:
        return self.obs.shape[0] if self.obs is not None else 0

    def add_trajectories(self, trajectories: Sequence[base.Trajectory]) -> None:
        """Insert trajectories, evicting the lowest-return entry when full.

        Rebuilds the flattened transition arrays only when the stored set
        actually changes.
        """
        updated = False

        for traj in trajectories:
            priority = traj.episode_return

            if self._num_trajs < self._max_trajs:
                heapq.heappush(self._heap, [priority, self._num_trajs])
                loc = self._num_trajs
                self._num_trajs += 1
            else:
                min_priority, loc = self._heap[0]
                if priority < min_priority:
                    continue
                heapq.heappushpop(self._heap, [priority, loc])

            if loc in self._traj_data:
                del self._traj_data[loc]
            self._traj_data[loc] = traj
            updated = True

        if updated:
            self._rebuild_flat_arrays()

    def _rebuild_flat_arrays(self) -> None:
        """Flatten all heap-stored trajectories into contiguous arrays for sampling."""
        obs_parts: List[np.ndarray] = []
        next_obs_parts: List[np.ndarray] = []
        action_parts: List[np.ndarray] = []
        done_parts: List[np.ndarray] = []
        credit_parts: List[np.ndarray] = []

        for _, traj in sorted(self._traj_data.items()):
            num_steps = len(traj.observations)
            next_obs = np.concatenate(
                [traj.observations[1:], traj.observations[-1:]], axis=0
            )
            dones = np.zeros((num_steps, 1), dtype=np.uint8)
            if traj.terminals[-1]:
                dones[-1] = 1

            credit = traj.episode_return / max(num_steps, 1)
            credits = np.full((num_steps, 1), credit, dtype=np.float32)

            obs_parts.append(traj.observations)
            next_obs_parts.append(next_obs)
            action_parts.append(traj.actions)
            done_parts.append(dones)
            credit_parts.append(credits)

        if len(obs_parts) == 0:
            self.obs = None
            self.next_obs = None
            self.actions = None
            self.dones = None
            self.credits = None
            self.min_credit_val = None
            self.max_credit_val = None
            return

        self.obs = np.concatenate(obs_parts, axis=0)
        self.next_obs = np.concatenate(next_obs_parts, axis=0)
        self.actions = np.concatenate(action_parts, axis=0)
        self.dones = np.concatenate(done_parts, axis=0)
        self.credits = np.concatenate(credit_parts, axis=0)

        self.min_credit_val = float(self.credits.min())
        self.max_credit_val = float(self.credits.max())

    def sample(self, batch_size: int) -> Optional[Mapping[str, np.ndarray]]:
        """Sample transitions, with replacement when fewer than batch_size."""
        if len(self) == 0:
            return None
        assert self.obs is not None
        assert self.next_obs is not None
        assert self.actions is not None
        assert self.dones is not None
        assert self.credits is not None
        replace = len(self) < batch_size
        inds = np.random.choice(np.arange(len(self)), size=batch_size, replace=replace)
        return {
            "observations": self.obs[inds],
            "next_observations": self.next_obs[inds],
            "actions": self.actions[inds],
            "dones": self.dones[inds],
            "credits": self.credits[inds],
        }


class IRCRRewardModel(base.RewardModel):
    """Guidance rewards via dual-buffer credit assignment.

    Manages a FIFO transition buffer (recent experience) and a min-heap
    trajectory buffer (best trajectories).  Each transition carries a
    credit equal to its trajectory's mean per-step return.  Guidance
    rewards are min-max normalised credits.

    Attributes:
        fifo_capacity: Maximum transitions in the FIFO ring buffer.
        heap_capacity: Maximum trajectories in the min-heap buffer.
    """

    def __init__(
        self,
        fifo_capacity: int = 300_000,
        heap_capacity: int = 10,
        obs_dim: int = 1,
        action_dim: int = 1,
    ):
        self._fifo = FIFOTransitionBuffer(obs_dim, action_dim, fifo_capacity)
        self._heap = MinHeapTrajectoryBuffer(obs_dim, action_dim, heap_capacity)

    @property
    def r_min(self) -> float:
        vals = [
            buf.min_credit_val
            for buf in (self._fifo, self._heap)
            if buf.min_credit_val is not None
        ]
        return min(vals) if vals else 0.0

    @property
    def r_max(self) -> float:
        vals = [
            buf.max_credit_val
            for buf in (self._fifo, self._heap)
            if buf.max_credit_val is not None
        ]
        return max(vals) if vals else 1.0

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Return uniform guidance reward for each query transition.

        With flat credit assignment every transition receives the same
        normalised value.  This method is used for logging; the primary
        training path goes through ``IRCRReplayBuffer.sample()``.
        """
        del actions, terminals
        n_queries = len(observations)
        if self._fifo.min_credit_val is None:
            return np.zeros(n_queries, dtype=np.float32)

        mean_credit = float(self._fifo.credits[: len(self._fifo)].mean())
        denom = max(self.r_max - self.r_min, 1e-8)
        value = (mean_credit - self.r_min) / denom
        return np.full(n_queries, value, dtype=np.float32)

    def update(
        self,
        trajectories: Sequence[base.Trajectory],
    ) -> Mapping[str, float]:
        """Add trajectories to both buffers."""
        self._fifo.add_trajectories(trajectories)
        self._heap.add_trajectories(trajectories)
        return {
            "fifo_size": float(len(self._fifo)),
            "heap_size": float(len(self._heap)),
            "r_min": self.r_min,
            "r_max": self.r_max,
        }

    def compute_guidance_rewards(self, credits: np.ndarray) -> np.ndarray:
        """Normalise raw credits to [0, 1] using global min/max."""
        r_min = self.r_min
        r_max = self.r_max
        denom = max(r_max - r_min, 1e-8)
        result = ((credits - r_min) / denom).astype(np.float32)
        return result.squeeze(-1) if result.ndim > 1 else result

    def sample(self, batch_size: int) -> Optional[Mapping[str, np.ndarray]]:
        """Draw transitions 50/50 from FIFO and MinHeap.

        Returns None when either buffer has insufficient data (FIFO must
        have at least half_batch entries; MinHeap may use replacement
        sampling).
        """
        half = batch_size // 2
        remainder = batch_size - half

        fifo_sample = self._fifo.sample(half)
        if fifo_sample is None:
            return None
        heap_sample = self._heap.sample(remainder)
        if heap_sample is None:
            return None

        merged: Dict[str, np.ndarray] = {}
        for key in fifo_sample:
            merged[key] = np.concatenate([fifo_sample[key], heap_sample[key]], axis=0)

        merged["guidance_rewards"] = self.compute_guidance_rewards(merged["credits"])
        return merged


class IRCRReplayBuffer(buffers.ReplayBuffer):
    """Replay buffer that draws training batches from IRCR's dual buffers.

    Overrides ``sample()`` to draw 50/50 from the FIFO and MinHeap
    buffers managed by ``IRCRRewardModel``, computing normalised guidance
    rewards as the training reward signal.

    Falls back to standard replay buffer sampling (with zero rewards)
    when the IRCR buffers don't have enough transitions yet.
    """

    def __init__(
        self,
        *args: Any,
        reward_model: Optional[IRCRRewardModel] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model

    def sample(
        self,
        batch_size: int,
        env: Optional[Any] = None,
    ) -> ReplayBufferSamples:
        if self.reward_model is not None:
            ircr_batch = self.reward_model.sample(batch_size)
            if ircr_batch is not None:
                return self._to_replay_buffer_samples(ircr_batch)

        batch = super().sample(batch_size, env)
        zeroed = torch.zeros_like(batch.rewards)
        return batch._replace(rewards=zeroed)

    def _to_replay_buffer_samples(
        self, batch: Mapping[str, np.ndarray]
    ) -> ReplayBufferSamples:
        obs = torch.as_tensor(batch["observations"], device=self.device)
        next_obs = torch.as_tensor(batch["next_observations"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device).float()
        rewards = torch.as_tensor(
            batch["guidance_rewards"][:, np.newaxis], device=self.device
        )
        return ReplayBufferSamples(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
        )
