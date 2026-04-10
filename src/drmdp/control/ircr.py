"""
IRCR: Iterative Relative Credit Refinement reward model.

Non-parametric guidance rewards computed as the normalised mean episode
return of the K nearest stored (state, action) pairs (KNN over a trajectory
database).  No neural network is required; the model improves as more
trajectories are added to the database.

Reference: "Guidance Rewards via Trajectory Smoothing" (IRCR paper).
"""

from typing import List, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import KDTree

from drmdp.control import base


class IRCRRewardModel(base.RewardModel):
    """Guidance rewards via KNN lookup over a trajectory database.

    For a query (s, a), the guidance reward is the mean episode return of
    the K nearest stored (s, a) pairs, normalised to [0, 1] using the
    observed return range.

    Attributes:
        max_buffer_size: Maximum number of trajectories to retain.
        k_neighbors: Number of nearest neighbours used to compute the
            guidance reward for each query point.
    """

    def __init__(self, max_buffer_size: int = 200, k_neighbors: int = 5):
        self._max_buffer_size = max_buffer_size
        self._k_neighbors = k_neighbors
        self._trajectories: List[base.Trajectory] = []
        # Flat matrix of all (s, a) pairs across stored trajectories.
        self._sa_matrix: Optional[np.ndarray] = None  # (N_total, obs_dim + act_dim)
        # Maps each row in _sa_matrix back to its trajectory index.
        self._traj_indices: Optional[np.ndarray] = None  # (N_total,)
        self._tree: Optional[KDTree] = None
        self._r_min: float = 0.0
        self._r_max: float = 1.0

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Return normalised guidance rewards for a batch of (s, a) pairs.

        Returns zeros for all queries when the trajectory database is empty.
        """
        if self._tree is None or self._sa_matrix is None or self._traj_indices is None:
            return np.zeros(len(observations), dtype=np.float32)

        query = np.concatenate([observations, actions], axis=-1).astype(np.float64)
        # Clamp k to the number of available (s, a) points.
        k_effective = min(self._k_neighbors, len(self._sa_matrix))
        _, neighbor_indices = self._tree.query(query, k=k_effective)

        # scipy returns shape (T,) when k=1; normalise to (T, k).
        if k_effective == 1:
            neighbor_indices = neighbor_indices[:, np.newaxis]

        traj_returns = np.array(
            [traj.episode_return for traj in self._trajectories], dtype=np.float64
        )
        # Average returns of the trajectories that own the k nearest points.
        raw: np.ndarray = traj_returns[self._traj_indices[neighbor_indices]].mean(
            axis=-1
        )

        denom = max(self._r_max - self._r_min, 1e-8)
        return ((raw - self._r_min) / denom).astype(np.float32)

    def update(
        self,
        trajectories: Sequence[base.Trajectory],
    ) -> Mapping[str, float]:
        """Add trajectories to the database and rebuild the KNN index.

        Oldest trajectories are evicted when the database exceeds
        `max_buffer_size`.
        """
        self._trajectories.extend(trajectories)
        if len(self._trajectories) > self._max_buffer_size:
            self._trajectories = self._trajectories[-self._max_buffer_size :]
        self._rebuild_index()
        return {
            "buffer_size": float(len(self._trajectories)),
            "r_min": self._r_min,
            "r_max": self._r_max,
        }

    def _rebuild_index(self) -> None:
        """Flatten all stored (s, a) pairs into a matrix and build KDTree."""
        sa_parts: List[np.ndarray] = []
        traj_idx_parts: List[np.ndarray] = []
        for traj_idx, traj in enumerate(self._trajectories):
            sa = np.concatenate([traj.observations, traj.actions], axis=-1)
            sa_parts.append(sa)
            traj_idx_parts.append(
                np.full(len(traj.observations), traj_idx, dtype=np.int64)
            )
        self._sa_matrix = np.concatenate(sa_parts, axis=0).astype(np.float64)
        self._traj_indices = np.concatenate(traj_idx_parts, axis=0)
        self._tree = KDTree(self._sa_matrix)
        self._r_min = float(min(traj.episode_return for traj in self._trajectories))
        self._r_max = float(max(traj.episode_return for traj in self._trajectories))
