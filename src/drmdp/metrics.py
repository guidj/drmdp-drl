"""Evaluation metrics for reward prediction."""

import numpy as np


def rmse(v_pred: np.ndarray, v_true: np.ndarray, axis: int) -> float:
    """Compute root mean squared error between predictions and targets along an axis."""
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
        )
    delta = v_pred - v_true
    sq = delta * delta  # np.power(delta, 2)
    sqsum = np.sum(sq, axis=axis) / np.shape(v_pred)[axis]
    sqsqrt: float = np.sqrt(sqsum)
    return sqsqrt
