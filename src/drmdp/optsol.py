import abc
import dataclasses
from typing import Callable, Optional, Sequence

import cvxpy as cp
import cvxpy.error
import numpy as np
from scipy import linalg


@dataclasses.dataclass
class MultivariateNormal:
    """
    A multivariate normal distribution.
    """

    mean: np.ndarray
    cov: np.ndarray

    @staticmethod
    def perturb_covariance_matrix(cov, noise: float = 1e-6):
        eig_values, eig_matrix = np.linalg.eig(cov)
        perturbed_eig_values = np.maximum(
            eig_values, np.array([noise] * len(eig_values))
        )
        return eig_matrix * np.diag(perturbed_eig_values) * np.linalg.inv(eig_matrix)

    @classmethod
    def least_squares(
        cls, matrix, rhs, inverse: str = "pseudo"
    ) -> Optional["MultivariateNormal"]:
        """
        Least-squares estimation: mean and covariance.
        """
        if inverse == "exact":
            inverse_op = linalg.inv
        elif inverse == "pseudo":
            inverse_op = linalg.pinv
        else:
            raise ValueError(f"Unknown inverse: {inverse}")

        coeff = solve_least_squares(matrix, rhs)
        try:
            # Σᵦ = (Σᵦ^-1 + X'X)^-1 * σ²
            # Sigma^2 is the variance of the error term
            # Here, we assume sigma^2 = 1
            cov = inverse_op(np.matmul(matrix.T, matrix))
        except linalg.LinAlgError as err:
            if "Singular matrix" in err.args[0]:
                return None
            raise err
        return MultivariateNormal(coeff, cov)

    @classmethod
    def convex_least_squares(
        cls,
        matrix,
        rhs,
        constraint_fn: Callable[[cp.Variable], Sequence],
        warm_start_initial_guess: Optional[np.ndarray] = None,
        inverse: str = "pseudo",
    ) -> Optional["MultivariateNormal"]:
        """
        Least-squares estimation: mean and covariance.
        """
        if inverse == "exact":
            inverse_op = linalg.inv
        elif inverse == "pseudo":
            inverse_op = linalg.pinv
        else:
            raise ValueError(f"Unknown inverse: {inverse}")

        coeff = solve_convex_least_squares(
            matrix=matrix,
            rhs=rhs,
            constraint_fn=constraint_fn,
            warm_start_initial_guess=warm_start_initial_guess,
        )
        try:
            # Σᵦ = (Σᵦ^-1 + X'X)^-1 * σ²
            # Sigma^2 is the variance of the error term
            # Here, we assume sigma^2 = 1
            cov = inverse_op(np.matmul(matrix.T, matrix))
        except linalg.LinAlgError as err:
            if "Singular matrix" in err.args[0]:
                return None
            raise err
        return MultivariateNormal(coeff, cov)

    @classmethod
    def bayes_linear_regression(
        cls, matrix, rhs, prior: "MultivariateNormal"
    ) -> Optional["MultivariateNormal"]:
        """
        Bayesian least-squares estimation: mean and covariance.
        """
        matrix = matrix.astype(prior.mean.dtype)
        rhs = rhs.astype(prior.mean.dtype)
        try:
            # Σᵦ_new = (Σᵦ^-1 + X'X)^-1 * σ²
            # μᵦ_new = (Σᵦ^-1 + X'X)^-1 * (Σᵦ^-1*μᵦ + X'Y)
            # Sigma^2 is the variance of the error term
            # Here, we assume sigma^2 = 1
            inv_prior_sigma = linalg.pinv(prior.cov)
            cov = linalg.pinv(inv_prior_sigma + np.matmul(matrix.T, matrix))
            coeff = np.matmul(
                cov, (np.matmul(inv_prior_sigma, prior.mean) + np.matmul(matrix.T, rhs))
            )
            return MultivariateNormal(coeff, cov)
        except linalg.LinAlgError as err:
            raise ValueError("Failed Bayesian estimation") from err


class LearningRateSchedule(abc.ABC):
    """
    This class updates the learning rate based on the episode,
    step or both - using a given `schedule` function.
    """

    def __init__(self, initial_lr: float):
        super().__init__()
        self.initial_lr = initial_lr

    @abc.abstractmethod
    def schedule(self, episode: Optional[int] = None, step: Optional[int] = None):
        pass

    def __call__(self, episode: int, step: int):
        return self.schedule(episode, step)


class ConstantLRSchedule(LearningRateSchedule):
    def schedule(self, episode=None, step=None):
        del episode
        del step
        return self.initial_lr


class StreamingMean:
    """
    Calculates a mean value in streaming mode.
    """

    def __init__(self):
        self.mean: Optional[float] = None
        self.count: int = 0

    def add(self, value: float) -> float:
        """
        Update's the mean with a new value
        """
        self.count += 1
        if self.mean is None:
            self.mean = value
        else:
            delta = value - self.mean
            self.mean = self.mean + delta / self.count
        return self.mean


def solve_least_squares(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solves a least-squares problem Ax = b.
    `A` is matrix.
    `b` is rhs (right-hand side).
    """
    try:
        solution, _, _, _ = np.linalg.lstsq(a=matrix, b=rhs, rcond=None)
        return solution  # type: ignore
    except np.linalg.LinAlgError as err:
        # the computation failed, likely due to the matix being unsuitable (no solution).
        raise ValueError("Failed to solve linear system") from err


def solve_convex_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
    constraint_fn: Callable[[cp.Variable], Sequence],
    warm_start_initial_guess: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solves Least Squares with convex optimisation,
    with added constraints.
    """
    solution = cp.Variable(matrix.shape[-1], value=warm_start_initial_guess)
    objective = cp.Minimize(cp.sum_squares(matrix @ solution - rhs))
    prob = cp.Problem(objective, constraint_fn(solution))

    try:
        prob.solve(warm_start=True)
        if prob.status == cp.OPTIMAL:
            return solution.value  # type: ignore
        raise ValueError(f"No Solution. Status: {prob.status}; Sol: {solution.value}")
    except cvxpy.error.SolverError as err:
        raise ValueError(f"No Solution. Status {prob.status}") from err


def matrix_factors_rank(matrix: np.ndarray) -> int:
    """
    Calculates factors present in columns
    """
    factors_sum = np.sum(matrix != 0, axis=0)
    factors_present: int = np.sum(factors_sum > 0).astype(np.int64).item()
    return factors_present
