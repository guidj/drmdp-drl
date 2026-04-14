import numpy as np
import pytest

from drmdp import optsol


def test_streaming_mean_estimator():
    xs = np.random.rand(100_000)
    estimator = optsol.StreamingMean()
    assert estimator.count == 0
    assert estimator.mean is None

    for val in xs:
        estimator.add(val)
    assert estimator.count == 100_000
    np.testing.assert_almost_equal(estimator.mean, 0.5, decimal=2)


class TestMultivariateNormalPerturbCovarianceMatrix:
    def test_output_has_positive_eigenvalues(self):
        # A matrix with a near-zero eigenvalue
        cov = np.diag([1.0, 0.0])
        perturbed = optsol.MultivariateNormal.perturb_covariance_matrix(cov, noise=1e-4)
        eigenvalues = np.linalg.eigvalsh(perturbed)
        assert np.all(eigenvalues > 0)

    def test_small_eigenvalues_clamped_to_noise(self):
        noise = 1e-3
        cov = np.diag([1.0, 1e-10])
        perturbed = optsol.MultivariateNormal.perturb_covariance_matrix(
            cov, noise=noise
        )
        eigenvalues = np.linalg.eigvalsh(perturbed)
        assert np.all(eigenvalues >= noise - 1e-9)


class TestMultivariateNormalLeastSquares:
    def test_basic_solution_pseudo_inverse(self):
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([2.0, 3.0])
        result = optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="pseudo")
        assert result is not None
        np.testing.assert_allclose(result.mean, [2.0, 3.0], atol=1e-6)

    def test_exact_inverse_mode(self):
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([1.0, 4.0])
        result = optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="exact")
        assert result is not None
        np.testing.assert_allclose(result.mean, [1.0, 4.0], atol=1e-6)

    def test_unknown_inverse_raises_value_error(self):
        matrix = np.eye(2)
        rhs = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="unknown")

    def test_returns_multivariate_normal_instance(self):
        matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        rhs = np.array([3.0, 4.0])
        result = optsol.MultivariateNormal.least_squares(matrix, rhs)
        assert isinstance(result, optsol.MultivariateNormal)
        assert result.mean.shape == (2,)
        assert result.cov.shape == (2, 2)


class TestMultivariateNormalBayesLinearRegression:
    def test_posterior_updates_mean(self):
        # Prior: mean=[0,0], cov=I
        prior = optsol.MultivariateNormal(
            mean=np.zeros(2, dtype=np.float64),
            cov=np.eye(2, dtype=np.float64),
        )
        # Single data point: X=[[1,0]], y=[3]
        matrix = np.array([[1.0, 0.0]])
        rhs = np.array([3.0])
        result = optsol.MultivariateNormal.bayes_linear_regression(matrix, rhs, prior)
        assert result is not None
        # Posterior mean should shift toward the data
        assert result.mean[0] > 0.0

    def test_posterior_covariance_shrinks_with_data(self):
        prior = optsol.MultivariateNormal(
            mean=np.zeros(2, dtype=np.float64),
            cov=np.eye(2, dtype=np.float64),
        )
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([1.0, 1.0])
        result = optsol.MultivariateNormal.bayes_linear_regression(matrix, rhs, prior)
        assert result is not None
        # Posterior variance should be smaller than prior variance (cov = pinv(I + X'X))
        prior_variance = np.diag(prior.cov)
        posterior_variance = np.diag(result.cov)
        assert np.all(posterior_variance < prior_variance)


class TestSolveLeastSquares:
    def test_basic_solution_exact(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([5.0, 7.0])
        solution = optsol.solve_least_squares(matrix, rhs)
        np.testing.assert_allclose(solution, [5.0, 7.0], atol=1e-6)

    def test_overdetermined_system(self):
        # 3 equations, 2 unknowns: least-squares solution
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rhs = np.array([1.0, 2.0, 3.0])
        solution = optsol.solve_least_squares(matrix, rhs)
        assert solution.shape == (2,)
        # Verify residual is minimised (solution satisfies normal equations)
        residual = matrix @ solution - rhs
        np.testing.assert_allclose(matrix.T @ residual, np.zeros(2), atol=1e-6)


class TestSolveConvexLeastSquares:
    def test_unconstrained_matches_lstsq(self):
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([2.0, 3.0])
        result = optsol.solve_convex_least_squares(
            matrix, rhs, constraint_fn=lambda x: []
        )
        np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-4)

    def test_non_negativity_constraint_satisfied(self):
        # Unconstrained minimum of (x - (-1))^2 is x=-1; constrained to x>=0 → x=0
        matrix = np.array([[1.0]])
        rhs = np.array([-1.0])
        result = optsol.solve_convex_least_squares(
            matrix, rhs, constraint_fn=lambda x: [x >= 0]
        )
        np.testing.assert_allclose(result, [0.0], atol=1e-4)


class TestMatrixFactorsRank:
    def test_all_nonzero_columns_counted(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        rank = optsol.matrix_factors_rank(matrix)
        assert rank == 2

    def test_zero_columns_not_counted(self):
        matrix = np.array([[1.0, 0.0], [1.0, 0.0]])
        rank = optsol.matrix_factors_rank(matrix)
        assert rank == 1

    def test_mixed_zero_and_nonzero_columns(self):
        matrix = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        rank = optsol.matrix_factors_rank(matrix)
        assert rank == 1


class TestConstantLRSchedule:
    def test_always_returns_initial_lr(self):
        schedule = optsol.ConstantLRSchedule(initial_lr=0.01)
        assert schedule.schedule() == 0.01
        assert schedule.schedule(episode=5) == 0.01
        assert schedule.schedule(step=100) == 0.01

    def test_callable_interface_returns_same_value(self):
        schedule = optsol.ConstantLRSchedule(initial_lr=0.001)
        assert schedule(episode=10, step=50) == 0.001


class TestMultivariateNormalConvexLeastSquares:
    def test_basic_solution_pseudo_inverse(self):
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([3.0, 4.0])
        result = optsol.MultivariateNormal.convex_least_squares(
            matrix, rhs, constraint_fn=lambda x: []
        )
        assert result is not None
        np.testing.assert_allclose(result.mean, [3.0, 4.0], atol=1e-4)

    def test_exact_inverse_mode(self):
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([1.0, 2.0])
        result = optsol.MultivariateNormal.convex_least_squares(
            matrix, rhs, constraint_fn=lambda x: [], inverse="exact"
        )
        assert result is not None
        np.testing.assert_allclose(result.mean, [1.0, 2.0], atol=1e-4)

    def test_unknown_inverse_raises_value_error(self):
        matrix = np.eye(2)
        rhs = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            optsol.MultivariateNormal.convex_least_squares(
                matrix, rhs, constraint_fn=lambda x: [], inverse="bad"
            )

    def test_non_negativity_constraint_satisfied(self):
        # Unconstrained minimum at x=-1; with x>=0, solution should be near 0
        matrix = np.array([[1.0]])
        rhs = np.array([-1.0])
        result = optsol.MultivariateNormal.convex_least_squares(
            matrix, rhs, constraint_fn=lambda x: [x >= 0]
        )
        assert result is not None
        np.testing.assert_allclose(result.mean, [0.0], atol=1e-4)

    def test_with_warm_start(self):
        matrix = np.eye(2, dtype=np.float64)
        rhs = np.array([2.0, 3.0])
        warm_start = np.array([2.0, 3.0])
        result = optsol.MultivariateNormal.convex_least_squares(
            matrix, rhs, constraint_fn=lambda x: [], warm_start_initial_guess=warm_start
        )
        assert result is not None
        np.testing.assert_allclose(result.mean, [2.0, 3.0], atol=1e-4)


class TestMultivariateNormalLeastSquaresSingular:
    def test_singular_matrix_with_exact_inverse_raises(self):
        # scipy.linalg.inv raises LinAlgError for singular matrices; the code re-raises it
        # because the error message is "singular matrix" (lowercase) not "Singular matrix"
        matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        rhs = np.array([1.0, 0.0])
        with pytest.raises(Exception):
            optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="exact")
