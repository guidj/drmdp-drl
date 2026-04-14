import numpy as np
import pytest

from drmdp import metrics


class TestRmse:
    def test_identical_arrays_return_zero(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = metrics.rmse(arr, arr, axis=0)
        np.testing.assert_allclose(result, np.zeros(2), atol=1e-6)

    def test_known_error_along_axis0(self):
        v_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        v_true = np.zeros((2, 2))
        # axis=0: [(1^2+3^2)/2, (2^2+4^2)/2] = [5, 10], sqrt = [sqrt(5), sqrt(10)]
        result = metrics.rmse(v_pred, v_true, axis=0)
        expected = np.sqrt(np.array([5.0, 10.0]))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_known_error_along_axis1(self):
        v_pred = np.array([[1.0, 3.0]])
        v_true = np.zeros((1, 2))
        # axis=1: [(1^2+3^2)/2] = [5], sqrt = [sqrt(5)]
        result = metrics.rmse(v_pred, v_true, axis=1)
        expected = np.array([np.sqrt(5.0)])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_shape_mismatch_raises_value_error(self):
        v_pred = np.array([1.0, 2.0, 3.0])
        v_true = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            metrics.rmse(v_pred, v_true, axis=0)

    def test_single_element_array(self):
        v_pred = np.array([2.0])
        v_true = np.array([0.0])
        result = metrics.rmse(v_pred, v_true, axis=0)
        np.testing.assert_allclose(result, 2.0, atol=1e-6)
