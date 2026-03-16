import numpy as np

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
