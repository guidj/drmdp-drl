import numpy as np
import pytest

from drmdp import tiles


def test_iht_init():
    iht = tiles.IHT(10)
    assert iht.size == 10
    assert iht.overfullCount == 0
    assert len(iht.dictionary) == 0


def test_iht_str():
    iht = tiles.IHT(10)
    expected = "Collision table: size:10 overfullCount:0 dictionary:0 items"
    assert str(iht) == expected


def test_iht_count():
    iht = tiles.IHT(10)
    assert iht.count() == 0
    iht.dictionary = {1: 1, 2: 2}
    assert iht.count() == 2


def test_iht_fullp():
    iht = tiles.IHT(2)
    assert not iht.fullp()
    iht.dictionary = {1: 1, 2: 2}
    assert iht.fullp()


def test_iht_getindex():
    iht = tiles.IHT(10)

    # Test getting new index
    idx1 = iht.getindex((1, 2, 3))
    assert idx1 == 0
    assert (1, 2, 3) in iht.dictionary

    # Test getting existing index
    idx2 = iht.getindex((1, 2, 3))
    assert idx2 == idx1

    # Test readonly for non-existent key
    idx3 = iht.getindex((4, 5, 6), readonly=True)
    assert idx3 is None


def test_hashcoords():
    iht = tiles.IHT(10)
    coords = [1, 2, 3]

    # Test with IHT
    assert tiles.hashcoords(coords, iht) == 0

    # Test with int
    result = tiles.hashcoords(coords, 5)
    assert 0 <= result < 5

    # Test with None
    assert tiles.hashcoords(coords, None) == coords


def test_tiles_basic():
    result = tiles.tiles(10, 2, [0.1, 0.2], [1])
    assert len(result) == 2
    assert all(isinstance(tile, int) for tile in result)


def test_tileswrap_basic():
    result = tiles.tileswrap(10, 2, [0.1, 0.2], [5, 5], [1])
    assert len(result) == 2
    assert all(isinstance(tile, int) for tile in result)
    assert all(0 <= tile < 10 for tile in result)


def test_pow2geq():
    assert tiles.pow2geq(3) == 4
    assert tiles.pow2geq(4) == 4
    assert tiles.pow2geq(5) == 8
    assert tiles.pow2geq(7) == 8
    assert tiles.pow2geq(8) == 8
    assert tiles.pow2geq(9) == 16


def test_tiles_class():
    dims_min = np.array([-1.0, -2.0])
    dims_max = np.array([1.0, 2.0])
    tiling = tiles.Tiles(dims_min, dims_max, tiling_dim=8)

    # Test initialization
    assert tiling.tiling_dim == 8
    assert tiling.num_tilings >= 8  # Should be at least 4 * num_dims
    assert np.array_equal(tiling.dims_min, dims_min)
    assert np.array_equal(tiling.dims_max, dims_max)

    # Test call
    x = np.array([0.0, 0.0])
    result = tiling(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == (tiling.max_size,)
    assert np.sum(result) == tiling.num_tilings  # One active tile per tiling


def test_tiles_class_input_validation():
    with pytest.raises(AssertionError):
        tiles.Tiles([1, 2], [3, 4], 8)  # Should be numpy arrays
