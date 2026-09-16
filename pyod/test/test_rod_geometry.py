import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyod.models.rod import geometric_median


@pytest.mark.parametrize('values', [[0, 0, 0, 10, 20], [0, 10, 20, 20, 20]])
@pytest.mark.parametrize('offset', [np.zeros(3), np.array([100., -50., 25.])])
def test_geometric_median_preserves_duplicate_weights(values, offset):
    points = np.array([[value, 0., 0.] for value in values]) + offset
    before = points.copy()
    expected = np.array([np.median(values), 0., 0.]) + offset
    result = geometric_median(points)
    assert_allclose(result, expected, atol=1e-4)
    assert_allclose(geometric_median(points[::-1]), result, atol=1e-4)
    assert_array_equal(points, before)


@pytest.mark.parametrize('count', [1, 5])
def test_geometric_median_identical_points(count):
    point = np.array([1., 2., 3.])
    assert_allclose(geometric_median(np.tile(point, (count, 1))), point)


def test_geometric_median_unique_symmetric_points():
    points = np.array([[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., 1., 0.]])
    assert_allclose(geometric_median(points), np.zeros(3), atol=1e-4)
