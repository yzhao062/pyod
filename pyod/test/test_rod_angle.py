import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyod.models.rod import angle


@pytest.mark.parametrize('sign, expected', [(1, 0.), (-1, np.pi)])
def test_angle_roundoff_at_parallel_boundaries(sign, expected):
    vector = np.array(
        [0.9034701816518086, 0.09401229776087457, -0.7434992493538084]
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert_allclose(angle(vector, sign * vector), expected, atol=1e-7)


@pytest.mark.parametrize(
    'other, expected',
    [([1., 0., 0.], 0.), ([0., 1., 0.], np.pi / 2), ([-1., 0., 0.], np.pi)],
)
def test_angle_ordinary_vectors(other, expected):
    assert_allclose(
        angle(np.array([1., 0., 0.]), np.array(other)), expected, atol=1e-7
    )


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('magnitude_name', ['max', 'tiny'])
@pytest.mark.parametrize('sign, expected', [(1, 0.), (-1, np.pi)])
def test_angle_parallel_vectors_at_extreme_magnitudes(
        dtype, magnitude_name, sign, expected):
    magnitude = getattr(np.finfo(dtype), magnitude_name)
    vector = np.array([magnitude, 0., 0.], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert_allclose(angle(vector, sign * vector), expected, atol=1e-7)


def test_angle_with_zero_vector_is_nan():
    assert np.isnan(angle(np.zeros(3), np.ones(3)))


@pytest.mark.parametrize('sign, expected', [(1, 0.), (-1, np.pi)])
def test_angle_with_signed_integer_minimum(sign, expected):
    vector = np.array([np.iinfo(np.int8).min, 0, 0], dtype=np.int8)
    assert_allclose(angle(vector, sign * vector.astype(float)), expected)
