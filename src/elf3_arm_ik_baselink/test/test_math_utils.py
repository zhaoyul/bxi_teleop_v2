import math

import numpy as np

from elf3_arm_ik_baselink.math_utils import (
    exponential_smooth,
    quaternion_xyzw_to_matrix,
    rate_limit,
    rotation_error_rad,
    rotation_matrix_to_vector,
)


def test_quaternion_identity_matrix():
    matrix = quaternion_xyzw_to_matrix([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-9)


def test_quaternion_z_rotation():
    half = math.sqrt(0.5)
    matrix = quaternion_xyzw_to_matrix([0.0, 0.0, half, half])
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-9)


def test_rate_limit():
    actual = rate_limit(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -1.0, 0.02]),
        0.1,
    )
    np.testing.assert_allclose(actual, [0.1, -0.1, 0.02])


def test_exponential_smooth():
    actual = exponential_smooth(
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
        0.25,
    )
    np.testing.assert_allclose(actual, [0.25, 1.5])


def test_rotation_error_rad():
    half = math.sqrt(0.5)
    rotation = quaternion_xyzw_to_matrix([0.0, 0.0, half, half])
    assert math.isclose(rotation_error_rad(np.eye(3), rotation), math.pi / 2.0)


def test_rotation_matrix_to_vector():
    half = math.sqrt(0.5)
    rotation = quaternion_xyzw_to_matrix([0.0, 0.0, half, half])
    np.testing.assert_allclose(
        rotation_matrix_to_vector(rotation),
        [0.0, 0.0, math.pi / 2.0],
        atol=1e-9,
    )
