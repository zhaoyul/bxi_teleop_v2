"""Small math helpers that do not depend on ROS."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def quaternion_xyzw_to_matrix(values: Iterable[float]) -> np.ndarray:
    """Convert an XYZW quaternion to a 3x3 rotation matrix."""
    x, y, z, w = [float(value) for value in values]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=float)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def clamp_vector(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clamp a vector component-wise."""
    return np.minimum(np.maximum(values, lower), upper)


def rate_limit(previous: np.ndarray, target: np.ndarray, max_abs_step: float) -> np.ndarray:
    """Limit per-channel joint motion in one controller tick."""
    delta = np.asarray(target, dtype=float) - np.asarray(previous, dtype=float)
    delta = np.clip(delta, -float(max_abs_step), float(max_abs_step))
    return np.asarray(previous, dtype=float) + delta


def exponential_smooth(previous: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """First-order low-pass smoothing for joint commands."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * np.asarray(previous, dtype=float) + alpha * np.asarray(
        target,
        dtype=float,
    )


def rotation_error_rad(target: np.ndarray, actual: np.ndarray) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    delta = np.asarray(target, dtype=float).reshape(3, 3).T @ np.asarray(
        actual,
        dtype=float,
    ).reshape(3, 3)
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def rotation_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a rotation vector with stable small angles."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    angle = rotation_error_rad(np.eye(3, dtype=float), rotation)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    if angle <= 1e-8:
        return 0.5 * skew

    sine = math.sin(angle)
    if abs(sine) <= 1e-8:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        norm = float(np.linalg.norm(axis))
        return angle * axis / norm if norm > 1e-12 else np.zeros(3)
    return angle * skew / (2.0 * sine)


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized XYZW quaternion."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        values = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        next_axis = (axis + 1) % 3
        last_axis = (axis + 2) % 3
        scale = 2.0 * math.sqrt(
            1.0
            + rotation[axis, axis]
            - rotation[next_axis, next_axis]
            - rotation[last_axis, last_axis]
        )
        values = np.zeros(4)
        values[axis] = 0.25 * scale
        values[3] = (
            rotation[last_axis, next_axis] - rotation[next_axis, last_axis]
        ) / scale
        values[next_axis] = (
            rotation[next_axis, axis] + rotation[axis, next_axis]
        ) / scale
        values[last_axis] = (
            rotation[last_axis, axis] + rotation[axis, last_axis]
        ) / scale
    return values / np.linalg.norm(values)
