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
