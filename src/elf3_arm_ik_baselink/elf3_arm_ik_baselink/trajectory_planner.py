"""ROS-free constrained joint trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlannedTrajectory:
    times_sec: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    duration_sec: float


def plan_smoothstep_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    max_velocity_rad_s: float,
    max_acceleration_rad_s2: float,
    control_period_sec: float,
    minimum_duration_sec: float = 0.0,
) -> PlannedTrajectory:
    """Generate a cubic trajectory with zero endpoint velocity."""
    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if start.shape != goal.shape or len(start) == 0:
        raise ValueError('start and goal must have the same non-empty shape')
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError('start and goal must contain finite values')
    if max_velocity_rad_s <= 0.0:
        raise ValueError('max_velocity_rad_s must be positive')
    if max_acceleration_rad_s2 <= 0.0:
        raise ValueError('max_acceleration_rad_s2 must be positive')
    if control_period_sec <= 0.0:
        raise ValueError('control_period_sec must be positive')
    if minimum_duration_sec < 0.0:
        raise ValueError('minimum_duration_sec cannot be negative')

    delta = goal - start
    max_delta = float(np.max(np.abs(delta)))
    velocity_duration = 1.5 * max_delta / float(max_velocity_rad_s)
    acceleration_duration = math.sqrt(
        6.0 * max_delta / float(max_acceleration_rad_s2)
    )
    duration = max(
        float(control_period_sec),
        float(minimum_duration_sec),
        velocity_duration,
        acceleration_duration,
    )
    step_count = max(1, int(math.ceil(duration / control_period_sec)))
    times = np.linspace(0.0, duration, step_count + 1)
    normalized = times / duration

    blend = 3.0 * normalized**2 - 2.0 * normalized**3
    blend_velocity = (6.0 * normalized - 6.0 * normalized**2) / duration
    blend_acceleration = (6.0 - 12.0 * normalized) / (duration**2)

    return PlannedTrajectory(
        times_sec=times,
        positions=start + np.outer(blend, delta),
        velocities=np.outer(blend_velocity, delta),
        accelerations=np.outer(blend_acceleration, delta),
        duration_sec=duration,
    )
