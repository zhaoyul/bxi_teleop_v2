import numpy as np
import pytest

from elf3_arm_ik_baselink.trajectory_planner import plan_smoothstep_trajectory


def test_trajectory_endpoints_and_limits():
    trajectory = plan_smoothstep_trajectory(
        start=np.array([0.0, 0.5]),
        goal=np.array([1.0, -0.5]),
        max_velocity_rad_s=0.5,
        max_acceleration_rad_s2=1.0,
        control_period_sec=0.01,
    )

    np.testing.assert_allclose(trajectory.positions[0], [0.0, 0.5])
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, -0.5])
    np.testing.assert_allclose(trajectory.velocities[[0, -1]], 0.0, atol=1e-12)
    assert np.max(np.abs(trajectory.velocities)) <= 0.5 + 1e-12
    assert np.max(np.abs(trajectory.accelerations)) <= 1.0 + 1e-12
    assert np.all(np.diff(trajectory.times_sec) > 0.0)


def test_trajectory_respects_minimum_duration():
    trajectory = plan_smoothstep_trajectory(
        start=np.zeros(2),
        goal=np.array([0.01, -0.01]),
        max_velocity_rad_s=1.0,
        max_acceleration_rad_s2=2.0,
        control_period_sec=0.01,
        minimum_duration_sec=2.0,
    )
    assert trajectory.duration_sec == 2.0


@pytest.mark.parametrize(
    'velocity,acceleration,period',
    [(0.0, 1.0, 0.01), (1.0, 0.0, 0.01), (1.0, 1.0, 0.0)],
)
def test_trajectory_rejects_invalid_limits(velocity, acceleration, period):
    with pytest.raises(ValueError):
        plan_smoothstep_trajectory(
            np.zeros(1),
            np.ones(1),
            velocity,
            acceleration,
            period,
        )
