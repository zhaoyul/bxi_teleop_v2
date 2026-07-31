from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('ikpy')

from elf3_arm_ik_baselink.ik_solver import (  # noqa: E402
    Elf3ArmIkSolver,
    STATUS_INVALID_INPUT,
    STATUS_POSITION_ERROR,
)
from elf3_arm_ik_baselink.constants import (  # noqa: E402
    RIGHT_HOME,
    RIGHT_IK_REFERENCE_SEEDS,
)
from elf3_arm_ik_baselink.reachability import (  # noqa: E402
    HANDSHAKE_ORIENTATION,
    TCP_OFFSET,
)


@pytest.fixture(scope='module')
def solver():
    urdf_dir = Path(__file__).resolve().parents[3] / 'data'
    return Elf3ArmIkSolver(str(urdf_dir))


def test_reachable_pose_is_validated(solver):
    joints = np.array([0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0])
    tcp_offset = np.array([0.08, 0.0, 0.0])
    transform = solver.left_chain.forward_kinematics(np.r_[0.0, joints])
    tcp_position = (
        solver.shoulder_origins['left']
        + transform[:3, 3]
        + transform[:3, :3] @ tcp_offset
    )

    result = solver.solve(
        'left',
        tcp_position,
        transform[:3, :3],
        joints,
        tcp_offset,
    )

    assert result.success
    assert result.position_error_m <= 0.002
    assert result.orientation_error_rad <= np.deg2rad(2.0)
    assert result.solve_time_ms >= 0.0
    assert result.validation_time_ms >= 0.0


def test_unreachable_pose_is_rejected(solver):
    result = solver.solve(
        'left',
        np.array([2.0, 2.0, 2.0]),
        np.eye(3),
        np.zeros(7),
        np.array([0.08, 0.0, 0.0]),
    )

    assert not result.success
    assert result.status == STATUS_POSITION_ERROR


def test_nonfinite_input_is_rejected(solver):
    result = solver.solve(
        'left',
        np.array([np.nan, 0.0, 0.0]),
        np.eye(3),
        np.zeros(7),
        np.array([0.08, 0.0, 0.0]),
    )

    assert not result.success
    assert result.status == STATUS_INVALID_INPUT


def test_fallback_seed_recovers_cross_midline_boundary_pose(solver):
    near = solver.solve(
        'right',
        np.array([0.34, 0.06, 0.02]),
        HANDSHAKE_ORIENTATION,
        np.asarray(RIGHT_HOME),
        TCP_OFFSET,
    )
    assert near.success

    result = solver.solve_with_seeds(
        side='right',
        tcp_position=np.array([0.34, 0.18, 0.02]),
        tcp_orientation=HANDSHAKE_ORIENTATION,
        primary_seed=near.joints,
        fallback_seeds=RIGHT_IK_REFERENCE_SEEDS,
        tcp_offset=TCP_OFFSET,
    )

    assert result.success
    assert result.min_joint_limit_margin_rad >= 0.005
