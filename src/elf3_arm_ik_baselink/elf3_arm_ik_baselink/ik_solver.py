"""IKPy-backed ELF3 arm solver.

This module is intentionally ROS-free so it can be unit-tested on macOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np

from .constants import SIDE_LEFT, SIDE_RIGHT
from .math_utils import rotation_error_rad, rotation_matrix_to_vector


STATUS_SUCCESS = 'success'
STATUS_INVALID_INPUT = 'invalid_input'
STATUS_SOLVER_ERROR = 'solver_error'
STATUS_NONFINITE_SOLUTION = 'nonfinite_solution'
STATUS_POSITION_ERROR = 'position_error'
STATUS_ORIENTATION_ERROR = 'orientation_error'
STATUS_JOINT_LIMIT = 'joint_limit_margin'
STATUS_NEAR_SINGULAR = 'near_singular'


@dataclass(frozen=True)
class IkResult:
    joints: np.ndarray
    wrist_target: np.ndarray
    success: bool
    status: str
    position_error_m: float
    orientation_error_rad: float
    solve_time_ms: float
    validation_time_ms: float
    jacobian_condition: float
    min_joint_limit_margin_rad: float
    message: str = ''


class Elf3ArmIkSolver:
    """Solve one or both ELF3 7-DoF arms from TCP pose targets."""

    def __init__(
        self,
        urdf_dir: str,
        max_position_error_m: float = 0.002,
        max_orientation_error_rad: float = np.deg2rad(2.0),
        max_jacobian_condition: float = 1000.0,
        min_joint_limit_margin_rad: float = 0.005,
        jacobian_step_rad: float = 1e-5,
        left_shoulder_origin: Optional[np.ndarray] = None,
        right_shoulder_origin: Optional[np.ndarray] = None,
    ) -> None:
        try:
            from ikpy.chain import Chain
        except ImportError as exc:
            raise ImportError(
                'ikpy is required for ELF3 arm IK. Install with '
                '`python3 -m pip install ikpy` in the ROS2 environment.'
            ) from exc

        urdf_path = Path(urdf_dir).expanduser()
        self.left_chain = Chain.from_urdf_file(
            str(urdf_path / 'elf3_arm_l.urdf'),
            active_links_mask=[False] + [True] * 7,
        )
        self.right_chain = Chain.from_urdf_file(
            str(urdf_path / 'elf3_arm_r.urdf'),
            active_links_mask=[False] + [True] * 7,
        )
        self.max_position_error_m = float(max_position_error_m)
        self.max_orientation_error_rad = float(max_orientation_error_rad)
        self.max_jacobian_condition = float(max_jacobian_condition)
        self.min_joint_limit_margin_rad = float(min_joint_limit_margin_rad)
        self.jacobian_step_rad = float(jacobian_step_rad)
        self.shoulder_origins = {
            SIDE_LEFT: self._origin_or_default(
                left_shoulder_origin,
                [0.0, 0.178, 0.087],
            ),
            SIDE_RIGHT: self._origin_or_default(
                right_shoulder_origin,
                [0.0, -0.178, 0.087],
            ),
        }

    def solve(
        self,
        side: str,
        tcp_position: np.ndarray,
        tcp_orientation: Optional[np.ndarray],
        seed_joints: Optional[np.ndarray],
        tcp_offset: np.ndarray,
    ) -> IkResult:
        """Return a 7-joint command for one arm.

        tcp_offset is expressed in the TCP/end-effector local frame. The URDF
        chain target is moved back from TCP to wrist origin before IK.
        """
        chain = self._chain(side)
        tcp_position = np.asarray(tcp_position, dtype=float).reshape(3)
        orientation = (
            np.asarray(tcp_orientation, dtype=float).reshape(3, 3)
            if tcp_orientation is not None
            else np.eye(3, dtype=float)
        )
        tcp_offset = np.asarray(tcp_offset, dtype=float).reshape(3)
        shoulder_origin = self.shoulder_origins[side]
        wrist_target = tcp_position - shoulder_origin - orientation @ tcp_offset

        initial = self._initial_position(chain, seed_joints)
        if not all(
            np.all(np.isfinite(value))
            for value in (tcp_position, orientation, tcp_offset, initial)
        ):
            return self._failed_result(
                initial,
                wrist_target,
                STATUS_INVALID_INPUT,
                'IK input contains non-finite values',
            )

        solve_started = time.perf_counter()
        try:
            kwargs = {
                'target_position': wrist_target,
                'initial_position': initial,
            }
            if tcp_orientation is not None:
                kwargs['target_orientation'] = orientation
                kwargs['orientation_mode'] = 'all'

            solution = chain.inverse_kinematics(**kwargs)
        except (ValueError, np.linalg.LinAlgError) as exc:
            solve_time_ms = (time.perf_counter() - solve_started) * 1000.0
            return self._failed_result(
                initial,
                wrist_target,
                STATUS_SOLVER_ERROR,
                str(exc),
                solve_time_ms=solve_time_ms,
            )
        solve_time_ms = (time.perf_counter() - solve_started) * 1000.0

        joints = np.asarray(solution[1:8], dtype=float)
        if not np.all(np.isfinite(joints)):
            return self._failed_result(
                initial,
                wrist_target,
                STATUS_NONFINITE_SOLUTION,
                'IK returned non-finite joint values',
                solve_time_ms=solve_time_ms,
            )

        validation_started = time.perf_counter()
        full_solution = np.asarray(solution, dtype=float)
        transform = np.asarray(chain.forward_kinematics(full_solution), dtype=float)
        actual_orientation = transform[:3, :3]
        actual_tcp_position = (
            shoulder_origin + transform[:3, 3] + actual_orientation @ tcp_offset
        )
        position_error_m = float(np.linalg.norm(actual_tcp_position - tcp_position))
        orientation_error = (
            rotation_error_rad(orientation, actual_orientation)
            if tcp_orientation is not None
            else 0.0
        )
        min_margin = self._minimum_joint_limit_margin(chain, full_solution)
        jacobian_condition = self._jacobian_condition(chain, full_solution)
        validation_time_ms = (time.perf_counter() - validation_started) * 1000.0

        status = STATUS_SUCCESS
        message = ''
        if position_error_m > self.max_position_error_m:
            status = STATUS_POSITION_ERROR
            message = (
                f'position error {position_error_m:.6f} m exceeds '
                f'{self.max_position_error_m:.6f} m'
            )
        elif (
            tcp_orientation is not None
            and orientation_error > self.max_orientation_error_rad
        ):
            status = STATUS_ORIENTATION_ERROR
            message = (
                f'orientation error {np.rad2deg(orientation_error):.3f} deg exceeds '
                f'{np.rad2deg(self.max_orientation_error_rad):.3f} deg'
            )
        elif min_margin < self.min_joint_limit_margin_rad:
            status = STATUS_JOINT_LIMIT
            message = (
                f'joint limit margin {min_margin:.6f} rad is below '
                f'{self.min_joint_limit_margin_rad:.6f} rad'
            )
        elif jacobian_condition > self.max_jacobian_condition:
            status = STATUS_NEAR_SINGULAR
            message = (
                f'Jacobian condition {jacobian_condition:.1f} exceeds '
                f'{self.max_jacobian_condition:.1f}'
            )

        return IkResult(
            joints=joints,
            wrist_target=wrist_target,
            success=status == STATUS_SUCCESS,
            status=status,
            position_error_m=position_error_m,
            orientation_error_rad=orientation_error,
            solve_time_ms=solve_time_ms,
            validation_time_ms=validation_time_ms,
            jacobian_condition=jacobian_condition,
            min_joint_limit_margin_rad=min_margin,
            message=message,
        )

    @staticmethod
    def _failed_result(
        initial: np.ndarray,
        wrist_target: np.ndarray,
        status: str,
        message: str,
        solve_time_ms: float = 0.0,
    ) -> IkResult:
        return IkResult(
            joints=np.asarray(initial[1:8], dtype=float).copy(),
            wrist_target=np.asarray(wrist_target, dtype=float).copy(),
            success=False,
            status=status,
            position_error_m=float('inf'),
            orientation_error_rad=float('inf'),
            solve_time_ms=float(solve_time_ms),
            validation_time_ms=0.0,
            jacobian_condition=float('inf'),
            min_joint_limit_margin_rad=0.0,
            message=message,
        )

    @staticmethod
    def _minimum_joint_limit_margin(chain, full_solution: np.ndarray) -> float:
        margins = []
        for link, value, active in zip(
            chain.links,
            full_solution,
            chain.active_links_mask,
        ):
            if not active:
                continue
            lower, upper = link.bounds
            margins.append(min(float(value) - float(lower), float(upper) - float(value)))
        return min(margins) if margins else float('inf')

    def _jacobian_condition(self, chain, full_solution: np.ndarray) -> float:
        step = self.jacobian_step_rad
        jacobian = np.zeros((6, 7), dtype=float)
        for joint_index in range(7):
            plus = full_solution.copy()
            minus = full_solution.copy()
            plus[joint_index + 1] += step
            minus[joint_index + 1] -= step
            plus_transform = np.asarray(chain.forward_kinematics(plus), dtype=float)
            minus_transform = np.asarray(chain.forward_kinematics(minus), dtype=float)
            jacobian[:3, joint_index] = (
                plus_transform[:3, 3] - minus_transform[:3, 3]
            ) / (2.0 * step)
            relative_rotation = (
                plus_transform[:3, :3] @ minus_transform[:3, :3].T
            )
            jacobian[3:, joint_index] = rotation_matrix_to_vector(
                relative_rotation
            ) / (2.0 * step)

        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        smallest = float(singular_values[-1])
        if smallest <= 1e-12:
            return float('inf')
        return float(singular_values[0] / smallest)

    def _chain(self, side: str):
        if side == SIDE_LEFT:
            return self.left_chain
        if side == SIDE_RIGHT:
            return self.right_chain
        raise ValueError(f'unsupported arm side: {side}')

    @staticmethod
    def _origin_or_default(
        value: Optional[np.ndarray],
        default: list[float],
    ) -> np.ndarray:
        selected = default if value is None else value
        return np.asarray(selected, dtype=float).reshape(3)

    @staticmethod
    def _initial_position(chain, seed_joints: Optional[np.ndarray]) -> np.ndarray:
        if seed_joints is None:
            initial = np.zeros(len(chain.links), dtype=float)
        else:
            initial = np.zeros(len(chain.links), dtype=float)
            initial[1:8] = np.asarray(seed_joints, dtype=float).reshape(7)

        lower = np.asarray([link.bounds[0] for link in chain.links], dtype=float)
        upper = np.asarray([link.bounds[1] for link in chain.links], dtype=float)
        return np.minimum(np.maximum(initial, lower), upper)
