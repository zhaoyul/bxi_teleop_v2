"""IKPy-backed ELF3 arm solver.

This module is intentionally ROS-free so it can be unit-tested on macOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import SIDE_LEFT, SIDE_RIGHT


@dataclass(frozen=True)
class IkResult:
    joints: np.ndarray
    wrist_target: np.ndarray
    success: bool
    message: str = ''


class Elf3ArmIkSolver:
    """Solve one or both ELF3 7-DoF arms from TCP pose targets."""

    def __init__(self, urdf_dir: str) -> None:
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
        orientation = (
            np.asarray(tcp_orientation, dtype=float).reshape(3, 3)
            if tcp_orientation is not None
            else np.eye(3, dtype=float)
        )
        wrist_target = (
            np.asarray(tcp_position, dtype=float).reshape(3)
            - orientation @ np.asarray(tcp_offset, dtype=float).reshape(3)
        )

        initial = self._initial_position(chain, seed_joints)

        try:
            kwargs = {
                'target_position': wrist_target,
                'initial_position': initial,
            }
            if tcp_orientation is not None:
                kwargs['target_orientation'] = orientation
                kwargs['orientation_mode'] = 'all'

            solution = chain.inverse_kinematics(**kwargs)
        except ValueError as exc:
            fallback = initial[1:8].copy()
            return IkResult(
                joints=fallback,
                wrist_target=wrist_target,
                success=False,
                message=str(exc),
            )

        joints = np.asarray(solution[1:8], dtype=float)
        return IkResult(
            joints=joints,
            wrist_target=wrist_target,
            success=True,
        )

    def _chain(self, side: str):
        if side == SIDE_LEFT:
            return self.left_chain
        if side == SIDE_RIGHT:
            return self.right_chain
        raise ValueError(f'unsupported arm side: {side}')

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
