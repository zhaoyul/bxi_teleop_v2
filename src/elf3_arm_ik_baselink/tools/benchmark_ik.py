#!/usr/bin/env python3
"""Run reproducible ELF3 FK-generated IK accuracy and latency checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from ament_index_python.packages import get_package_share_directory
import numpy as np

from elf3_arm_ik_baselink.ik_solver import Elf3ArmIkSolver


TCP_OFFSET = np.array([0.08, 0.0, 0.0], dtype=float)
CASES = {
    'left_home': (
        'left',
        np.array([0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0]),
    ),
    'left_demo': (
        'left',
        np.array([0.4, 2.45, 1.0, 1.45, -1.2, 0.8, -0.6]),
    ),
    'right_handshake': (
        'right',
        np.array([0.2, -0.65, -0.35, 0.95, 0.0, 0.2, 0.0]),
    ),
    'right_demo': (
        'right',
        np.array([0.45, -1.8, -0.65, 1.2, 0.8, 0.6, 0.4]),
    ),
}
HOME = {
    'left': np.array([0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0]),
    'right': np.array([0.5, -0.3, 0.1, -0.2, 0.0, 0.0, 0.0]),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error('--iterations must be positive')

    share = Path(get_package_share_directory('elf3_arm_ik_baselink'))
    solver = Elf3ArmIkSolver(str(share / 'data'))
    tracking_solve_times = []
    cold_solve_times = []
    failures = []

    for name, (side, target_joints) in CASES.items():
        chain = solver._chain(side)
        target_transform = chain.forward_kinematics(np.r_[0.0, target_joints])
        target_tcp = (
            solver.shoulder_origins[side]
            + target_transform[:3, 3]
            + target_transform[:3, :3] @ TCP_OFFSET
        )
        cold_result = solver.solve(
            side,
            target_tcp,
            target_transform[:3, :3],
            np.zeros(7),
            TCP_OFFSET,
        )
        cold_solve_times.append(cold_result.solve_time_ms)

        seed = HOME[side].copy()
        case_results = []
        for progress in np.linspace(1.0 / args.iterations, 1.0, args.iterations):
            sample_joints = HOME[side] + (target_joints - HOME[side]) * progress
            transform = chain.forward_kinematics(np.r_[0.0, sample_joints])
            tcp_position = (
                solver.shoulder_origins[side]
                + transform[:3, 3]
                + transform[:3, :3] @ TCP_OFFSET
            )
            result = solver.solve(
                side,
                tcp_position,
                transform[:3, :3],
                seed,
                TCP_OFFSET,
            )
            case_results.append(result)
            if result.success:
                seed = result.joints

        tracking_solve_times.extend(result.solve_time_ms for result in case_results)
        result = case_results[-1]
        if not all(item.success for item in case_results):
            failures.append(f'{name}:{result.status}')
        print(
            f'{name}: status={result.status} '
            f'position={result.position_error_m * 1000.0:.3f}mm '
            f'orientation={np.rad2deg(result.orientation_error_rad):.3f}deg '
            f'cold={cold_result.solve_time_ms:.3f}ms '
            f'track_p50={statistics.median(item.solve_time_ms for item in case_results):.3f}ms '
            f'condition={result.jacobian_condition:.1f} '
            f'limit_margin={result.min_joint_limit_margin_rad:.3f}rad'
        )

    unreachable = solver.solve(
        'left',
        np.array([2.0, 2.0, 2.0]),
        np.eye(3),
        np.zeros(7),
        TCP_OFFSET,
    )
    print(
        f'unreachable: status={unreachable.status} '
        f'position={unreachable.position_error_m:.3f}m '
        f'solve={unreachable.solve_time_ms:.3f}ms'
    )
    if unreachable.success:
        failures.append('unreachable:unexpected_success')

    sorted_times = sorted(tracking_solve_times)
    p95_index = min(len(sorted_times) - 1, int(0.95 * len(sorted_times)))
    print(
        f'tracking_summary: samples={len(sorted_times)} '
        f'p50={statistics.median(sorted_times):.3f}ms '
        f'p95={sorted_times[p95_index]:.3f}ms '
        f'max={max(sorted_times):.3f}ms '
        f'under_20ms={sum(value <= 20.0 for value in sorted_times) / len(sorted_times):.1%}'
    )
    print(
        f'cold_summary: samples={len(cold_solve_times)} '
        f'p50={statistics.median(cold_solve_times):.3f}ms '
        f'max={max(cold_solve_times):.3f}ms'
    )
    if sorted_times[p95_index] > 20.0:
        failures.append(f'tracking_p95:{sorted_times[p95_index]:.3f}ms')
    if failures:
        print('FAILED: ' + ', '.join(failures))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
