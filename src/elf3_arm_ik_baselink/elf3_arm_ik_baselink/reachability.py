"""Right-arm cross-midline reachability sampling for the customer demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import (
    RIGHT_HOME as RIGHT_HOME_VALUES,
    RIGHT_IK_REFERENCE_SEEDS,
)
from .ik_solver import Elf3ArmIkSolver, IkResult


HANDSHAKE_ORIENTATION = np.array(
    [
        [0.78980, -0.52256, -0.32116],
        [0.43515, 0.84639, -0.30702],
        [0.43226, 0.10273, 0.89588],
    ],
    dtype=float,
)
RIGHT_HOME = np.asarray(RIGHT_HOME_VALUES, dtype=float)
REFERENCE_SEEDS = tuple(
    np.asarray(seed, dtype=float) for seed in RIGHT_IK_REFERENCE_SEEDS
)
TCP_OFFSET = np.array([0.08, 0.0, 0.0])


@dataclass(frozen=True)
class ReachabilitySample:
    position: np.ndarray
    result: IkResult

    @property
    def reachable(self) -> bool:
        return self.result.success


def cross_midline_targets(
    x: float = 0.34,
    y_values: Iterable[float] = (0.02, 0.06, 0.10, 0.14, 0.18, 0.22),
    z_values: Iterable[float] = (0.02, 0.08, 0.14),
) -> list[np.ndarray]:
    """Return body-left sample points; positive Y is the robot's left side."""
    return [
        np.array([float(x), float(y), float(z)], dtype=float)
        for z in z_values
        for y in y_values
    ]


def scan_right_arm_reachability(
    solver: Elf3ArmIkSolver,
    targets: Iterable[np.ndarray],
) -> list[ReachabilitySample]:
    """Evaluate fixed-orientation 6D IK with deterministic seed selection."""
    samples = []
    previous = RIGHT_HOME.copy()
    for target in targets:
        result = solver.solve_with_seeds(
            side='right',
            tcp_position=np.asarray(target, dtype=float),
            tcp_orientation=HANDSHAKE_ORIENTATION,
            primary_seed=previous,
            fallback_seeds=(RIGHT_HOME, *REFERENCE_SEEDS),
            tcp_offset=TCP_OFFSET,
        )
        if result.success:
            previous = result.joints.copy()
        samples.append(
            ReachabilitySample(
                position=np.asarray(target, dtype=float).copy(),
                result=result,
            )
        )
    return samples


def select_boundary_demo_samples(
    samples: Iterable[ReachabilitySample],
) -> list[ReachabilitySample]:
    """Select near-midline and stable 10 cm cross-midline points per height."""
    by_height: dict[float, list[ReachabilitySample]] = {}
    for sample in samples:
        if sample.reachable:
            by_height.setdefault(round(float(sample.position[2]), 6), []).append(sample)

    selected = []
    for height in sorted(by_height):
        row = sorted(by_height[height], key=lambda item: float(item.position[1]))
        near = min(row, key=lambda item: abs(float(item.position[1]) - 0.06))
        interior = min(row, key=lambda item: abs(float(item.position[1]) - 0.10))
        selected.append(near)
        if not np.allclose(near.position, interior.position):
            selected.append(interior)
    return selected
