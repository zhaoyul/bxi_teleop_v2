import numpy as np

from elf3_arm_ik_baselink.reachability import (
    ReachabilitySample,
    cross_midline_targets,
    select_boundary_demo_samples,
)


class Result:
    def __init__(self, success):
        self.success = success


def test_cross_midline_targets_are_on_robot_left():
    targets = cross_midline_targets()
    assert len(targets) == 18
    assert all(np.isclose(target[0], 0.38) for target in targets)
    assert all(target[1] > 0.0 for target in targets)
    assert sorted({round(float(target[2]), 2) for target in targets}) == [
        -0.08,
        -0.02,
        0.04,
    ]


def test_selects_near_and_stable_interior_points_per_height():
    samples = [
        ReachabilitySample(np.array([0.34, 0.02, 0.02]), Result(True)),
        ReachabilitySample(np.array([0.34, 0.06, 0.02]), Result(True)),
        ReachabilitySample(np.array([0.34, 0.10, 0.02]), Result(True)),
        ReachabilitySample(np.array([0.34, 0.14, 0.02]), Result(True)),
        ReachabilitySample(np.array([0.34, 0.18, 0.02]), Result(False)),
        ReachabilitySample(np.array([0.34, 0.06, 0.08]), Result(True)),
        ReachabilitySample(np.array([0.34, 0.10, 0.08]), Result(True)),
    ]

    selected = select_boundary_demo_samples(samples)
    positions = [sample.position.tolist() for sample in selected]
    assert positions == [
        [0.34, 0.06, 0.02],
        [0.34, 0.10, 0.02],
        [0.34, 0.06, 0.08],
        [0.34, 0.10, 0.08],
    ]
