import numpy as np

from mujoco_ros2_core.motion import MotionTrajectory, validate_trajectory


def test_valid_trajectory() -> None:
    trajectory = MotionTrajectory(
        robot_id="fixture",
        model_id="fixture-v1",
        joint_names=("joint_a",),
        time_from_start=np.asarray([0.0, 0.5, 1.0]),
        positions=np.asarray([[0.0], [0.2], [0.3]]),
    )
    report = validate_trajectory(
        trajectory,
        known_joints={"joint_a"},
        position_limits={"joint_a": (-1.0, 1.0)},
        max_velocity=1.0,
    )
    assert report.is_valid
    assert report.errors == ()
    assert report.sample_count == 3


def test_invalid_trajectory_reports_all_relevant_failures() -> None:
    trajectory = MotionTrajectory(
        robot_id="fixture",
        model_id="fixture-v1",
        joint_names=("joint_a",),
        time_from_start=np.asarray([0.0, 0.5, 0.4]),
        positions=np.asarray([[0.0], [2.0], [0.0]]),
    )
    report = validate_trajectory(
        trajectory,
        known_joints={"joint_a"},
        position_limits={"joint_a": (-1.0, 1.0)},
        max_velocity=1.0,
    )
    assert not report.is_valid
    assert any("strictly increasing" in error for error in report.errors)
    assert any("position limit exceeded" in error for error in report.errors)

