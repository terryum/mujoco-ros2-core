"""Timed joint-motion data and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class JointStateSnapshot:
    """A simulator or robot joint-state sample in canonical SI units."""

    joint_names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    stamp_seconds: float | None = None


@dataclass(frozen=True)
class MotionTrajectory:
    """A joint trajectory with rows ordered by time and columns by joint name."""

    robot_id: str
    model_id: str
    joint_names: tuple[str, ...]
    time_from_start: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable validation result suitable for promotion gates."""

    is_valid: bool
    errors: tuple[str, ...]
    model_id: str
    sample_count: int
    duration_seconds: float


def validate_trajectory(
    trajectory: MotionTrajectory,
    *,
    known_joints: set[str] | tuple[str, ...] | list[str],
    position_limits: Mapping[str, tuple[float, float]],
    max_velocity: Mapping[str, float] | float | None = None,
) -> ValidationReport:
    """Validate shape, time, joint identity, limits, and optional speed bounds."""

    errors: list[str] = []
    names = tuple(trajectory.joint_names)
    times = np.asarray(trajectory.time_from_start, dtype=np.float64)
    positions = np.asarray(trajectory.positions, dtype=np.float64)
    velocities = (
        None
        if trajectory.velocities is None
        else np.asarray(trajectory.velocities, dtype=np.float64)
    )

    if not trajectory.robot_id:
        errors.append("robot_id must not be empty")
    if not trajectory.model_id:
        errors.append("model_id must not be empty")
    if not names:
        errors.append("joint_names must not be empty")
    if len(names) != len(set(names)):
        errors.append("joint_names must be unique")

    unknown = sorted(set(names) - set(known_joints))
    if unknown:
        errors.append(f"unknown joints: {', '.join(unknown)}")

    if times.ndim != 1 or times.size == 0:
        errors.append("time_from_start must be a non-empty 1-D array")
    else:
        if not np.all(np.isfinite(times)):
            errors.append("time_from_start must contain finite values")
        if times[0] < 0.0:
            errors.append("time_from_start must begin at or after zero")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            errors.append("time_from_start must be strictly increasing")

    expected_shape = (times.size, len(names))
    if positions.shape != expected_shape:
        errors.append(
            f"positions shape must be {expected_shape}, got {positions.shape}"
        )
    elif not np.all(np.isfinite(positions)):
        errors.append("positions must contain finite values")

    if velocities is not None:
        if velocities.shape != expected_shape:
            errors.append(
                f"velocities shape must be {expected_shape}, got {velocities.shape}"
            )
        elif not np.all(np.isfinite(velocities)):
            errors.append("velocities must contain finite values")

    if positions.shape == expected_shape and np.all(np.isfinite(positions)):
        for column, name in enumerate(names):
            if name not in position_limits:
                errors.append(f"missing position limit for joint: {name}")
                continue
            lower, upper = position_limits[name]
            values = positions[:, column]
            if np.any(values < lower) or np.any(values > upper):
                errors.append(
                    f"position limit exceeded for {name}: [{lower}, {upper}]"
                )

        if max_velocity is not None and times.size > 1:
            measured = np.abs(np.diff(positions, axis=0) / np.diff(times)[:, None])
            for column, name in enumerate(names):
                limit = (
                    float(max_velocity[name])
                    if isinstance(max_velocity, Mapping)
                    else float(max_velocity)
                )
                if np.any(measured[:, column] > limit):
                    errors.append(f"velocity limit exceeded for {name}: {limit}")

    duration = float(times[-1]) if times.ndim == 1 and times.size else 0.0
    return ValidationReport(
        is_valid=not errors,
        errors=tuple(errors),
        model_id=trajectory.model_id,
        sample_count=int(times.size),
        duration_seconds=duration,
    )

