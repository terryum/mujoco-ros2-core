"""Configurable MuJoCo backend for scalar robot joints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from mujoco_ros2_core.motion import JointStateSnapshot


class RobotBackend(Protocol):
    """Minimal state/target boundary shared by simulation and adapters."""

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def read_state(self) -> JointStateSnapshot: ...

    def set_joint_targets(self, targets: Mapping[str, float]) -> None: ...

    def step(self) -> None: ...

    def reset(self, initial_positions: Mapping[str, float] | None = None) -> None: ...


@dataclass(frozen=True)
class ControllerConfig:
    position_gain: float = 25.0
    velocity_gain: float = 4.0
    hold_position_gain: float = 20.0
    hold_velocity_gain: float = 4.0
    max_control_force: float = 20.0
    max_hold_force: float = 10.0


class MujocoJointBackend:
    """Apply bounded generalized-force PD control to selected scalar joints."""

    def __init__(
        self,
        model_path: str | Path,
        controlled_joints: tuple[str, ...] | list[str],
        *,
        frame_skip: int = 10,
        controller: ControllerConfig | None = None,
        disable_gravity: bool = False,
        disable_contacts: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {self.model_path}")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if not controlled_joints:
            raise ValueError("controlled_joints must not be empty")
        if len(controlled_joints) != len(set(controlled_joints)):
            raise ValueError("controlled_joints must be unique")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.controller = controller or ControllerConfig()
        self.controlled_joints = tuple(controlled_joints)

        if disable_gravity:
            self.model.opt.gravity[:] = 0.0
        if disable_contacts:
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0

        scalar_types = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        all_joint_ids = np.arange(self.model.njnt, dtype=np.int32)
        scalar_mask = np.isin(self.model.jnt_type[all_joint_ids], scalar_types)
        self._state_joint_ids = all_joint_ids[scalar_mask]
        self._joint_names = tuple(
            str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)))
            for joint_id in self._state_joint_ids
        )
        self._state_qpos = self.model.jnt_qposadr[self._state_joint_ids].astype(np.int32)
        self._state_dof = self.model.jnt_dofadr[self._state_joint_ids].astype(np.int32)

        ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.controlled_joints
        ]
        if any(joint_id < 0 for joint_id in ids):
            missing = [name for name, joint_id in zip(self.controlled_joints, ids) if joint_id < 0]
            raise ValueError(f"model is missing controlled joints: {', '.join(missing)}")
        if not np.all(np.isin(self.model.jnt_type[np.asarray(ids)], scalar_types)):
            raise ValueError("controlled joints must be hinge or slide joints")

        self._control_joint_ids = np.asarray(ids, dtype=np.int32)
        self._control_qpos = self.model.jnt_qposadr[self._control_joint_ids].astype(np.int32)
        self._control_dof = self.model.jnt_dofadr[self._control_joint_ids].astype(np.int32)
        limited = self.model.jnt_limited[self._control_joint_ids].astype(bool)
        if not np.all(limited):
            raise ValueError("controlled joints must define finite ranges")
        ranges = self.model.jnt_range[self._control_joint_ids]
        self.lower = ranges[:, 0].copy()
        self.upper = ranges[:, 1].copy()
        self.center = ranges.mean(axis=1)

        scalar_limited = self.model.jnt_limited[self._state_joint_ids].astype(bool)
        self._hold_joint_ids = self._state_joint_ids[scalar_limited]
        self._hold_qpos = self.model.jnt_qposadr[self._hold_joint_ids].astype(np.int32)
        self._hold_dof = self.model.jnt_dofadr[self._hold_joint_ids].astype(np.int32)
        self._hold_positions = self.model.jnt_range[self._hold_joint_ids].mean(axis=1)

        self.model.dof_armature[:] = np.maximum(self.model.dof_armature, 0.05)
        self.model.dof_damping[:] = np.maximum(self.model.dof_damping, 2.0)
        self.target = self.center.copy()
        self.reset()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    @property
    def position_limits(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(lower), float(upper))
            for name, lower, upper in zip(self.controlled_joints, self.lower, self.upper)
        }

    def read_state(self) -> JointStateSnapshot:
        return JointStateSnapshot(
            joint_names=self.joint_names,
            positions=self.data.qpos[self._state_qpos].copy(),
            velocities=self.data.qvel[self._state_dof].copy(),
        )

    def read_control_state(self) -> JointStateSnapshot:
        """Return only controlled joints in configured action order."""

        return JointStateSnapshot(
            joint_names=self.controlled_joints,
            positions=self.data.qpos[self._control_qpos].copy(),
            velocities=self.data.qvel[self._control_dof].copy(),
        )

    def set_joint_targets(self, targets: Mapping[str, float]) -> None:
        unknown = sorted(set(targets) - set(self.controlled_joints))
        if unknown:
            raise ValueError(f"unsupported joint targets: {', '.join(unknown)}")
        for name, value in targets.items():
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"target for {name} must be finite")
            index = self.controlled_joints.index(name)
            self.target[index] = np.clip(numeric, self.lower[index], self.upper[index])

    def step(self) -> None:
        config = self.controller
        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:] = 0.0
            hold_error = self._hold_positions - self.data.qpos[self._hold_qpos]
            hold_force = (
                config.hold_position_gain * hold_error
                - config.hold_velocity_gain * self.data.qvel[self._hold_dof]
            )
            self.data.qfrc_applied[self._hold_dof] = np.clip(
                hold_force, -config.max_hold_force, config.max_hold_force
            )

            error = self.target - self.data.qpos[self._control_qpos]
            force = (
                config.position_gain * error
                - config.velocity_gain * self.data.qvel[self._control_dof]
            )
            self.data.qfrc_applied[self._control_dof] = np.clip(
                force, -config.max_control_force, config.max_control_force
            )
            mujoco.mj_step(self.model, self.data)

    def reset(self, initial_positions: Mapping[str, float] | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._hold_qpos] = self._hold_positions
        self.data.qpos[self._control_qpos] = self.center
        self.target[:] = self.center
        if initial_positions:
            unknown = sorted(set(initial_positions) - set(self.controlled_joints))
            if unknown:
                raise ValueError(f"unsupported initial joints: {', '.join(unknown)}")
            for name, value in initial_positions.items():
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"initial position for {name} must be finite")
                index = self.controlled_joints.index(name)
                clipped = np.clip(numeric, self.lower[index], self.upper[index])
                self.data.qpos[self._control_qpos[index]] = clipped
                self.target[index] = clipped
        mujoco.mj_forward(self.model, self.data)
