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


class MujocoPositionActuatorBackend:
    """Drive scalar joints through native MuJoCo position actuators.

    The model must expose exactly one ``<position>`` actuator for every
    controlled joint. Commands use the intersection of the finite joint and
    actuator control ranges. If an actuator omits ``ctrlrange``, the finite
    joint range is used as the command boundary. Unlike
    :class:`MujocoJointBackend`, this backend preserves the gains, force limits,
    damping, contacts, and other dynamics authored in the source MJCF.
    """

    def __init__(
        self,
        model_path: str | Path,
        controlled_joints: tuple[str, ...] | list[str],
        *,
        frame_skip: int = 10,
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

        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.controlled_joints
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            missing = [
                name for name, joint_id in zip(self.controlled_joints, joint_ids) if joint_id < 0
            ]
            raise ValueError(f"model is missing controlled joints: {', '.join(missing)}")
        if not np.all(np.isin(self.model.jnt_type[joint_ids], scalar_types)):
            raise ValueError("controlled joints must be hinge or slide joints")

        actuator_ids: list[int] = []
        for name, joint_id in zip(self.controlled_joints, joint_ids):
            candidates = np.flatnonzero(
                (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (self.model.actuator_trnid[:, 0] == joint_id)
            )
            position_candidates = [
                int(actuator_id)
                for actuator_id in candidates
                if self._is_position_actuator(int(actuator_id))
            ]
            if len(position_candidates) != 1:
                raise ValueError(
                    f"joint {name} must have exactly one native position actuator; "
                    f"found {len(position_candidates)}"
                )
            actuator_ids.append(position_candidates[0])

        self._control_joint_ids = joint_ids
        self._control_qpos = self.model.jnt_qposadr[joint_ids].astype(np.int32)
        self._control_dof = self.model.jnt_dofadr[joint_ids].astype(np.int32)
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
        if not np.all(self.model.jnt_limited[joint_ids].astype(bool)):
            raise ValueError("controlled joints must define finite ranges")

        joint_ranges = self.model.jnt_range[joint_ids]
        ctrl_limited = self.model.actuator_ctrllimited[self._actuator_ids].astype(bool)
        ctrl_ranges = joint_ranges.copy()
        ctrl_ranges[ctrl_limited] = self.model.actuator_ctrlrange[
            self._actuator_ids[ctrl_limited]
        ]
        self.lower = np.maximum(joint_ranges[:, 0], ctrl_ranges[:, 0])
        self.upper = np.minimum(joint_ranges[:, 1], ctrl_ranges[:, 1])
        if np.any(self.lower >= self.upper):
            raise ValueError("joint and actuator control ranges do not overlap")
        self.center = 0.5 * (self.lower + self.upper)
        self.home = self.model.qpos0[self._control_qpos].copy()
        if np.any(self.home < self.lower) or np.any(self.home > self.upper):
            raise ValueError("model qpos0 lies outside a controlled joint range")
        self.target = self.home.copy()
        self.reset()

    def _is_position_actuator(self, actuator_id: int) -> bool:
        gain = float(self.model.actuator_gainprm[actuator_id, 0])
        return bool(
            self.model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
            and self.model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
            and gain > 0.0
            and np.isclose(self.model.actuator_biasprm[actuator_id, 1], -gain)
            and self.model.actuator_biasprm[actuator_id, 2] <= 0.0
        )

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
        self.data.ctrl[self._actuator_ids] = self.target

    def step(self) -> None:
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def reset(self, initial_positions: Mapping[str, float] | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)

        # Keep every compatible position actuator at its authored qpos0 even
        # when callers control only a subset of the model.
        for actuator_id in range(self.model.nu):
            if not self._is_position_actuator(actuator_id):
                continue
            if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            self.data.ctrl[actuator_id] = self.model.qpos0[qpos_address]

        self.target[:] = self.home
        self.data.ctrl[self._actuator_ids] = self.target
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
            self.data.ctrl[self._actuator_ids] = self.target
        mujoco.mj_forward(self.model, self.data)
