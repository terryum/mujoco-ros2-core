from pathlib import Path

import numpy as np
import pytest

from mujoco_ros2_core import MujocoPositionActuatorBackend


MODEL_XML = """
<mujoco model="native_position">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="joint_a" type="hinge" range="-1 1" damping=".1" armature=".01"/>
      <geom type="capsule" fromto="0 0 0 0 0 .2" size=".02" mass=".1"/>
      <body pos="0 0 .2">
        <joint name="joint_b" type="hinge" range="-.5 .5" damping=".1" armature=".01"/>
        <geom type="capsule" fromto="0 0 0 0 0 .2" size=".02" mass=".1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="actuator_a" joint="joint_a" kp="5" kv="1" ctrlrange="-.8 .8"/>
    <position name="actuator_b" joint="joint_b" kp="5" kv="1" ctrlrange="-.5 .5"/>
  </actuator>
</mujoco>
"""


def write_model(tmp_path: Path, xml: str = MODEL_XML) -> Path:
    model_path = tmp_path / "model.xml"
    model_path.write_text(xml)
    return model_path


def test_native_position_motion_and_reset(tmp_path: Path) -> None:
    backend = MujocoPositionActuatorBackend(
        write_model(tmp_path),
        ("joint_a", "joint_b"),
        frame_skip=5,
    )
    assert backend.joint_names == ("joint_a", "joint_b")
    assert backend.control_dt == pytest.approx(0.01)
    assert backend.position_limits["joint_a"] == pytest.approx((-0.8, 0.8))
    np.testing.assert_allclose(backend.read_control_state().positions, (0.0, 0.0))

    backend.set_joint_targets({"joint_a": 2.0, "joint_b": -0.3})
    for _ in range(200):
        backend.step()
    moved = backend.read_control_state().positions
    assert moved[0] > 0.5
    assert moved[1] < -0.15
    np.testing.assert_allclose(backend.target, (0.8, -0.3))

    backend.reset({"joint_a": 0.2})
    np.testing.assert_allclose(backend.read_control_state().positions, (0.2, 0.0))
    np.testing.assert_allclose(backend.target, (0.2, 0.0))


def test_native_position_validation(tmp_path: Path) -> None:
    backend = MujocoPositionActuatorBackend(write_model(tmp_path), ["joint_a"])
    with pytest.raises(ValueError, match="unsupported joint"):
        backend.set_joint_targets({"missing": 0.0})
    with pytest.raises(ValueError, match="must be finite"):
        backend.set_joint_targets({"joint_a": np.nan})

    motor_xml = MODEL_XML.replace(
        '<position name="actuator_a" joint="joint_a" kp="5" kv="1" ctrlrange="-.8 .8"/>',
        '<motor name="actuator_a" joint="joint_a" ctrlrange="-.8 .8"/>',
    )
    with pytest.raises(ValueError, match="exactly one native position actuator"):
        MujocoPositionActuatorBackend(write_model(tmp_path, motor_xml), ["joint_a"])


def test_native_position_uses_joint_range_without_ctrlrange(tmp_path: Path) -> None:
    no_ctrlrange_xml = MODEL_XML.replace(' ctrlrange="-.8 .8"', "")
    backend = MujocoPositionActuatorBackend(
        write_model(tmp_path, no_ctrlrange_xml),
        ["joint_a"],
    )

    assert backend.position_limits["joint_a"] == pytest.approx((-1.0, 1.0))
    backend.set_joint_targets({"joint_a": 2.0})
    np.testing.assert_allclose(backend.target, (1.0,))
    for _ in range(200):
        backend.step()
    assert np.all(np.isfinite(backend.read_state().positions))
