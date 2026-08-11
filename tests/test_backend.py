from pathlib import Path

import numpy as np
import pytest

from mujoco_ros2_core.backend import MujocoJointBackend


MODEL_XML = """
<mujoco model="two_joint_test">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="joint_a" type="hinge" range="-1 1" damping="1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.05"/>
      <body pos="0 0 0.5">
        <joint name="joint_b" type="hinge" range="-0.5 0.5" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.xml"
    path.write_text(MODEL_XML)
    return path


def test_backend_moves_resets_and_reports_limits(model_path: Path) -> None:
    backend = MujocoJointBackend(model_path, ["joint_a"], frame_skip=10)
    assert backend.joint_names == ("joint_a", "joint_b")
    assert backend.position_limits == {"joint_a": (-1.0, 1.0)}
    initial = backend.read_state().positions.copy()

    backend.set_joint_targets({"joint_a": 0.7})
    for _ in range(80):
        backend.step()
    moved = backend.read_state().positions
    assert moved[0] > initial[0] + 0.1
    assert np.all(np.isfinite(moved))

    backend.reset()
    np.testing.assert_allclose(backend.read_state().positions, initial)

    backend.reset({"joint_a": 0.25})
    control_state = backend.read_control_state()
    assert control_state.joint_names == ("joint_a",)
    np.testing.assert_allclose(control_state.positions, [0.25])


def test_backend_rejects_invalid_targets(model_path: Path) -> None:
    backend = MujocoJointBackend(model_path, ["joint_a"])
    with pytest.raises(ValueError, match="unsupported joint"):
        backend.set_joint_targets({"missing": 0.0})
    with pytest.raises(ValueError, match="must be finite"):
        backend.set_joint_targets({"joint_a": np.nan})
