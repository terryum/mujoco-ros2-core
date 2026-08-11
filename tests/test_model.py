from pathlib import Path

from mujoco_ros2_core.model import prepare_urdf


def test_prepare_urdf_resolves_package_uri_and_is_deterministic(tmp_path: Path) -> None:
    package = tmp_path / "robot_description"
    package.mkdir()
    (package / "mesh.stl").write_bytes(b"mesh-placeholder")
    source = tmp_path / "robot.urdf"
    source.write_text(
        """<robot name="fixture">
  <link name="base">
    <visual><geometry><mesh filename="package://robot_description/mesh.stl"/></geometry></visual>
  </link>
</robot>
"""
    )
    first = prepare_urdf(
        source,
        tmp_path / "first.urdf",
        package_map={"robot_description": package},
    )
    second = prepare_urdf(
        source,
        tmp_path / "second.urdf",
        package_map={"robot_description": package},
    )

    assert first.replacements == 1
    assert first.sha256 == second.sha256
    text = first.output.read_text()
    assert str(package / "mesh.stl") in text
    assert 'discardvisual="false"' in text
    assert 'fusestatic="false"' in text

