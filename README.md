# mujoco-ros2-core

Reusable, robot-neutral building blocks for MuJoCo motion experiments and ROS 2
visualization. The package keeps robot assets, hardware drivers, policies, and
datasets in their owning repositories.

## Capabilities

- Prepare MuJoCo-compatible URDF files without modifying vendor sources.
- Run a configurable joint-position backend for scalar MuJoCo joints.
- Drive MJCF-authored native position actuators without replacing their gains;
  commands are clamped to the finite actuator/joint range intersection, with
  the joint range used when the actuator omits `ctrlrange`.
- Represent and validate timed joint trajectories.
- Publish simulated state and accept joint targets through a generic ROS 2 node.

## Development

```bash
uv sync --group dev
uv run pytest
```

The unit tests use synthetic XML/URDF fixtures and do not download robot assets.
ROS 2 is supplied by the consuming workspace; it is intentionally not installed
from PyPI by this package.

## Use from a robot lab

Pin this repository as a submodule and install it into the lab environment as an
editable path dependency. Keep robot-specific joint maps, model preparation
configuration, safety limits, and hardware adapters in the robot repository.

## License

Apache-2.0. Third-party robot assets remain under their original licenses and
are not redistributed here.
