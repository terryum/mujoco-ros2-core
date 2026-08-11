"""Optional ROS 2 bridge around :class:`MujocoJointBackend`."""

from __future__ import annotations


def main(args=None) -> None:
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger
    except ImportError as error:
        raise SystemExit(
            "ROS 2 Python packages are unavailable. Run this command inside a "
            "ROS 2 environment supplied by the consuming robot workspace."
        ) from error

    from mujoco_ros2_core.backend import MujocoJointBackend

    class MujocoBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("mujoco_robot_bridge")
            self.declare_parameter("model_path", "")
            self.declare_parameter("controlled_joints", [])
            self.declare_parameter("frame_skip", 10)
            self.declare_parameter("command_topic", "/mujoco_joint_command")
            self.declare_parameter("state_topic", "/joint_states")
            self.declare_parameter("disable_gravity", False)
            self.declare_parameter("disable_contacts", False)

            model_path = str(self.get_parameter("model_path").value)
            joints = tuple(self.get_parameter("controlled_joints").value)
            if not model_path:
                raise ValueError("model_path ROS parameter must not be empty")
            if not joints:
                raise ValueError("controlled_joints ROS parameter must not be empty")

            self.backend = MujocoJointBackend(
                model_path,
                joints,
                frame_skip=int(self.get_parameter("frame_skip").value),
                disable_gravity=bool(self.get_parameter("disable_gravity").value),
                disable_contacts=bool(self.get_parameter("disable_contacts").value),
            )
            command_topic = str(self.get_parameter("command_topic").value)
            state_topic = str(self.get_parameter("state_topic").value)
            self.publisher = self.create_publisher(JointState, state_topic, 10)
            self.subscription = self.create_subscription(
                JointState, command_topic, self.receive_command, 10
            )
            self.reset_service = self.create_service(
                Trigger, "/mujoco_reset", self.reset
            )
            self.timer = self.create_timer(self.backend.control_dt, self.advance)
            self.publish_state()

        def receive_command(self, message) -> None:
            if len(message.name) != len(message.position):
                self.get_logger().warning(
                    "Ignoring command with mismatched name and position lengths"
                )
                return
            try:
                self.backend.set_joint_targets(dict(zip(message.name, message.position)))
            except ValueError as error:
                self.get_logger().warning(f"Ignoring invalid joint command: {error}")

        def advance(self) -> None:
            self.backend.step()
            self.publish_state()

        def publish_state(self) -> None:
            state = self.backend.read_state()
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(state.joint_names)
            message.position = state.positions.tolist()
            message.velocity = state.velocities.tolist()
            self.publisher.publish(message)

        def reset(self, request, response):
            del request
            self.backend.reset()
            self.publish_state()
            response.success = True
            response.message = "MuJoCo state and targets reset"
            return response

    rclpy.init(args=args)
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

