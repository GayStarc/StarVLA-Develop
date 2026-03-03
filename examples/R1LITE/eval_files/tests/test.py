import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class EEPoseMonitor(Node):
    def __init__(self):
        super().__init__('ee_pose_monitor')

        # 缓存左右臂 ee pose
        self.left_ee_pose = None
        self.right_ee_pose = None

        # 左臂
        self.sub_left = self.create_subscription(
            PoseStamped,
            '/motion_control/pose_ee_arm_left',
            self.left_callback,
            10
        )

        # 右臂
        self.sub_right = self.create_subscription(
            PoseStamped,
            '/motion_control/pose_ee_arm_right',
            self.right_callback,
            10
        )

        # 可选：定时器，用来“像变量一样读”
        self.timer = self.create_timer(
            1.0 / 30.0,  # 30Hz
            self.timer_callback
        )

    def left_callback(self, msg):
        self.left_ee_pose = msg

    def right_callback(self, msg):
        self.right_ee_pose = msg

    def timer_callback(self):
        if self.left_ee_pose is None or self.right_ee_pose is None:
            return

        lpos = self.left_ee_pose.pose.position
        rpos = self.right_ee_pose.pose.position

        self.get_logger().info(
            f"L: ({lpos.x:.3f}, {lpos.y:.3f}, {lpos.z:.3f}) | "
            f"R: ({rpos.x:.3f}, {rpos.y:.3f}, {rpos.z:.3f})"
        )


def main():
    rclpy.init()
    node = EEPoseMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()