import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('detectron2_ros'),
        'config',
        'detectron2_ros.yaml'
        )
        
    node=Node(
        package = 'detectron2_ros',
        name = 'detectron2_ros_node',
        executable = 'detectron2_ros',
        parameters = [config]
    )
    ld.add_action(node)
    return ld

