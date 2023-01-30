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
        #parameters = [config]
        parameters = [
            {'input': "/camera/color/image_raw"},
            {'detection_threshold': 0.75},
            {'detectron2_config': "/home/user1/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"},
            {'model': "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"},
            {'visualization': True}
	    ]
    )
    ld.add_action(node)
    return ld

