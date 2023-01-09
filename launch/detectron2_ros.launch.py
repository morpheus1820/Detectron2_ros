<launch>
  <arg name="input" default="ViMantic/ToCNN" />
  <arg name="detection_threshold" default="0.5" />
  <arg name="config" default="$(find detectron2_ros)/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" />
  <arg name="model" default="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" />
  <arg name="visualization" default="true" />

  <node name="detectron2_ros"  pkg="detectron2_ros" type="detectron2_ros" output="screen" >
    <param name="input" value="$(arg input)" />
    <param name="detection_threshold" value="$(arg detection_threshold)" />
    <param name="config" value="$(arg config)" />
    <param name="model" value="$(arg model)" />
    <param name="visualization" value="$(arg visualization)" />
  </node>
</launch>


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
Footer
