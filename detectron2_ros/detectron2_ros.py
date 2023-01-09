#!/usr/bin/env python
import cv2
import numpy as np
import rclpy
import sys
import threading
import time

from rclpy.node import Node
from rclpy.parameter import Parameter

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from cv_bridge import CvBridge, CvBridgeError
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2_ros_msgs.msg import Result
from sensor_msgs.msg import Image, RegionOfInterest


class Detectron2node(Node):
    def __init__(self):
        super().__init__("detectron2_ros")
        
        setup_logger()

        self._bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()
        self._image_counter = 0

        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.get_parameter_or('config', "config.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.get_parameter_or('detection_threshold', 0.9) # set threshold for this model
        self.cfg.MODEL.WEIGHTS = self.get_parameter_or('model', "model.ckpt")
        self.predictor = DefaultPredictor(self.cfg)
        self._class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

        self._visualization = self.get_parameter_or('visualization', True)
        self._result_pub = self.create_publisher(Result, 'result', 1)
        self._vis_pub = self.create_publisher(Image, 'visualization', 1)
        self.input_topic = self.get_parameter_or("input", "/camera/color/image_raw")
        self._sub = self.create_subscription(Image, self.input_topic, self.callback_image, 1)
        self.start_time = time.time()

        run()

    def run(self):
        rate = 100
        while not rclpy.is_shutdown():
            if self._msg_lock.acquire(False):
                img_msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                time.sleep(rate)
                continue

            if img_msg is not None:
                self._image_counter = self._image_counter + 1
                #if (self._image_counter % 11) == 10:
                #    rospy.loginfo("Images detected per second=%.2f",
                #                  float(self._image_counter) / (time.time() - self.start_time))

                np_image = self.convert_to_cv_image(img_msg)

                outputs = self.predictor(np_image)
                result = outputs["instances"].to("cpu")
                result_msg = self.getResult(result)

                self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    img = v.get_image()[:, :, ::-1]

                    image_msg = self._bridge.cv2_to_imgmsg(img)
                    self._vis_pub.publish(image_msg)

            time.sleep(rate)

    def getResult(self, predictions):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
        else:
            return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(self._class_names)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self._bridge.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

        return cv_img

    def callback_image(self, msg):
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._header = msg.header
            self._msg_lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = Detectron2node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        sys.exit(1)