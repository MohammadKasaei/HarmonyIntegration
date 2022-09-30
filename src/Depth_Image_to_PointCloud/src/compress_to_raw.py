#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from image_geometry import PinholeCameraModel
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import message_filters


class ImageTools(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

    def convert_ros_msg_to_cv2(self, ros_data, image_encoding='bgr8'):
        """
        Convert from a ROS Image message to a cv2 image.
        """
        try:
            return self._cv_bridge.imgmsg_to_cv2(ros_data, image_encoding)
        except CvBridgeError as e:
            if "[16UC1] is not a color format" in str(e):
                raise CvBridgeError(
                    "You may be trying to use a Image method " +
                    "(Subscriber, Publisher, conversion) on a depth image" +
                    " message. Original exception: " + str(e))
            raise e

    def convert_ros_compressed_to_cv2(self, compressed_msg):
        np_arr = np.fromstring(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def convert_ros_compressed_msg_to_ros_msg(self, compressed_msg,
                                              encoding='bgr8', rotate=False):
        cv2_img = self.convert_ros_compressed_to_cv2(compressed_msg)
        ros_img = self._cv_bridge.cv2_to_imgmsg(cv2_img, encoding=encoding)
        ros_img.header = compressed_msg.header
        
        if rotate:
            cv2_img_rot = cv2.rotate(cv2_img, cv2.ROTATE_180)
            ros_img_rot = self._cv_bridge.cv2_to_imgmsg(cv2_img_rot, encoding=encoding)
            ros_img_rot.header = compressed_msg.header

            return ros_img, ros_img_rot
        
        else:
            return ros_img

        
        

    def convert_cv2_to_ros_msg(self, cv2_data, image_encoding='bgr8'):
        """
        Convert from a cv2 image to a ROS Image message.
        """
        return self._cv_bridge.cv2_to_imgmsg(cv2_data, image_encoding)

    def convert_cv2_to_ros_compressed_msg(self, cv2_data,
                                          compressed_format='jpg'):
        """
        Convert from cv2 image to ROS CompressedImage.
        """
        return self._cv_bridge.cv2_to_compressed_imgmsg(cv2_data,
                                                        dst_format=compressed_format)

    def convert_ros_msg_to_ros_compressed_msg(self, image,
                                              image_encoding='bgr8',
                                              compressed_format="jpg"):
        """
        Convert from ROS Image message to ROS CompressedImage.
        """
        cv2_img = self.convert_ros_msg_to_cv2(image, image_encoding)
        cimg_msg = self._cv_bridge.cv2_to_compressed_imgmsg(cv2_img,
                                                            dst_format=compressed_format)
        cimg_msg.header = image.header
        return cimg_msg

    def convert_to_cv2(self, image):
        """
        Convert any kind of image to cv2.
        """
        cv2_img = None
        if type(image) == np.ndarray:
            cv2_img = image
        elif image._type == 'sensor_msgs/Image':
            cv2_img = self.convert_ros_msg_to_cv2(image)
        elif image._type == 'sensor_msgs/CompressedImage':
            cv2_img = self.convert_ros_compressed_to_cv2(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return cv2_img

    def convert_to_ros_msg(self, image):
        """
        Convert any kind of image to ROS Image.
        """
        ros_msg = None
        if type(image) == np.ndarray:
            ros_msg = self.convert_cv2_to_ros_msg(image)
        elif image._type == 'sensor_msgs/Image':
            ros_msg = image
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_msg = self.convert_ros_compressed_msg_to_ros_msg(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_msg

    def convert_to_ros_compressed_msg(self, image, compressed_format='jpg'):
        """
        Convert any kind of image to ROS Compressed Image.
        """
        ros_cmp = None
        if type(image) == np.ndarray:
            ros_cmp = self.convert_cv2_to_ros_compressed_msg(
                image, compressed_format=compressed_format)
        elif image._type == 'sensor_msgs/Image':
            ros_cmp = self.convert_ros_msg_to_ros_compressed_msg(
                image, compressed_format=compressed_format)
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_cmp = image
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_cmp

    def convert_depth_to_ros_msg(self, image):
        ros_msg = None
        if type(image) == np.ndarray:
            ros_msg = self.convert_cv2_to_ros_msg(image,
                                                  image_encoding='mono16')
        elif image._type == 'sensor_msgs/Image':
            image.encoding = '16UC1'
            ros_msg = image
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_msg = self.convert_compressedDepth_to_image_msg(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_msg

    def convert_depth_to_ros_compressed_msg(self, image):
        ros_cmp = None
        if type(image) == np.ndarray:
            ros_cmp = self.convert_cv2_to_ros_compressed_msg(image,
                                                             compressed_format='png')
            ros_cmp.format = '16UC1; compressedDepth'
            # This is a header ROS depth CompressedImage have, necessary
            # for viewer tools to see the image
            # extracted from a real image from a robot
            # The code that does it in C++ is this:
            # https://github.com/ros-perception/image_transport_plugins/blob/indigo-devel/compressed_depth_image_transport/src/codec.cpp
            ros_cmp.data = "\x00\x00\x00\x00\x88\x9c\x5c\xaa\x00\x40\x4b\xb7" + ros_cmp.data
        elif image._type == 'sensor_msgs/Image':
            image.encoding = "mono16"
            ros_cmp = self.convert_ros_msg_to_ros_compressed_msg(
                image,
                image_encoding='mono16',
                compressed_format='png')
            ros_cmp.format = '16UC1; compressedDepth'
            ros_cmp.data = "\x00\x00\x00\x00\x88\x9c\x5c\xaa\x00\x40\x4b\xb7" + ros_cmp.data
        elif image._type == 'sensor_msgs/CompressedImage':
            ros_cmp = image
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return ros_cmp

    def convert_depth_to_cv2(self, image):
        cv2_img = None
        if type(image) == np.ndarray:
            cv2_img = image
        elif image._type == 'sensor_msgs/Image':
            image.encoding = 'mono16'
            cv2_img = self.convert_ros_msg_to_cv2(image,
                                                  image_encoding='mono16')
        elif image._type == 'sensor_msgs/CompressedImage':
            cv2_img = self.convert_compressedDepth_to_cv2(image)
        else:
            raise TypeError("Cannot convert type: " + str(type(image)))
        return cv2_img

    def convert_compressedDepth_to_image_msg(self, compressed_image, rotate=False):
        """
        Convert a compressedDepth topic image into a ROS Image message.
        compressed_image must be from a topic /bla/compressedDepth
        as it's encoded in PNG
        Code from: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
        """
        depth_img_raw = self.convert_compressedDepth_to_cv2(compressed_image)
        img_msg = self._cv_bridge.cv2_to_imgmsg(depth_img_raw, "mono16")
        img_msg.header = compressed_image.header
        img_msg.encoding = "16UC1"

        if rotate:
            depth_img_raw_rot = cv2.rotate(depth_img_raw, cv2.ROTATE_180)

            img_msg_rot = self._cv_bridge.cv2_to_imgmsg(depth_img_raw_rot, "mono16")
            img_msg_rot.header = compressed_image.header
            img_msg_rot.encoding = "16UC1"

            return img_msg, img_msg_rot
        
        else:
            return img_msg

    def convert_compressedDepth_to_cv2(self, compressed_depth):
        """
        Convert a compressedDepth topic image into a cv2 image.
        compressed_depth must be from a topic /bla/compressedDepth
        as it's encoded in PNG
        Code from: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
        """
        depth_fmt, compr_type = compressed_depth.format.split(';')
        # remove white space
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()

        
        depth_header_size = 0
        raw_data = compressed_depth.data[depth_header_size:]

        depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8),
                                     # the cv2.CV_LOAD_IMAGE_UNCHANGED has been removed
                                     -1)  # cv2.CV_LOAD_IMAGE_UNCHANGED)
        if depth_img_raw is None:
            # probably wrong header size
            raise Exception("Could not decode compressed depth image."
                            "You may need to change 'depth_header_size'!")
        return depth_img_raw

    def display_image(self, image):
        """
        Use cv2 to show an image.
        """
        cv2_img = self.convert_to_cv2(image)
        window_name = 'show_image press q to exit'
        cv2.imshow(window_name, cv2_img)
        # TODO: Figure out how to check if the window
        # was closed... when a user does it, the program is stuck
        key = cv2.waitKey(0)
        if chr(key) == 'q':
            cv2.destroyWindow(window_name)

    def save_image(self, image, filename):
        """
        Given an image in numpy array or ROS format
        save it using cv2 to the filename. The extension
        declares the type of image (e.g. .jpg .png).
        """
        cv2_img = self.convert_to_cv2(image)
        cv2.imwrite(filename, cv2_img)

    def save_depth_image(self, image, filename):
        """
        Save a normalized (easier to visualize) version
        of a depth image into a file.
        """
        # Duc's smart undocummented code
        im_array = self.convert_depth_to_cv2(image)
        min_distance, max_distance = np.min(im_array), np.max(im_array)
        im_array = im_array * 1.0
        im_array = (im_array < max_distance) * im_array
        im_array = (im_array - min_distance) / max_distance * 255.0
        im_array = (im_array >= 0) * im_array

        cv2.imwrite(filename, im_array)

    def load_from_file(self, file_path, cv2_imread_mode=None):
        """
        Load image from a file.
        :param file_path str: Path to the image file.
        :param cv2_imread_mode int: cv2.IMREAD_ mode, modes are:
            cv2.IMREAD_ANYCOLOR 4             cv2.IMREAD_REDUCED_COLOR_4 33
            cv2.IMREAD_ANYDEPTH 2             cv2.IMREAD_REDUCED_COLOR_8 65
            cv2.IMREAD_COLOR 1                cv2.IMREAD_REDUCED_GRAYSCALE_2 16
            cv2.IMREAD_GRAYSCALE 0            cv2.IMREAD_REDUCED_GRAYSCALE_4 32
            cv2.IMREAD_IGNORE_ORIENTATION 128 cv2.IMREAD_REDUCED_GRAYSCALE_8 64
            cv2.IMREAD_LOAD_GDAL 8            cv2.IMREAD_UNCHANGED -1
            cv2.IMREAD_REDUCED_COLOR_2 17
        """
        if cv2_imread_mode is not None:
            img = cv2.imread(file_path, cv2_imread_mode)
        img = cv2.imread(file_path)
        if img is None:
            raise RuntimeError("No image found to load at " + str(file_path))
        return img

class CompressedDepth2Depth:
    def __init__(self) -> None:
        rospy.init_node("compress_to_raw")
        self.dep_sub = rospy.Subscriber('/kinect_subordinate/depth_to_rgb/image_raw/compressed', CompressedImage, self.depth_callback)
        self.img_sub = rospy.Subscriber('/kinect_subordinate/rgb/image_raw/compressed', CompressedImage, self.rgb_callback)

        
        self.dep_pub = rospy.Publisher('/kinect_subordinate/depth_to_rgb/image_raw', Image, queue_size=1)
        self.rot_dep_pub = rospy.Publisher('/kinect_subordinate/depth_to_rgb/image_rot', Image, queue_size=1)

        self.rgb_pub = rospy.Publisher('/kinect_subordinate/rgb/image_raw', Image, queue_size=1)
        self.rot_rgb_pub = rospy.Publisher('/kinect_subordinate/rgb/image_rot', Image, queue_size=1)
        
        self.img_tools = ImageTools()
    
    def depth_callback(self, depth):
        dep_raw_msg, depth_rot_msg = self.img_tools.convert_compressedDepth_to_image_msg(depth, rotate=True)

        self.dep_pub.publish(dep_raw_msg)
        self.rot_dep_pub.publish(depth_rot_msg)

    def rgb_callback(self, rgb):
        rgb_raw_msg, rgb_rot_msg = self.img_tools.convert_ros_compressed_msg_to_ros_msg(rgb, rotate=True)

        self.rgb_pub.publish(rgb_raw_msg)
        self.rot_rgb_pub.publish(rgb_rot_msg)


if __name__ == '__main__':
    CompressedDepth2Depth()
    rospy.spin()
