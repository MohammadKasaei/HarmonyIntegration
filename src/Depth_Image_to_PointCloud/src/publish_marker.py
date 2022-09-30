import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2
from image_geometry import PinholeCameraModel
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import message_filters
import tf
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker



class ObjectLocalization:
    def __init__(self) -> None:
        rospy.init_node("object_localization")
        
        self.img_sub = rospy.Subscriber('/kinect_subordinate/rgb/image_raw', Image, self.img_callback)
        self.pc_sub = rospy.Subscriber('/points', PointCloud2, self.pc_callback)

        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)

        self.bridge = CvBridge()

        self.listener = tf.TransformListener()
        self.listener.waitForTransform("myumi_005_base_link", "kinect_subordinate_rgb_camera_link", rospy.Time(0), rospy.Duration(0.1))

        self.pc = None

    def mousePoints(self, event, x, y, flags, params):
        # Left button mouse click event opencv
        if event == cv2.EVENT_LBUTTONDOWN:
            # The image is 180 degree rotated

            print(self.pc.shape)


            point_3d_camera_frame = self.pc[y,x]

            grasp_point = PointStamped()
            grasp_point.header.frame_id = "kinect_subordinate_rgb_camera_link"
            grasp_point.header.stamp = rospy.Time(0)
            grasp_point.point.x = point_3d_camera_frame[0]
            grasp_point.point.y = point_3d_camera_frame[1]
            grasp_point.point.z = point_3d_camera_frame[2]

            base_point = self.listener.transformPoint("/myumi_005_base_link", grasp_point)

            print("============================================")
            print("Image Coords: ({}, {})".format(x, y))
            print("---> Camera frame coords: ({}, {}, {})".format(
                point_3d_camera_frame[0], point_3d_camera_frame[1], point_3d_camera_frame[2]
            ))
            print("---> Base frame coords: ({}, {}, {}) ".format(
                base_point.point.x, base_point.point.y, base_point.point.z
            ))

            marker = Marker()

            marker = Marker()
            marker.header.frame_id = "myumi_005_base_link"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = base_point.point.x
            marker.pose.position.y = base_point.point.y
            marker.pose.position.z = base_point.point.z

            self.marker_pub.publish(marker)

    def img_callback(self, data):
        np_img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        cv2.imshow("camera", np_img)
        cv2.setMouseCallback("camera", self.mousePoints)
        cv2.waitKey(1)

    
    def pc_callback(self, data):
        self.pc = np.asarray(list(point_cloud2.read_points(data))).reshape(720, 1280, 3)
        



if __name__ == '__main__':
    ObjectLocalization()
    rospy.spin()