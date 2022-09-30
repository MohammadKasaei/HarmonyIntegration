import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped
from grasp_detection_core.srv import DetectGrasp, DetectGraspRequest
from grasp_detection_msgs.msg import Grasp
from cv_bridge import CvBridge, CvBridgeError



rospy.init_node("test_ssg")
rospy.wait_for_service("detect_grasp_ssg")

detector = rospy.ServiceProxy('detect_grasp_ssg', DetectGrasp)


req = DetectGraspRequest()

# rgb = rospy.wait_for_message("/kinect_master/rgb/image_rot", Image)
# depth = rospy.wait_for_message("/kinect_master/depth_to_rgb/image_rot", Image)

rgb = rospy.wait_for_message("/kinect_subordinate/rgb/image_raw", Image)
depth = rospy.wait_for_message("/kinect_subordinate/depth_to_rgb/image_raw", Image)

req.RGBImage = rgb
req.DepthImage = depth

resutls = detector(req)

print(resutls)