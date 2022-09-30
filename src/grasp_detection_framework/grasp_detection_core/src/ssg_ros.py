import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped
from grasp_detection_core.srv import DetectGrasp, DetectGraspResponse
from grasp_detection_msgs.msg import Grasp
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
import tf

import torch

import numpy as np
import cv2
import pybullet as p
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F

from ssg import SingleStageGraspSynthesis
from ssg_config import get_config
from ssg_utils import gr_nms_v2, gr_post_processing


parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='../trained-models/ssg-OCID.pth')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')


class SSGROS:
    def __init__(self) -> None:
        # Initialize Network
        args = parser.parse_args()
        args.cfg = "res101_ssg"
        self.cfg = get_config(args, mode='val')

        self.net = SingleStageGraspSynthesis(self.cfg)
        self.net.eval()
        state_dict = torch.load("../trained-models/ssg-OCID.pth", map_location='cpu')
        correct_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        with torch.no_grad():
            import torch.nn as nn
            self.net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.net.load_weights(correct_state_dict, self.cfg.cuda)
        self.net = self.net.cuda()

        self.input_size = 544
        self.norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        self.norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)


        # Initialize ROS
        rospy.init_node("ssg_ros")
        srv = rospy.Service("detect_grasp_ssg", DetectGrasp, self.callback)

        self.listener = tf.TransformListener()
        self.listener.waitForTransform("myumi_005_base_link", "kinect_subordinate_rgb_camera_link", rospy.Time(0), rospy.Duration(0.1))

        
        camera_info = rospy.wait_for_message("/kinect_subordinate/rgb/camera_info", CameraInfo, timeout=5)
        self.img_height = camera_info.height
        self.img_width = camera_info.width
        self.bridge = CvBridge()
        # print(camera_info)

        # @TODO Change this crop area for different experiments setup
        self.crop_area = [320, 120, 960, 600]
        self.visualize = True

        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        
        rospy.loginfo("ready to detect grasps")
    
    
    def visualize_results(self, img, depth, bboxes, masks, grasps, labels):
        from ssg_config import colors_list, cls_list

        masks_semantic = masks * (labels[:, None, None]+1)
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 31

        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')
        img_fused = (color_masks * 0.6 + img_u8 * 0.8)

        fig = plt.figure(figsize=(10, 10))

        for i in range(bboxes.shape[0]):
            name = cls_list[int(bboxes[i, -1])]
            color = colors_list[int(bboxes[i, -1])]
            cv2.rectangle(img_fused, (int(bboxes[i, 0]), int(bboxes[i, 1])),
                        (int(bboxes[i, 2]), int(bboxes[i, 3])), color.tolist(), 1)
            cv2.putText(img_fused, "{}:{}".format(name, int(bboxes[i, -1])), (int(bboxes[i, 0]), int(bboxes[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


        if grasps is not None:
            for rect in grasps:
                cls_id = rect[-1]
                name = cls_list[int(cls_id)]
                color = colors_list[int(cls_id)].tolist()
                center_x, center_y, width, height, theta, cls_id = rect
                box = ((center_x, center_y), (width, height), -(theta))
                box = cv2.boxPoints(box)
                box = np.int0(box)
                # cv2.drawContours(img_fused, [box], 0, color, 2)

                inv_color = (255, 255-color[1], 255-color[2])

                p1, p2, p3, p4 = box
                length = width
                p5 = (p1+p2)/2
                p6 = (p3+p4)/2
                p7 = (p5+p6)/2

                rad = theta / 180 * np.pi
                p8 = (p7[0]-length*np.sin(rad), p7[1]+length*np.cos(rad))
                cv2.circle(img_fused, (int(p7[0]), int(p7[1])), 2, (0,0,255), 2)
                cv2.line(img_fused, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 3, 8)
                cv2.line(img_fused, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 3, 8)
                cv2.line(img_fused, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (255,0,0), 2, 8)


        ax = fig.add_subplot(1, 3, 1)
        ax.imshow((img_u8/255.)[...,::-1])
        ax.set_title('RGB')
        ax.axis('off')

        if depth is not None:
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(depth, cmap='gray')
            ax.set_title('Depth')
            ax.axis('off')
        
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(img_fused/255.)
        ax.set_title('Results')
        ax.axis('off')

        cv2.imwrite("results.png", img_fused)

        # plt.savefig("overall_cogr_result.png")
    
    
    def normalize_and_to_rgb(self, img):
        img = (img - self.norm_mean) / self.norm_std
        img = img[:, :, (2, 1, 0)]
        return img

    
    def pad_to_square(self, rgb, depth):
        h, w, c = rgb.shape
        pad_size = max(h, w)
        pad_img = np.zeros((pad_size, pad_size, c), dtype='float32')
        pad_img[:, :, :] = self.norm_mean
        pad_depth = np.zeros((pad_size, pad_size), dtype='float32')

        pad_img[0: h, 0: w, :] = rgb
        pad_depth[0: h, 0: w] = depth

        pad_img = cv2.resize(pad_img, (self.input_size, self.input_size))
        pad_depth = cv2.resize(pad_depth, (self.input_size, self.input_size))

        return pad_img, pad_depth

    
    def pixel_to_ee(self, x, y):
        pc_msg = rospy.wait_for_message('/points', PointCloud2, timeout=5)
        pc_np = np.asarray(list(point_cloud2.read_points(pc_msg))).reshape(self.img_height, self.img_width, 3)
        
        point_3d_camera_frame = pc_np[y,x]

        grasp_point_cam = PointStamped()
        grasp_point_cam.header.frame_id = "kinect_subordinate_rgb_camera_link"
        grasp_point_cam.header.stamp = rospy.Time(0)
        grasp_point_cam.point.x = point_3d_camera_frame[0]
        grasp_point_cam.point.y = point_3d_camera_frame[1]
        grasp_point_cam.point.z = point_3d_camera_frame[2]

        grasp_point_base = self.listener.transformPoint("/myumi_005_base_link", grasp_point_cam)

        return grasp_point_base
    
    def callback(self, req):
        rospy.loginfo("Received request from client")

        depth_msg = req.DepthImage
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        depth_image = depth_image.astype(np.uint8)
        depth_image = 1 - (depth_image / np.max(depth_image))
        depth_image = depth_image[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2]]

        rgb_msg = req.RGBImage
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        rgb_image = rgb_image[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2],:]
        h, w, c = rgb_image.shape

        # Pad to square
        rgb, depth = self.pad_to_square(rgb_image, depth_image)
        rgb = self.normalize_and_to_rgb(rgb)

        if len(depth.shape) < 3:
            depth = np.expand_dims(depth, -1)
        
        input_tensor = torch.from_numpy(np.concatenate([rgb, depth], axis=-1)).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()

        # ==========================================
        # Process input with network
        # ==========================================
        with torch.no_grad():
            class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = self.net(input_tensor)

            ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                    class_pred, box_pred, coef_pred, proto_out,
                    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
                    self.net.anchors, self.cfg
                )

            img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing(
                rgb, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, 
                ori_h=int(self.crop_area[3]-self.crop_area[1]),
                ori_w=int(self.crop_area[2]-self.crop_area[0]),
                num_grasp_per_object=1
            )

            all_grasps = []
            for obj_rects in grasps:
                all_grasps.extend(obj_rects)
        
            self.visualize_results(img, depth, box_p, instance_masks, all_grasps, ids_p)


        # ==========================================
        # Get 3D grasp poses
        # ==========================================
        res = DetectGraspResponse()
        for i in range(len(all_grasps)):
            grasp_msg = Grasp()
            
            center_x, center_y, width, height, theta, cls_id = all_grasps[i]
            print(center_x, center_y)
            rad = theta / 180 * np.pi

            # print(g)

            grasp_msg.conf = 1.0
            grasp_msg.obj_cls = cls_id
            grasp_msg.x = int(center_x)
            grasp_msg.y = int(center_y)
            grasp_msg.w = int(width)
            grasp_msg.h = int(height)
            grasp_msg.rot = rad
            grasp_msg.ee_pose.pose.position = self.pixel_to_ee(int(center_x), int(center_y)).point

            # Calculate quaterion
            # @TODO You may need to change this part
            q = p.getQuaternionFromEuler([-np.pi, 0, rad-np.pi/2]) 
            grasp_msg.ee_pose.pose.orientation.x = q[0]
            grasp_msg.ee_pose.pose.orientation.y = q[1]
            grasp_msg.ee_pose.pose.orientation.z = q[2]
            grasp_msg.ee_pose.pose.orientation.w = q[3]

            res.results.append(grasp_msg)


            # ======================================================
            # visualize grasps position in rviz as a set of markers
            # Comment this part if you don't need it
            # ======================================================
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
            marker.pose.position.x = grasp_msg.ee_pose.pose.position.x
            marker.pose.position.y = grasp_msg.ee_pose.pose.position.y
            marker.pose.position.z = grasp_msg.ee_pose.pose.position.z

            self.marker_pub.publish(marker)
            # ========================================================
            # ========================================================


        rospy.loginfo("Finised detecting grasps")

        return res


if __name__ == "__main__":
    demo = SSGROS()

    rospy.spin()