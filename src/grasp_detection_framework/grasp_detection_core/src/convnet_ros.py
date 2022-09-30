import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped
from grasp_detection_core.srv import DetectGrasp, DetectGraspResponse
from grasp_detection_msgs.msg import Grasp
from cv_bridge import CvBridge, CvBridgeError
import tf

import torch
from grconvnet3 import GenerativeResnet
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import numpy as np
import cv2
import pybullet as p
import matplotlib.pyplot as plt



class ConvNetROS:
    def __init__(self) -> None:
        self.net = GenerativeResnet(
            input_channels=4,
            dropout=True,
            prob=0.1,
            channel_size=32
        )
        self.net.load_state_dict(torch.load("../trained-models/Gr-ConvNet-cornell.pth"))

        self.net.eval().cuda()

        # fake_img = torch.randn((1, 4, 720, 1280))
        # pos_output, cos_output, sin_output, width_output = self.net(fake_img)
        # print(pos_output.shape)
        # print(cos_output.shape)
        # print(sin_output.shape)
        # print(width_output.shape)

        self.bridge = CvBridge()
        # self.norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        # self.norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)

        rospy.init_node("gr_convnet_ros")
        srv = rospy.Service("detect_grasp_convnet", DetectGrasp, self.callback)

        self.listener = tf.TransformListener()
        self.listener.waitForTransform("myumi_005_base_link", "kinect_subordinate_rgb_camera_link", rospy.Time(0), rospy.Duration(0.1))

        
        camera_info = rospy.wait_for_message("/kinect_subordinate/rgb/camera_info", CameraInfo, timeout=5)
        self.img_height = camera_info.height
        self.img_width = camera_info.width
        # print(camera_info)

        self.crop_area = [600, 240, 750, 360]
        self.visualize = True
        
        rospy.loginfo("ready to detect grasps")
    
    
    def normalize_and_to_rgb(self, img):
        img = img.astype(np.float32) / 255.0
        img -= img.mean()
        img = img[:, :, (2, 1, 0)]
        return img
    
    
    def normalize_depth(self, img):
        img = np.clip((img - img.mean()), -1, 1)

        return img
    

    def depth_inpaint(self, depth_img, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(depth_img).max()
        depth_img = depth_img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_img = depth_img[1:-1, 1:-1]
        depth_img = depth_img * scale

        return depth_img

    
    def pad_to_square(self, rgb, depth):
        h, w, c = rgb.shape
        pad_size = max(h, w)
        pad_img = np.zeros((pad_size, pad_size, c), dtype='float32')
        pad_img[:, :, :] = self.norm_mean
        pad_depth = np.zeros((pad_size, pad_size), dtype='float32')

        pad_img[0: h, 0: w, :] = rgb
        pad_depth[0: h, 0: w] = depth

        # pad_img = cv2.resize(pad_img, (self.input_size, self.input_size))
        # pad_depth = cv2.resize(pad_depth, (self.input_size, self.input_size))

        return pad_img, pad_depth
    

    def post_process_output(self, q_img, cos_img, sin_img, width_img, num_grasps):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * 75.0

        q_img = gaussian(q_img, 2.0, preserve_range=True)
        ang_img = gaussian(ang_img, 2.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=num_grasps)

        grasps = []
        for grasp_point_array in local_max:
            grasp_point = tuple(grasp_point_array)

            grasp_angle = ang_img[grasp_point]

            grasp_width = width_img[grasp_point]
            grasp_height = 50
            
            grasps.append([int(grasp_point[1]), int(grasp_point[0]), int(grasp_width), int(grasp_height), grasp_angle])
        
        return grasps

    
    def pixel_to_ee(self, x, y):
        pc_msg = rospy.wait_for_message('/points', PointCloud2, timeout=5)
        pc_np = np.asarray(list(point_cloud2.read_points(pc_msg))).reshape(self.img_height, self.img_width, 3)
        
        point_3d_camera_frame = pc_np[720-y,1280-x]

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
        depth_image = self.depth_inpaint(depth_image)
        depth_image = 1 - depth_image
        depth_image = depth_image[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2]]

        rgb_msg = req.RGBImage
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        rgb_image = rgb_image[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2],:]
        h, w, c = rgb_image.shape

        # rgb, depth = self.pad_to_square(rgb_image, depth_image)
        rgb_norm = self.normalize_and_to_rgb(rgb_image)
        depth_norm = self.normalize_depth(depth_image)

        # print(rgb_norm.shape, depth_norm.shape)

        if len(depth_norm.shape) < 3:
            depth_norm = np.expand_dims(depth_norm, -1)
        
        input_tensor = torch.from_numpy(np.concatenate([rgb_norm, depth_norm], axis=-1)).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.net.device)

        # ==========================================
        # Process input with network
        # ==========================================
        with torch.no_grad():
            pos_output, cos_output, sin_output, width_output = self.net(input_tensor)
        
        grasps = self.post_process_output(pos_output, cos_output, sin_output, width_output, 10)

        # ==========================================
        # Get 3D grasp poses
        # ==========================================
        res = DetectGraspResponse()
        for i in range(len(grasps)):
            grasp_msg = Grasp()
            g = grasps[i]

            # print(g)

            grasp_msg.conf = 1.0
            grasp_msg.obj_cls = 0
            grasp_msg.x = g[0]
            grasp_msg.y = g[1]
            grasp_msg.w = g[2]
            grasp_msg.h = g[3]
            grasp_msg.rot = g[4]
            grasp_msg.ee_pose.pose.position = self.pixel_to_ee(grasp_msg.x, grasp_msg.y).point

            # Calculate quaterion
            q = p.getQuaternionFromEuler([-np.pi, 0, grasp_msg.rot-np.pi/2]) 
            grasp_msg.ee_pose.pose.orientation.x = q[0]
            grasp_msg.ee_pose.pose.orientation.y = q[1]
            grasp_msg.ee_pose.pose.orientation.z = q[2]
            grasp_msg.ee_pose.pose.orientation.w = q[3]

            res.results.append(grasp_msg)
        rospy.loginfo("Finised detecting grasps")

        if self.visualize:
            fig = plt.figure(figsize=(10, 10))

            plt.ion()
            plt.clf()
            ax = fig.add_subplot(2, 3, 1)
            ax.imshow(rgb_image)
            ax.set_title('RGB')
            ax.axis('off')

            ax = fig.add_subplot(2, 3, 2)
            ax.imshow(depth_image, cmap='gray')
            ax.set_title('Depth')
            ax.axis('off')

            ax = fig.add_subplot(2, 3, 3)
            plot = ax.imshow(pos_output.cpu().numpy().squeeze(), cmap='jet', vmin=0, vmax=1)
            ax.set_title('Quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 4)
            ang_img = (torch.atan2(sin_output, cos_output) / 2.0).cpu().numpy().squeeze()
            plot = ax.imshow(ang_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
            ax.set_title('Angle')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 5)
            plot = ax.imshow(width_output.cpu().numpy().squeeze(), cmap='jet', vmin=0, vmax=1)
            ax.set_title('Width')
            ax.axis('off')
            plt.colorbar(plot)

            for g in grasps:
                center_x, center_y, width, height, rad = g
                theta = rad / np.pi * 180
                box = ((center_x, center_y), (width, height), -(theta))
                box = cv2.boxPoints(box)
                box = np.int0(box)
                # cv2.drawContours(img_fused, [box], 0, color, 2)

                p1, p2, p3, p4 = box
                length = width
                p5 = (p1+p2)/2
                p6 = (p3+p4)/2
                p7 = (p5+p6)/2

                cv2.circle(rgb_image, (int(p7[0]), int(p7[1])), 2, (0,0,255), 2)
                cv2.line(rgb_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 3, 8)
                cv2.line(rgb_image, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 3, 8)
                cv2.line(rgb_image, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (255,0,0), 2, 8)

            ax = fig.add_subplot(2, 3, 6)
            ax.imshow(rgb_image)
            ax.set_title('RGB')
            ax.axis('off')

            fig.savefig('results.png')

            fig.canvas.draw()
            plt.close(fig)

        return res


if __name__ == "__main__":
    demo = ConvNetROS()

    rospy.spin()