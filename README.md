## Harmony Project ---- Edinburgh

### Package overview:

#### 1. depth_image_proc: 

​	This package provides useful library for processing depth image

#### 2. Depth_Image_to_PointCloud: 

​	This package provide two nodes:

  1. compress_to_raw.py (Python): node for convert compressed Image topics to raw image topics. This node is only used when you need to deal with compressed image topics from recorded rosbag.

  2. depthImage_to_PointCloud (C++): node for generate 3D point cloud from depth image.

     Please modify the topic names of depth image and camera info according to the real robot setup.

     The generated point clouds will be published into "/points" topic with the reference frame of "kinect_subordinate_rgb_camera_link"

#### 3. grasp_detection_frame

​	this package is composed of two packages: 

 	1. grasp_detection_core: package for main grasp detection modules.
 	2. grasp_detection_msgs: pre-defined custom ROS messages for grasp detection.





### Basic Requirements:

- pybullet
- opencv-python
- matplotlib
- scikit-image
- pytorch (1.10.2 + CUDA 11.3)
- pip install --extra-index-url https://otamachan.github.io/rospy-index rospy-all **(for install ROS basic packages in conda environment)**





### Usage:

#### 0. Before running

```
catkin build & catkin_make

source devel/setup.bash

# Download pre-trained weight of SSG and put the .pth file in the pretraned-models folder
https://drive.google.com/file/d/1rbdNy0XShpLbQG5aLZEnngefQhmFbmOQ/view?usp=sharing
```



#### 1. Run with rosbag

```python
# 1. preprocess rosbag
##### the compressed RGB & depth image will be published into a new topic
##### Please check the compress_to_raw.py for details
##### Then 3D point clouds will be generated from depth image using camera model from camera info topic
source devel/setup.bash
roslaunch Depth_Image_to_PointCloud preprocess_rosbag.launch

# 2. Start SSG service
source devel/setup.bash
cd path/to/grasp_detection_core/src
python ssg_ros.py # Remember to run with correct python env

# 3. Test SSG service
source devel/setup.bash
cd path/to/grasp_detection_core/src
python test_ssg.py

### NOTE:
# of courese you can use rosrun to start those python nodes
# Just by adding #!/usr/bin/env python3 to the beginning of each python node
# I'm using anaconda env to execute those python nodes, so I prefer to directly use python xxx.py to run them
```



#### 3. Run with real robot (NOT TESTED)

```python
# 1. Start SSG service
source devel/setup.bash
cd path/to/grasp_detection_core/src
python ssg_ros.py # Remember to run with correct python env

# 2. Test SSG service
source devel/setup.bash
cd path/to/grasp_detection_core/src
python test_ssg.py



### NOTE:
# The service will receive a depth image message and a RGB image message as input
# and return a array of custom Grasp message as output (Please check the grasp_detection_msgs package for details)
# The returned grasps is composed of grasp configuration (x, y, width, height, rotation) in image plane, and the 3D  grasp configuration (position, orientation in quaternion) in "myumi_005_base_link" frame.
# Regarding the 3D grasp configuration, the 3D position has already been tested, but the orientation value remains untested.
```

