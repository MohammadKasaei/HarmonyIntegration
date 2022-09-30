//ROS
#include <ros/ros.h>
//#include <ros/package.h>
//#include <tf/tf.h>
//#include <tf/transform_listener.h>
//#include <ros/service.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_geometry/pinhole_camera_model.h>
#include <depth_image_proc/depth_conversions.h>
//#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <iostream>

using namespace sensor_msgs;
using namespace message_filters;
image_geometry::PinholeCameraModel* cam_model_ptr = nullptr;



class Depth2Points {

private:
	image_geometry::PinholeCameraModel* cam_model_ptr = nullptr;

	ros::Subscriber img_sub;
	ros::Publisher points_pub;


	sensor_msgs::PointCloud2::Ptr depth_to_pointcloud(const sensor_msgs::ImageConstPtr& dimg)
	{
		sensor_msgs::PointCloud2::Ptr cloud_msg(new sensor_msgs::PointCloud2);
		cloud_msg->header = dimg->header;
		cloud_msg->height = dimg->height;
		cloud_msg->width  = dimg->width;
		cloud_msg->is_dense = false;
		cloud_msg->is_bigendian = false;
		sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
		pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

		std::cout << "Undistort Depth Image" << std::endl;
		//to cv::mat type
		cv_bridge::CvImagePtr imPtr = cv_bridge::toCvCopy(dimg);
		cv::Mat rect;
		cam_model_ptr->rectifyImage(imPtr->image,rect,0);

		//back to ROS msg
		cv_bridge::CvImage rectCV(imPtr->header,imPtr->encoding,rect);
		auto rectMsg = rectCV.toImageMsg();

		std::cout << "Converting Depth Image to PointCloud" << std::endl;
		//auto time_start = std::chrono::system_clock::now();

		if (dimg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 || dimg->encoding == sensor_msgs::image_encodings::MONO16)
		{
			depth_image_proc::convert<uint16_t>(rectMsg, cloud_msg, *cam_model_ptr);
		}
		else if (dimg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
		{
			depth_image_proc::convert<float>(rectMsg, cloud_msg, *cam_model_ptr);
		}
		else
		{
			std::cout << "ERROR dealing with depth image - encoding problem - returning nullptr" << std::endl;
			return nullptr;
		}
		//auto time_finished = std::chrono::system_clock::now();

		//approx. 2 ms
		//std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(time_finished - time_start).count()/1000 << " us" << std::endl;

		return cloud_msg;
	}

	void imageCallback(const ImageConstPtr& image) {
		std::cout << "receive depth image and camera info" << std::endl;
  		auto points = depth_to_pointcloud(image);

		points_pub.publish(points);

	}

public:
	void run(int argc, char** argv) {
		ROS_INFO("----------INIT----------");
		ros::init (argc, argv, "depth_to_points");
		ros::NodeHandle nh;
		ROS_INFO("----Waiting for image----");

		const sensor_msgs::CameraInfoConstPtr& cam_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/kinect_subordinate/rgb/camera_info",nh);
		image_geometry::PinholeCameraModel cam_model;
		cam_model.fromCameraInfo(cam_info);
		cam_model_ptr = &cam_model;

		ros::Duration(1.0).sleep();

		img_sub = nh.subscribe("/kinect_subordinate/depth_to_rgb/image_raw", 1000, &Depth2Points::imageCallback, this);
		points_pub = nh.advertise<sensor_msgs::PointCloud2>("/points", 1);

		

		ros::spin();

	}

};

int main(int argc, char** argv) {
	Depth2Points demo;
	demo.run(argc, argv);

	return 0;
}