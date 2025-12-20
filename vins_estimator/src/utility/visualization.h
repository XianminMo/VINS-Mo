#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../estimator.h"
#include "../parameters.h"
#include <fstream>
#include <vector>

// ========================================================================
// Depth Constraint Visualization Structure
// ========================================================================
struct DepthConstraintDebugInfo
{
    Eigen::Vector3d feature_pos_world;  // Feature 3D position in world frame
    Eigen::Vector3d camera_pos_world;   // Camera center in world frame
    bool accepted;                      // true: accepted, false: rejected by chi2
    double chi2_value;                  // Chi-square error value
    int feature_id;                     // Feature ID for debugging
};

extern ros::Publisher pub_odometry;
extern ros::Publisher pub_path, pub_pose;
extern ros::Publisher pub_cloud, pub_map;
extern ros::Publisher pub_key_poses;
extern ros::Publisher pub_ref_pose, pub_cur_pose;
extern ros::Publisher pub_key;
extern nav_msgs::Path path;
extern ros::Publisher pub_pose_graph;
extern ros::Publisher pub_depth_constraints;  // NEW: Depth constraint visualization
extern int IMAGE_ROW, IMAGE_COL;

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyframe(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

void pubDepthMap(const cv::Mat &depth_map, const std_msgs::Header &header);

/**
 * @brief Publish depth constraint visualization markers
 * @param debug_info Vector of depth constraint information (feature pos, camera pos, status)
 * @param header ROS message header with timestamp
 */
void pubDepthConstraints(const std::vector<DepthConstraintDebugInfo>& debug_info,
                        const std_msgs::Header &header);