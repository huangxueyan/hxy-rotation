#pragma once 

// std 
#include <vector>
#include <string> 
#include <iostream>

// ros 
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std; 


struct EventData
{
    double time_stamp; 
    vector<dvs_msgs::Event> event;
};

struct ImageData
{
    double time_stamp; 
    cv::Mat image;
    uint32_t seq; 
};

struct PoseData
{
    double time_stamp; 
    uint32_t seq;
    Eigen::Quaterniond quat; 
    Eigen::Vector3f pose; 
};

struct CameraPara
{

    CameraPara(); 
    double fx;
    double fy;
    double cx;
    double cy;
    double rd1;
    double rd2;

    int width, height; 

    cv::Mat cameraMatrix, distCoeffs ;
    
    Eigen::Matrix3f eg_cameraMatrix, eg_MapMatrix;
    
};

enum PlotOption
{
    SIGNED_EVNET_IMAGE_COLOR
};


/**
* \brief a set of local events, stores as Eigen::Matrix2xf.
*/
struct EventBundle{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EventBundle(): size(0) {}
    EventBundle(const EventBundle& eb);

    // core opearte
    void Append(EventData& eventData);
    void CopySize(const EventBundle& eb); 

    void Clear(); 
    void DiscriminateInner(int widht, int height);

    // image process 
    // GetEventImage() // options with signed or color 
    void InverseProjection(Eigen::Matrix3f& K);
    void Projection(Eigen::Matrix3f& K);


    // events in eigen form used as 3d porjection    
    Eigen::Matrix2Xf coord;    // row, col = (2,pixels)
    Eigen::Matrix3Xf coord_3d;

    // relative time of event
    Eigen::VectorXf time_delta;       // front is beginning
    Eigen::VectorXf time_delta_rev;   // back is  beginning

    // the estimate para of local events
    Eigen::Vector3f angular_velocity, angular_position; 

    // events in vector form, used as data storage
    double abs_tstamp;          // receiving time at ROS system
    ros::Time first_tstamp, last_tstamp; // event time 
    vector<float> x, y;         // event coor
    vector<bool> polar;         // event polar
    vector<bool> isInner;       // indicating the event is inner 
    size_t size; 

};
