#pragma once 


// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <cv_bridge/cv_bridge.h>


// third party 
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// std 
#include <vector> 
#include <string>
#include <fstream>
#include <cmath>

// self 
#include "database.hpp"
#include "numerics.hpp"


using namespace std; 

/**
* \brief receive ros_msg + imgproc + optimize + visualize 
*/
class System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    System(const string& yaml);
    ~System();

// ros msg 
    void pushEventData(EventData& eventData); 
    void pushimageData(ImageData& imageData); 
    void pushPoseData(PoseData& poseData);

// imgproc
    void undistortEvents();
    void getWarpedEventImage(const Eigen::Vector3f & temp_ang_vel,
        const PlotOption& option = PlotOption::SIGNED_EVNET_IMAGE_COLOR); 
    void getWarpedEventPoints(const Eigen::Vector3f & cur_ang_vel, 
        const Eigen::Vector3f& cur_ang_pos=Eigen::Vector3f::Zero());
    void getImageFromBundle(EventBundle& eventBundle, cv::Mat& image,
        const PlotOption& option = PlotOption::SIGNED_EVNET_IMAGE_COLOR);

// optimize

// visualize 
    void visualize();



private:

    string yaml;  // configration 

// motion 
    vector<double> vec_curr_time;
    vector<Eigen::Vector3f> vec_angular_velocity;
    vector<Eigen::Vector3f> vec_angular_position;


// optimization 


// camera param
    CameraPara camera; 

// image data 
    ImageData curr_imageData; 

    // image output 
    cv::Mat curr_raw_image, curr_undis_image, 
            curr_event_image, curr_undis_event_image, curr_warpped_event_image, 
            curr_map_image;

// undistor 
    cv::Mat undist_mesh_x, undist_mesh_y;  

// event data
    EventBundle  eventBundle;           // local events 
    EventBundle  event_undis_Bundle;    // local events 
    EventBundle  event_warpped_Bundle;    // local events 
    
    vector<EventData> vec_eventData; 
    // EventBundle eventBundle;  ordered event data, not like ros msg 

// event bundle 
    double delta_time = 0.01;       // 0.01 seconds
    int max_store_count = int(1e5); // max local event nums


// pose 
    vector<PoseData> vec_gt_poseData; 
    Eigen::Vector3f gt_angular_velocity; 

// output 
    fstream gt_theta_file, gt_velocity_file; 

};





