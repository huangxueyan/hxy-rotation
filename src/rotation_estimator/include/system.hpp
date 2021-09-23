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
#include <queue>
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

    Eigen::Matrix3d get_local_rotation_b2f(bool inverse = false); 
    Eigen::Matrix3d get_global_rotation_b2f(size_t idx_t1, size_t idx_t2);


// imgproc
    void undistortEvents();
    void getWarpedEventImage(const Eigen::Vector3d & temp_ang_vel,
        const PlotOption& option = PlotOption::U16C3_EVNET_IMAGE_COLOR); 
    void getWarpedEventPoints(const EventBundle& eventIn, EventBundle& eventOut,
        const Eigen::Vector3d& cur_ang_vel, const Eigen::Vector3d& cur_ang_pos=Eigen::Vector3d::Zero(), double delta_time=0);
    cv::Mat getImageFromBundle(EventBundle& eventBundle,
        const PlotOption option = PlotOption::U16C3_EVNET_IMAGE_COLOR, bool is_mapping=false);

    void getMapImage(); 

// optimize
    void localCM(); 

    void EstimateMotion_kim();  
    void EstimateMotion_ransca();
    Eigen::Vector3d DeriveErrAnalytic(const Eigen::Vector3d &vel_angleAxis, const Eigen::Vector3d &pos_angleAxis);
    Eigen::Vector3d DeriveTimeErrAnalytic(const Eigen::Vector3d &vel_angleAxis, 
        const std::vector<int>& vec_sampled_idx, double warp_time, double& residuals);
    double getTimeResidual(int sampled_x, int sampled_y, double sampled_time, double warp_time);

// visualize 
    void visualize();


private:

    string yaml;  // configration 

// motion 
    vector<double> vec_curr_time;
    vector<Eigen::Vector3d> vec_angular_velocity;
    vector<Eigen::Vector3d> vec_angular_position;


// optimization 
    cv::Mat cv_3D_surface_index, cv_3D_surface_index_count ;

// camera param
    CameraPara camera; 

// image data 
    ImageData curr_imageData; 

    // image output 
    cv::Mat curr_raw_image,             // grey image from ros      
            curr_undis_image,           // undistort grey image  
            curr_event_image,           // current blur event image 
            curr_undis_event_image,     // current undistorted event image 
            curr_warpped_event_image,   // current sharp local event image 
            curr_map_image,            // global image at t_curr view
            hot_image;                  // time surface with applycolormap
// undistor 
    cv::Mat undist_mesh_x, undist_mesh_y;  

// event data
    EventBundle  eventBundle;             // current blur local events 
    EventBundle  event_undis_Bundle;      // current undistort local events 
    EventBundle  event_warpped_Bundle;    // current sharp local events 
    EventBundle  event_Map_Bundle;        // current sharp local events the that warp to t0. 
    queue<EventData>  que_eventData;     // saved eventData inorder to save

// map 3d 
    vector<EventBundle> vec_Bundle_Maps;  // all the eventbundles that warpped to t0.  


// event bundle, how many msgs to build a bundle 
    double delta_time = 0.01;       // 0.01 seconds
    int max_store_count = int(1e5); // max local event nums


// pose 
    vector<PoseData> vec_gt_poseData; // stored rosmsg 

    // 逻辑是t2-t1->t0. 如eq(3)所示
    Eigen::Vector3d gt_angleAxis ; // gt anglar anxis from t2->t1,.  = theta / delta_time 
    Eigen::Vector3d est_angleAxis; // estimated anglar anxis from t2->t1.  = theta / delta_time 

// output 
    fstream gt_theta_file, gt_velocity_file; 

};





