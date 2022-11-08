#pragma once 


// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <cv_bridge/cv_bridge.h>


// third party 
#include <eigen3/Eigen/Core>
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
#include <thread> 
#include <mutex>
#include<chrono>


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
    void pushEventData(const std::vector<dvs_msgs::Event>& ros_vec_event);
    void pushEventData(const std::vector<dvs_msgs::Event> &ros_vec_event, int count); //RT + dynamic version
    void pushimageData(const ImageData& imageData); 
    void pushPoseData(const PoseData& poseData);
    void updateEventBundle();  // use less
    void setBeginTime(ros::Time);

    void save_velocity(); // save velocity date to txt

    Eigen::Matrix3d get_local_rotation_b2f(bool inverse = false); 
    Eigen::Matrix3d get_global_rotation_b2f(size_t idx_t1, size_t idx_t2);


// imgproc
    void Run();
    void undistortEvents();
    void undistortEvents(EventBundle &ebin, EventBundle &ebout);
    cv::Mat getWarpedEventImage(const Eigen::Vector3d & temp_ang_vel,EventBundle& event_out,
        const PlotOption& option = PlotOption::U16C3_EVNET_IMAGE_COLOR, bool ref_t1 = false); 

    void getWarpedEvent(EventBundle& eventIn, const Eigen::Vector3d & cur_ang_vel, EventBundle& event_out, bool later_warp = false);
    double GetInsideRatioSingle(EventBundle& evin);
    double GetInsideRatioDouble(EventBundle &evin);
    
    void getWarpedEventPoints(const EventBundle& eventIn, EventBundle& eventOut,
        const Eigen::Vector3d& cur_ang_vel,  bool ref_t1=false);
    cv::Mat getImageFromBundle(EventBundle& eventBundle,
        const PlotOption option = PlotOption::U16C3_EVNET_IMAGE_COLOR);

    // void getMapImage(); 

// optimize
    void localCM(); 

    void EstimateMotion_kim();  
    void EstimateMotion_CM_ceres();
    void EstimateMotion_PPP_ceres();
    void EstimateMotion_ransca_ceres();
    void EstimateMotion_KS_ceres();

    void EstimateRunTime_CM();
    void EstimateRunTime_PPP();
    void EstimateRunTime_Single();
    void EstimateRunTime_Double();

    // void EstimateMotion_ransca_once(double sample_ratio, double warp_time_ratio, double opti_steps);
    // void EstimateMotion_ransca_warp2bottom(double sample_start, double sample_end, double opti_steps);

    Eigen::Vector3d DeriveErrAnalytic(const Eigen::Vector3d &vel_angleAxis, const Eigen::Vector3d &pos_angleAxis);
    

    Eigen::Vector3d DeriveTimeErrAnalyticRansacBottom(const Eigen::Vector3d &vel_angleAxis, 
        const std::vector<int>& vec_sampled_idx, double& residuals);
    Eigen::Vector3d GetGlobalTimeResidual();

    // random warp time version 
    Eigen::Vector3d DeriveTimeErrAnalyticRansac(const Eigen::Vector3d &vel_angleAxis, 
        const std::vector<int>& vec_sampled_idx, double warp_time, double& residuals);
    void getTimeResidual(int sampled_x, int sampled_y, double sampled_time, double warp_time,
            double& residual, double& grad_x, double& grad_y);
    // Eigen::Vector3d DeriveTimeErrAnalyticLayer(const Eigen::Vector3d &vel_angleAxis, 
    //     const std::vector<int>& vec_sampled_idx, double warp_time, double& residuals);

    void getSampledVec(vector<int>& vec_sampled_idx, int samples_count, double sample_start, double sample_end);
    void getSampledVecUniform(vector<int>& vec_sampled_idx, int samples_count, double sample_start, double sample_end);
    
// visualize 
    void visualize();
    
// file 
    bool inline file_opened() {return est_velocity_file.is_open();};

    double total_evaluate_time, 
            total_undistort_time, 
            total_visual_time, 
            total_timesurface_time, 
            total_ceres_time, 
            total_readevents_time, 
            total_eventbundle_time, 
            total_warpevents_time;
    int total_processing_events = 0;


    // bool inline checkEmpty(){return que_vec_eventData.empty();}

// thread
    // thread* thread_run;
    // thread* thread_view;
    // std::mutex que_vec_eventData_mutex;

private:
    cv::Mat cv_later_timesurface_float_;
    cv::Mat cv_early_timesurface_float_; // used for dynamic batchsize
    double last_merge_ratio_;
    // dynamic batch para 
    bool init_flag_ = true;
    double time_length_;
    int batch_length_;
    double outlier_ratio_;
    int invalid_merge_count = 0;
    int invalid_time_count = 0;
    int invalid_batch_count = 0;
    int valid_merge_count = 0;
    double t_threshold_ = 0;


// configration 
    string yaml;  
    int yaml_iter_num;
    int yaml_iter_num_final;
    float yaml_ts_start;
    float yaml_ts_end;
    int   yaml_sample_count;
    int yaml_ceres_iter_num;
    int yaml_gaussian_size;
    float yaml_gaussian_size_sigma;
    int yaml_denoise_num;
    float yaml_default_value_factor; 
    int yaml_ceres_iter_thread;
    double yaml_ros_starttime;
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
    cv::Mat curr_raw_image,              // grey image from ros      
            curr_undis_image,            // undistort grey image  
            curr_event_image,            // current blur event image 
            curr_undis_event_image,      // current undistorted event image 
            curr_warpped_event_image,    // current sharp local event image using est
            curr_warpped_event_image_gt, // current sharp local event image using gt 
            curr_map_image,              // global image at t_curr view
            hot_image_C1,
            hot_image_C3;                  // time surface with applycolormap
// undistor parameter
    cv::Mat undist_mesh_x, undist_mesh_y;  

// event data
    ros::Time beginTS;                    // begin time stamp of ros Time. 

    EventBundle  eventBundle;             // current blur local events 
    EventBundle  event_undis_Bundle;      // current undistort local events 
    EventBundle  event_warpped_Bundle;    // current sharp local events 
    EventBundle  event_warpped_Bundle_gt;    // current sharp local events 
    EventBundle  event_Map_Bundle;        // current sharp local events the that warp to t0. 
    
    // std::queue<std::vector<dvs_msgs::Event>> que_vec_eventData;     // saved eventData inorder to save
    std::vector<std::vector<dvs_msgs::Event>> vec_vec_eventData;     // saved eventData inorder to save
    std::vector<std::vector<double>> vec_vec_eventdepth;
    int vec_vec_eventData_iter;               // used to indicating the position of unpushed events. 

// map 3d 
    vector<EventBundle> vec_Bundle_Maps;  // all the eventbundles that warpped to t0.  


// event bundle, how many msgs to build a bundle 
    double delta_time = 0.01;       // 0.01 seconds
    int max_store_count = int(1e5); // max local event nums


// pose 
    bool using_gt; 
    vector<PoseData> vec_gt_poseData; // stored rosmsg 

    // 逻辑是t2-t1->t0. 如eq(3)所示
    Eigen::Vector3d gt_angleAxis ; // gt anglar anxis from t2->t1,.  = theta / delta_time 
    Eigen::Vector3d est_angleAxis; // estimated anglar anxis from t2->t1.  = theta / delta_time 

    Eigen::Vector3d last_est_var; // 正则项， 控制est_N_norm的大小

// output 
    size_t seq_count;
    fstream est_theta_file, est_velocity_file;
    // fstream est_velocity_file_quat;   // evo

};





