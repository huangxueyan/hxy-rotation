// std
#include <iostream>
#include <string> 

// third party 

// ros 
#include <ros/ros.h>
#include "callbacks.hpp"
#include "system.hpp"
#include "event_reader.hpp"
// #include <event_publisher/event_reade.hpp>


using namespace std; 


// rosbag play -r 0.1 ~/Documents/rosbag/Mono-rosbag/slider_depth.bag

int main(int argc, char** argv)
{

    ros::init(argc, argv, "rotation_estimator");
    ros::start();

    ros::NodeHandle nh("~");

    string yaml;  // system configration 
    nh.param<string>("yaml", yaml, "");

    System* sys = new System(yaml);
    if(!sys->file_opened())
    {
        cout << "failed opening file " << endl;
        return 0;
    }
    /* Event ROS version */
    // ImageGrabber imageGrabber(&sys); 
    // ros::Subscriber image_sub = nh.subscribe("/dvs/image_raw", 10, &ImageGrabber::GrabImage, &imageGrabber);
    // EventGrabber eventGrabber(sys);
    // ros::Subscriber event_sub = nh.subscribe("/dvs/events", 10, &EventGrabber::GrabEvent, &eventGrabber);
    // // PoseGrabber poseGrabber(sys);
    // // ros::Subscriber pose_sub = nh.subscribe("/optitrack/davis", 10, &PoseGrabber::GrabPose, &poseGrabber);
    // ros::spin();
    
    /** Event TXT version */ 
    ros::Time t1 = ros::Time::now(); 
    Event_reader event_reader(yaml); 
    while (true)
    {
        // read data 
        dvs_msgs::EventArrayPtr msg_ptr = dvs_msgs::EventArrayPtr(new dvs_msgs::EventArray());
        event_reader.acquire(msg_ptr);

        if(msg_ptr==nullptr || msg_ptr->events.empty() )
        {
            cout << "wrong reading events, msgptr==null" << int(msg_ptr==nullptr) << "empty events " << int(msg_ptr->events.empty()) << endl;
            break;
        }

        sys->pushEventData(msg_ptr->events);
        // cout << "success reveive" << endl;
        // break;
    }

    ros::Time t2 = ros::Time::now();
    double total_program_runtime = (ros::Time::now() - t1).toSec(); 
    cout << " total program time "<< total_program_runtime << endl; 
    cout << " total create event bundle time "<< sys->total_eventbundle_time << endl;
    cout << " total evaluate time "<< sys->total_evaluate_time << endl;
    cout << " total warpevents time "<< sys->total_warpevents_time << endl; 
    cout << " total timesurface time "<< sys->total_timesurface_time << endl; 
    cout << " total ceres time "<< sys->total_ceres_time << endl;
    cout << " total undistort time "<< sys->total_undistort_time << endl;
    cout << " total visual time "<< sys->total_visual_time << endl;
    cout << " total processsing events " << sys->total_processing_events / float(1e6) << " M evs" << endl;
    
    cout << "shutdown rotation estimator" << endl;

    return 0;
}

