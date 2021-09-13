// std
#include <iostream>
#include <string> 

// third party 

// ros 
#include <ros/ros.h>
#include "callbacks.hpp"
#include "system.hpp"



using namespace std; 


// rosbag play -r 0.1 ~/Documents/rosbag/Mono-rosbag/slider_depth.bag

int main(int argc, char** argv)
{

    ros::init(argc, argv, "rotation_estimator");
    ros::start();

    ros::NodeHandle nh("~");

    string yaml;  // system configration 
    nh.param<string>("yaml", yaml, "");

    System sys(yaml);

    // ImageGrabber imageGrabber(&sys); 
    // ros::Subscriber image_sub = nh.subscribe("/dvs/image_raw", 10, &ImageGrabber::GrabImage, &imageGrabber);
    
    EventGrabber eventGrabber(&sys);
    ros::Subscriber event_sub = nh.subscribe("/dvs/events", 10, &EventGrabber::GrabEvent, &eventGrabber);

    PoseGrabber poseGrabber(&sys);
    ros::Subscriber pose_sub = nh.subscribe("/optitrack/davis", 100, &PoseGrabber::GrabPose, &poseGrabber);

    ros::spin();  // exec for callbacks 

    cout << "shutdown rotation estimator" << endl;
    return 0;
}

