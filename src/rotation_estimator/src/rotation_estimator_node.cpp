// std
#include <iostream>
#include <string> 

// third party 

// ros 
#include <ros/ros.h>
#include "callbacks.hpp"




using namespace std; 


// callback function 



int main(int argc, char** argv)
{

    ros::init(argc, argv, "rotation_estimator");
    ros::start();

    ros::NodeHandle nh("~");

    string yaml; 
    nh.param<string>("yaml", yaml, "");


    ImageGrabber imageGrabber; 
    ros::Subscriber image_sub = nh.subscribe("/raw_image", 10, &ImageGrabber::GrabImage, &imageGrabber);


    ros::spin();  // wait for callbacks 


    cout << "shutdown rotation estimator" << endl;
    return 0;
}