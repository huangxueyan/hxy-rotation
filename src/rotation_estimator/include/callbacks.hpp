#pragma once 


// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <cv_bridge/cv_bridge.h>


// std 
#include <vector> 
#include <string>


// thirdparty 


using namespace std;

class ImageGrabber
{
public:
    ImageGrabber();
    
    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
}; 


class EventGrabber
{
public:
    EventGrabber();
    
    void GrabberEvent(const dvs_msgs::EventArrayConstPtr & msg);
}; 

