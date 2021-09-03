#pragma once 

// std
#include <fstream>
#include <sstream>
#include <vector> 
#include <string> 

// ros 
#include <ros/ros.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>


// third party 
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace std;


class Event_reader
{

public:
    Event_reader(std::string yaml="", 
        ros::Publisher* event_array_pub = nullptr, 
        ros::Publisher* event_image_pub_ = nullptr);
    ~Event_reader();

    void read(const std::string& dir);
    void publish();
    void render();

private:

    ros::Publisher* event_array_pub_; 
    ros::Publisher* event_image_pub_; 

    std::vector<dvs_msgs::Event> eventData;  // single event 
    int eventData_counter; 

    int height, width;  // event image size
    
    int event_bundle_size, max_events;    // maximum size of event vector 
    
    double delta_time; // 
    double curr_time;  

    cv_bridge::CvImage event_image; 

};







