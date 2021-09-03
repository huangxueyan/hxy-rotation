// std 
#include <iostream> 
#include <fstream> 
#include <vector> 
#include <string>

// ros 
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h> 
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

// third party
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// self 
#include "event_reader.hpp"


using namespace std; 
using namespace Eigen;



int main(int argc, char** argv)
{

    cout << " hello event publisher " << endl;

    ros::init(argc, argv, "event_publisher");

    ros::NodeHandle nh_private("~");

    string yaml, event_dir; 
    bool loop = false; // mease loop the events
    nh_private.param<string>("event_dir", event_dir,"");
    nh_private.param<string>("yaml", yaml,"");
    nh_private.param<bool>("loop", loop, false);
    
    ros::Publisher event_array_pub = nh_private.advertise<dvs_msgs::EventArray>("/events",10);
    ros::Publisher raw_image = nh_private.advertise<sensor_msgs::Image>("/raw_image",10);
    ros::Publisher event_image = nh_private.advertise<sensor_msgs::Image>("/event_image",10);
    
    ros::Rate loop_rate(1); // 1s 10 times 


    Event_reader event_reader(yaml, &event_array_pub, &event_image); 
    event_reader.read(event_dir);

    while(ros::ok())
    {
        // ros::spinOnce(); // 

        cv_bridge::CvImage cv_image; 
        cv_image.encoding = "bgr8";
        cv_image.image = cv::imread("/home/hxt/Pictures/test.jpg",cv::IMREAD_COLOR);
        sensor_msgs::ImagePtr msg = cv_image.toImageMsg();
        // raw_image.publish(msg);

        event_reader.publish();


        cout << "sending event sessage " << cv_image.image.cols 
            << cv_image.image.rows << cv_image.image.channels()<< endl;

        loop_rate.sleep();
    }

    ROS_INFO("Finish publishing date;");


    return 0;   
}