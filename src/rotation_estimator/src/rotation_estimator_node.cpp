// std
#include <iostream>
#include <string> 

// third party 

// ros 
#include <ros/ros.h>
// #include "callbacks.hpp"
#include "system.hpp"
#include "event_reader.hpp"
// #include <event_publisher/event_reade.hpp>


using namespace std; 

class EventGrabber
{
public:
    EventGrabber(System* sys) : system(sys) {}

    void GrabEvent(const dvs_msgs::EventArrayConstPtr& msg);

    System* system;
};

void EventGrabber::GrabEvent(const dvs_msgs::EventArrayConstPtr& msg)
{
    system->pushEventData(msg->events);
}

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
        // return 0;
    }

    /* Event ROS version */
        EventGrabber eventGrabber(sys);
        ros::Subscriber event_sub = nh.subscribe("/dvs/events", 10, &EventGrabber::GrabEvent, &eventGrabber);
        ros::spin();
    
    /** Event TXT version */ 
    // Event_reader event_reader(yaml); 
    // while (true)
    // {
    //     // read data 
    //     ros::Time t1, t2; 
    //     t1 = ros::Time::now();
    //         dvs_msgs::EventArrayPtr msg_ptr = dvs_msgs::EventArrayPtr(new dvs_msgs::EventArray());
    //         event_reader.acquire(msg_ptr);
    //     t2 = ros::Time::now();
    //     // cout << "event_reader.acquire time " << (t2-t1).toSec() << endl;  // 0.50643


    //     if(msg_ptr==nullptr || msg_ptr->events.empty() )
    //     {
    //         cout << "wrong reading events, msgptr==null" << int(msg_ptr==nullptr) << "empty events " << int(msg_ptr->events.empty()) << endl;
    //         break;
    //     }


    //     t1 = ros::Time::now();
    //         sys->pushEventData(msg_ptr->events);
    //     t2 = ros::Time::now();
    //     // cout << "sys pushEventData" << (t2-t1).toSec() << endl;  // 0.00691187 s

    //     // cout << "success reveive" << endl;
    //     // break;
    // }

    cout << "total evaluate time "<< sys->total_evaluate_time << 
            " total undistort time "<< sys->total_undistort_time << 
            " total visual time "<< sys->total_visual_time << endl;

    cout << "shutdown rotation estimator" << endl;
    return 0;
}

