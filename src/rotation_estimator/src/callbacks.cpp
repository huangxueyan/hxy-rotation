#include "callbacks.hpp"
#include "database.hpp"

ros::Time begin_time = ros::Time(0); 

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    if(begin_time == ros::Time(0))
    {
        begin_time =  msg->header.stamp; 
        system->setBeginTime(msg->header.stamp);
    } 

    static int last_img_seq = msg->header.seq; 

    cv_bridge::CvImageConstPtr cv_ptr; 
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg,"mono8"); // gray 8char
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("receive image error: %s", e.what());
        return ;
    }

    // fixme not need to constru a new imageData obj 
    ImageData ImageData; 
    ImageData.image = cv_ptr->image.clone(); 
    ImageData.seq = msg->header.seq; 
    ImageData.time_stamp = (cv_ptr->header.stamp - begin_time).toSec();
    cout << "receiving image t: " << ImageData.time_stamp<< endl;

    system->pushimageData(ImageData);
}


void EventGrabber::GrabEvent(const dvs_msgs::EventArrayConstPtr& msg) 
{
    if(msg->events.empty()) return; 

    if(begin_time == ros::Time(0)) 
    {
        begin_time = msg->events[0].ts; 
        system->setBeginTime(msg->events[0].ts);
    }


    // not need to copy eventdata obj
    // EventData eventdata; 
    // eventdata.event = msg->events; // vector<dvsmsg::event> 
    double time_stamp = (msg->events[0].ts - begin_time).toSec(); 
    double delta_time = (msg->events.back().ts-msg->events.front().ts).toSec();
    cout<<"receiving events " << msg->events.size() <<", time: " << time_stamp<<", delta time " << delta_time <<endl;

    // system->que_vec_eventData_mutex.lock();
    system->pushEventData(msg->events);
    // system->que_vec_eventData_mutex.unlock();

}


void PoseGrabber::GrabPose(const geometry_msgs::PoseStampedConstPtr& msg)
{

    if(begin_time == ros::Time(0))
    {
        begin_time =  msg->header.stamp; 
        system->setBeginTime(msg->header.stamp);
    } 
    
    // not need to copy eventdata obj
    PoseData poseData; 
    poseData.time_stamp = (msg->header.stamp-begin_time).toSec(); 
    poseData.time_stamp_ros = msg->header.stamp; 
    
    // vector<geometry::pose> 
    poseData.pose << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    
    // input w,x,y,z. output and store: x,y,z,w
    poseData.quat = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                msg->pose.orientation.y, msg->pose.orientation.z);
    
    // cout<<"receiving poses t: " << poseData.time_stamp << "， ros: " << std::to_string(poseData.time_stamp_ros.toSec()) << endl;
    // cout << "----(xyz)(wxyz)" << poseData.pose.transpose() <<  poseData.quat.coeffs().transpose() << endl;

    system->pushPoseData(poseData);
}