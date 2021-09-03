#include "callbacks.hpp"


ImageGrabber::ImageGrabber()
{
    cout << "hello image grabber" << endl;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{

    cout << "grabe one image " << endl;
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

    cout << "cols" << cv_ptr->image.cols << endl;


}
