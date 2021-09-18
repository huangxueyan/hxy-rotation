
#include "system.hpp"


using namespace std;

System::System(const string& yaml)
{
    cout << "init system" << endl;

    cv::FileStorage fSetting(yaml, cv::FileStorage::READ);
    if(!fSetting.isOpened())
    {
        ROS_ERROR("counld not open file %s", yaml.c_str());
    }

    // undistore 
    undist_mesh_x, undist_mesh_y;  
    cv::initUndistortRectifyMap(camera.cameraMatrix, camera.distCoeffs, 
                cv::Mat::eye(3,3,CV_32FC1), camera.cameraMatrix, cv::Size(camera.width, camera.height), 
                CV_32FC1, undist_mesh_x, undist_mesh_y);

    // visualize 

    // before processing 
    // cv::namedWindow("curr_raw_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_undis_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_event_image", cv::WINDOW_NORMAL);

    // cv::namedWindow("curr_event_image_fc3", cv::WINDOW_NORMAL);

    // after processing 
    cv::namedWindow("curr_undis_event_image", cv::WINDOW_NORMAL);
    cv::namedWindow("curr_warpped_event_image", cv::WINDOW_NORMAL);

    cv::namedWindow("curr_map_image", cv::WINDOW_NORMAL);

    // before processing 
    curr_undis_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_raw_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_event_image = cv::Mat(camera.height,camera.width, CV_32FC3);

    // curr_event_image_fc3 = cv::Mat(camera.height,camera.width, CV_8UC3);
    
    // after processing 
    curr_undis_event_image = cv::Mat(camera.height,camera.width, CV_32F);
    curr_map_image = cv::Mat(camera.height_map,camera.width_map, CV_32F);
    curr_warpped_event_image = cv::Mat(camera.height,camera.width, CV_32F);

    // output file 
    gt_theta_file = fstream("/home/hxt/Desktop/hxy-rotation/data/saved_gt_theta.txt", ios::out);
    gt_velocity_file = fstream("/home/hxt/Desktop/hxy-rotation/data/saved_gt_theta_velocity.txt", ios::out);

}

System::~System()
{
    cout << "saving files " << endl;
    cv::destroyAllWindows();
    gt_theta_file.close();
    gt_velocity_file.close();

}



/**
* \brief undistr events, and save the data to event_undis_Bundle (2d and 3d).
*/
void System::undistortEvents()
{
    int point_size = eventBundle.size;
    cout << "------undisotrt events num:" << point_size <<  endl;
    // cout << "------undisotrt eventBundle cols " << eventBundle.coord.rows() << "," << eventBundle.coord.cols()  <<  endl;
    
    vector<cv::Point2f> raw_event_points(point_size), undis_event_points(point_size);

    for(size_t i=0; i<point_size; ++i)
        raw_event_points[i] = cv::Point2f(eventBundle.coord(0,i),eventBundle.coord(1,i));
    
    cv::undistortPoints(raw_event_points, undis_event_points, 
            camera.cameraMatrix, camera.distCoeffs, cv::noArray(), camera.cameraMatrix);
    
    // convert points to cv_mat 
    cv::Mat temp_mat = cv::Mat(undis_event_points); 
    temp_mat = temp_mat.reshape(1,point_size); // channel 1, row = 2
    cv::transpose(temp_mat, temp_mat);
    
    // convert cv2eigen 
    event_undis_Bundle.CopySize(eventBundle); 
    cv::cv2eigen(temp_mat, event_undis_Bundle.coord); 
    
    // store 3d data
    event_undis_Bundle.InverseProjection(camera.eg_cameraMatrix); 
    
    // store inner 
    event_undis_Bundle.DiscriminateInner(camera.width, camera.height);

    getImageFromBundle(event_undis_Bundle, PlotOption::U16C3_EVNET_IMAGE_COLOR, false).convertTo(curr_undis_event_image, CV_32F);
    cout << "success undistort events " << endl;
}


/**
* \brief Constructor.
* \param is_mapping means using K_map image size.
*/
cv::Mat System::getImageFromBundle(EventBundle& cur_event_bundle, const PlotOption& option, bool is_mapping /*=false*/)
{

    cout << "getImageFromBundle " << cur_event_bundle.coord.cols() << ", is_mapping "<< is_mapping << endl;
    cout << "enter for interval " << "cols " << cur_event_bundle.isInner.rows()<< endl;

    cv::Mat image;

    int width = camera.width, height = camera.height; 

    if(is_mapping)
    {
        width = camera.width_map; 
        height = camera.height_map; 
    }
    cout << "  image size (h,w) = " << height << "," << width << endl;

    switch (option)
    {
    case PlotOption::U16C3_EVNET_IMAGE_COLOR:
        cout << "  choosing U16C3_EVNET_IMAGE_COLOR" << endl;
        image = cv::Mat(height,width, CV_16UC3);
        image = cv::Scalar(0,0,0); // clear first 
        
        for(int i=0; i<cur_event_bundle.coord.cols(); i++)
        {
            // bgr
            int bgr = eventBundle.polar[i] ? 2 : 0; 
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1]; 

            // descriminate inner 
            if(cur_event_bundle.isInner(i) < 1)
                continue;

            if(x >= width  ||  x < 0 || y >= height|| y < 0 ) 
                cout << "x, y" << x << "," << y << endl;
            
            // x = x >= width  ? width -1 : x; 
            // y = y >= height ? height-1 : y; 
            // x = x < 1 ? 0 : x; 
            // y = y < 1 ? 0 : y; 

            // cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x,y);

            image.at<cv::Vec3w>(point_temp) += (eventBundle.polar[i] ? cv::Vec3w(0, 0, 1) : cv::Vec3w(1, 0, 0));
            // image.at<cv::Vec3s>(point_temp) += (eventBundle.polar[i] ? cv::Vec3s(1, 0, 0) : cv::Vec3s(0, 0, 1));
        }

        cout << "image size"  << image.size() <<endl;
        break;
    
    case PlotOption::U16C1_EVNET_IMAGE:
        cout << "enter case U16C1_EVNET_IMAGE" << endl;
        image = cv::Mat(height, width, CV_16UC1);
        image = cv::Scalar(0);

        for(int i=0; i<cur_event_bundle.coord.cols(); i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1]; 

            if(cur_event_bundle.isInner(i) < 1) continue;

            if(x >= width  ||  x < 0 || y >= height || y < 0 ) 
                cout << "x, y" << x << "," << y << endl;

            // cout << "x, y" << x <<  "," << y << endl;

            // x = x >= width  ? width -1 : x; 
            // y = y >= height ? height-1 : y; 
            // x = x < 1 ? 0 : x; 
            // y = y < 1 ? 0 : y; 

            cv::Point2i point_temp(x,y);

            image.at<unsigned short>(point_temp) += 1;
        }
        break;
    default:
        cout << "default choice " << endl;
        break;
    }

    cout << "  success get image " << endl;
    return image;
}



/**
* \brief get rotatino from last (sharp) bundle to first bundle rotation. 
* \param idx_t1 front idx
*/
Eigen::Matrix3d System::get_global_rotation_b2f(size_t idx_t1, size_t idx_t2)
{
    // from frist bundle to world coord
    Eigen::Matrix3d R1 = vec_gt_poseData[idx_t1].quat.toRotationMatrix();
    Eigen::Matrix3d R2 = vec_gt_poseData[idx_t2].quat.toRotationMatrix();

    return R1.transpose()*R2;
}


/**
* \brief get_local_rotation_b2f using current eventbundle, return the rotation matrix from t1(start) to t2(end). 
*/
Eigen::Matrix3d System::get_local_rotation_b2f()
{
    int target1_pos, target2_pos; 
    size_t start_pos = vec_gt_poseData.size() > 50 ? vec_gt_poseData.size()-50 : 0;

    double interval_1 = 1e5, interval_2 = 1e5; 

    for(size_t i = start_pos; i< vec_gt_poseData.size(); i++)
    {
        // cout << "ros time " <<  std::to_string(vec_gt_poseData[i].time_stamp_ros.toSec()) << endl; 
        if(abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.first_tstamp).toSec()) < interval_1)
        {
            target1_pos = i;
            interval_1 = abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.first_tstamp).toSec());
        }

        if(abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.last_tstamp).toSec()) < interval_2)
        {
            target2_pos = i;
            interval_2 = abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.last_tstamp).toSec());
        }
    }

    // TODO NSECONDS, and check the event timestamp and pose time stamp match. 
    // cout << "event first time " << std::to_string(eventBundle.first_tstamp.toSec()) <<  
    //         ", pose time: "<< std::to_string(vec_gt_poseData[target1_pos].time_stamp_ros.toSec())<<endl;
    // cout << "event  last time " << std::to_string(eventBundle.last_tstamp.toSec()) <<  
    //         ", pose time: "<< std::to_string(vec_gt_poseData[target2_pos].time_stamp_ros.toSec())<<endl;


    Eigen::Matrix3d R1 = vec_gt_poseData[target1_pos].quat.toRotationMatrix();
    Eigen::Matrix3d R2 = vec_gt_poseData[target2_pos].quat.toRotationMatrix();

    return R1.transpose()*R2;

}



/**
* \brief input evene vector from ros msg.
* \param[in] eventData event array of dvs_msg::Event .
*/
void System::pushEventData(EventData& eventData)
{
    // save in vector eventData and event bundle (local events)
    // vec_eventData.push_back(eventData);  // in system 
    eventBundle.Append(eventData);       // in event bundle 

    // check the time interval is match 
    double time_inteval = (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec();

    if(time_inteval > delta_time || eventBundle.coord.cols() > max_store_count)
    {
        cout << "----processing event bundle" << endl; 

        /* undistort events */ 
        undistortEvents();

        /* estimate current motion */ 
        if(vec_angular_velocity.empty())
        {
            cout << "no engough pose" << endl;
            eventBundle.Clear();
            return;
        }
        else
        {
            cout << "engough pose" << endl;
        }

         /* get local bundle sharper */ 
        Eigen::Matrix3d R_t1_t2 = get_local_rotation_b2f();
        Eigen::AngleAxisd ang_axis(R_t1_t2);
        double _delta_time = eventBundle.time_delta[eventBundle.time_delta.rows()-1]; 
        ang_axis.angle() /= _delta_time;  // get angular velocity

        getWarpedEventImage(ang_axis.axis() * ang_axis.angle());
        cout<< "output getWarpedEventImage" << endl;


        /* get global maps */ 
        getMapImage();


        // show event image 

        visualize();     
        // clear event bundle 
    }

    eventBundle.Clear();
}


/**
* \brief input evene vector from ros msg.
* \param[in] ImageData self defined imagedata.
*/
void System::pushimageData(ImageData& imageData)
{

    // can be save in vector 

    // update current image 
    curr_imageData = imageData;  // copy construct 
    curr_raw_image = imageData.image.clone(); 

    // undistort image 
    cv::remap(curr_raw_image, curr_undis_image, undist_mesh_x, undist_mesh_y, cv::INTER_LINEAR );
}

void System::pushPoseData(PoseData& poseData)
{

    // Eigen::Vector3d v_2 = poseData.quat.toRotationMatrix().eulerAngles(2,1,0);
    Eigen::Vector3d curr_pos = toEulerAngles(poseData.quat);

    int loop = 3; 
    if(vec_gt_poseData.size() > 6)
    {
        // [* * * target * * *] to get target velocity.  
        vector<PoseData>::iterator it = vec_gt_poseData.end() - loop - 1;  

        Eigen::Vector3d velocity(0,0,0); 

        for(int k = 1; k<=loop; ++k)
        {
            // FIXME rpy: zyx, so v_1=(theta_z,y,x)
            // Eigen::Vector3d v_1 = (*it).quat.toRotationMatrix().eulerAngles(2,1,0);
            // Eigen::Vector3d v_2 = (*(it-loop)).quat.toRotationMatrix().eulerAngles(2,1,0);
            
            Eigen::Vector3d v_1 = toEulerAngles((*(it-loop-1+k)).quat);
            Eigen::Vector3d v_2 = toEulerAngles((*(it+k)).quat);

            double delta_time = (*(it+k)).time_stamp - (*(it-loop-1+k)).time_stamp  ; 

            Eigen::Vector3d delta_theta = v_2 - v_1;             
            velocity += delta_theta / delta_time;

            // cout<< "loop " << k << " delta_t: " <<delta_time 
            //     << ", delta_theta: " << delta_theta.transpose() <<", vel: " << (delta_theta / delta_time).transpose() << endl;
            // cout << "pose delta time " << delta_time << endl;
        }

        velocity = velocity.array() / loop;

        // Eigen::Vector3d velocity(0,0,0); 
        // double delta_time = poseData.time_stamp - vec_gt_poseData.back().time_stamp;
        // Eigen::Vector3d theta_1 = toEulerAngles(poseData.quat);
        // Eigen::Vector3d theta_2 = toEulerAngles(vec_gt_poseData.back().quat);
        // Eigen::Vector3d delta_theta = theta_1 - theta_2; 
        // // cout << "theta 1: " << theta_1.transpose() <<"\ntheta 2: " << theta_2.transpose() << "\ndelta: " << delta_theta.transpose() << endl; 
        // velocity = delta_theta / delta_time;
    
        // cout << "  final velocity " << (velocity.array()/3.14*180).transpose() << endl;
        gt_velocity_file << poseData.time_stamp <<" " << velocity.transpose() << endl;
        
        vec_angular_velocity.push_back(velocity);
        vec_curr_time.push_back(poseData.time_stamp);

        gt_theta_file << (*(vec_gt_poseData.end()-loop)).time_stamp <<" " << curr_pos.transpose() << endl;
    }


    vec_gt_poseData.push_back(poseData);
    // cout << "push pose to system " << endl;
}


void System::visualize()
{

    cout << "visualize" << endl; 

    // TODO update all images 

    // cv::imshow("curr_raw_image", curr_raw_image);
    // cv::imshow("curr_undis_image", curr_undis_image);
    // cv::imshow("curr_event_image", curr_event_image);

    // cout << "channels " << curr_undis_event_image.channels() << 
        // "types " << curr_undis_event_image.type() << endl;
    cv::imshow("curr_undis_event_image", curr_undis_event_image);
    cv::imshow("curr_warpped_event_image", curr_warpped_event_image);
    cv::imshow("curr_map_image", curr_map_image);

    // cv::imshow("curr_event_image_fc3", curr_event_image_fc3);
    cv::waitKey(100);

}


