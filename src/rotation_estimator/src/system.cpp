
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

    // undistore data 
    undist_mesh_x, undist_mesh_y;  
    cv::initUndistortRectifyMap(camera.cameraMatrix, camera.distCoeffs, 
                cv::Mat::eye(3,3,CV_32FC1), camera.cameraMatrix, cv::Size(camera.width, camera.height), 
                CV_32FC1, undist_mesh_x, undist_mesh_y);

    using_gt = true;
    vec_vec_eventData_iter = 0;

    // ros msg 
    // vec_last_event_idx = 0;

    // visualize 
    // before processing 
    // cv::namedWindow("curr_raw_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_undis_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_event_image", cv::WINDOW_NORMAL);

    // after processing 
    // cv::namedWindow("curr_undis_event_image", cv::WINDOW_NORMAL);
    cv::namedWindow("curr_warpped_event_image", cv::WINDOW_NORMAL);
    cv::namedWindow("curr_warpped_event_image_gt", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_map_image", cv::WINDOW_NORMAL);
     cv::namedWindow("hot_image_C3", cv::WINDOW_NORMAL);
     cv::namedWindow("opti", cv::WINDOW_NORMAL);

    // before processing 
    curr_undis_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_raw_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_event_image = cv::Mat(camera.height,camera.width, CV_32FC3);
    hot_image_C1 = cv::Mat(camera.height,camera.width, CV_8UC1);
    hot_image_C3 = cv::Mat(camera.height,camera.width, CV_8UC3);

    // optimizeing 
    int dims[] = {180,240,20};   // row, col, channels
    cv_3D_surface_index = cv::Mat(3, dims, CV_32S);
    cv_3D_surface_index_count = cv::Mat(180, 240, CV_32S);
    
    // after processing 
    curr_undis_event_image = cv::Mat(camera.height,camera.width, CV_32F);
    curr_map_image = cv::Mat(camera.height_map,camera.width_map, CV_32F);
    curr_warpped_event_image = cv::Mat(camera.height,camera.width, CV_32F);
    curr_warpped_event_image_gt = cv::Mat(camera.height,camera.width, CV_32F); 

    // output file 
    gt_theta_file = fstream("/home/hxt/Desktop/hxy-rotation/data/saved_gt_theta.txt", ios::out);
    gt_velocity_file = fstream("/home/hxt/Desktop/hxy-rotation/data/saved_gt_theta_velocity.txt", ios::out);
    est_velocity_file = fstream("/home/hxt/Desktop/hxy-rotation/data/ransac_velocity.txt", ios::out);

    // thread in background 
    // thread_view = new thread(&System::visualize, this);
    // thread_run = new thread(&System::Run, this);


}

System::~System()
{
    cout << "saving files " << endl;

    // delete thread_run; 
    cv::destroyAllWindows();
    gt_theta_file.close();
    gt_velocity_file.close();

    est_velocity_file.close(); 

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
    
    // gt data 
    // cv::undistortPoints(raw_event_points, undis_event_points, 
    //         camera.cameraMatrix, camera.distCoeffs, cv::noArray(), camera.cameraMatrix);
    
    // (0,0,0,0)
    cv::Mat undis = (cv::Mat_<double>(1,4) << 0, 0, 0 ,0 );
    cv::undistortPoints(raw_event_points, undis_event_points, 
            camera.cameraMatrix,undis , cv::noArray(), camera.cameraMatrix);
    
    
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
* \param cv_3D_surface_index store index (height, width, channel)
* \param cv_3D_surface_index_count store index count (height, width, count)
*/
cv::Mat System::getImageFromBundle(EventBundle& cur_event_bundle, const PlotOption option, bool is_mapping /*=false*/)
{

    // cout << "getImageFromBundle " << cur_event_bundle.coord.cols() << ", is_mapping "<< is_mapping << endl;
    // cout << "enter for interval " << "cols " << cur_event_bundle.isInner.rows()<< endl;

    cv::Mat image;
    // cv::Mat img_surface_index; // store time index 

    int max_count = 0;


    int width = camera.width, height = camera.height; 

    if(is_mapping)
    {
        width = camera.width_map; 
        height = camera.height_map; 
    }
    // cout << "  image size (h,w) = " << height << "," << width << endl;

    switch (option)
    {
    case PlotOption::U16C3_EVNET_IMAGE_COLOR:
        // cout << "  choosing U16C3_EVNET_IMAGE_COLOR" << endl;
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
        
            // cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x,y);

            image.at<cv::Vec3w>(point_temp) += eventBundle.polar[i] > 0 ? cv::Vec3w(0, 0, 1) : cv::Vec3w(1, 0, 0);
        }

        // cout << "image size"  << image.size() <<endl;
        break;
    
    case PlotOption::U16C1_EVNET_IMAGE:
        // cout << "enter case U16C1_EVNET_IMAGE" << endl;
        image = cv::Mat(height, width, CV_16UC1);
        image = cv::Scalar(0);

        for(int i=0; i<cur_event_bundle.coord.cols(); i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1]; 

            if(cur_event_bundle.isInner(i) < 1) continue;

            // if(x >= width  ||  x < 0 || y >= height || y < 0 ) 
            //     cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x,y);
            image.at<unsigned short>(point_temp) += 1;
        }
        break;
    
    case PlotOption::TIME_SURFACE:
        cout << "build time surface " << endl;
        cv_3D_surface_index.setTo(0); cv_3D_surface_index_count.setTo(0); 
        image = cv::Mat(height, width, CV_32FC1);
        image = cv::Scalar(0);

        for(int i=0; i<cur_event_bundle.size; i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1];

            if(cur_event_bundle.isInner(i) < 1) continue;

            if(x >= width  ||  x < 0 || y >= height || y < 0 ) 
                cout << "x, y" << x << "," << y << endl;

            image.at<float>(y,x) = eventBundle.time_delta(i)*1e3;  // only for visualization 

            cv_3D_surface_index.at<int>(y,x,cv_3D_surface_index_count.at<int>(y,x)) = i;
            cv_3D_surface_index_count.at<int>(y,x) += 1; 
            max_count = std::max(max_count,  cv_3D_surface_index_count.at<int>(y,x));

            // cout << eventBundle.time_delta(i)<< endl;
            // img_surface_index.at<unsigned short>(y,x) = i;
            // cout << eventBundle.time_delta(i) << endl;
        }

        cout << "max_count channels " << max_count << endl;
        // cout << "size " << eventBundle.time_delta.size() << endl; 
        // cout << "size " << cur_event_bundle.coord.cols() << endl; 
        break; 
    default:
        cout << "default choice " << endl;
        break;
    }

    // cout << "  success get image " << endl;
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
* \brief get_local_rotation_b2f using current eventbundle, return the rotation matrix from t2(end) to t1(start). 
* \param reverse from t1->t2. as intuision. 
*/
Eigen::Matrix3d System::get_local_rotation_b2f(bool inverse)
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

    // from t1->t2
    if(inverse) 
        return R2.transpose()*R1;
    
    // from t2->t1
    return R1.transpose()*R2;
}


/**
* \brief run in back ground, avoid to affect the ros call back function.
*/
void System::Run()
{
    
    /* update eventBundle */ 
    eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter]);      
    vec_vec_eventData_iter++;

    // check current eventsize or event interval 
    double time_interval = (eventBundle.last_tstamp-eventBundle.first_tstamp).toSec();
    if(time_interval < 0.01 || eventBundle.size < 500)
    {
        cout << "no enough interval or num: " <<time_interval << ", "<< eventBundle.size << endl;
        return; 
    }



    cout << "----processing event bundle------ size: " <<eventBundle.size  << 
        ", vec leave:" <<vec_vec_eventData.size() - vec_vec_eventData_iter << endl; 

    /* undistort events */ 
    undistortEvents();


    /* get local bundle sharper using gt*/ 
    if(using_gt)
    {
        if(vec_gt_poseData.size()< 10) gt_angleAxis.setConstant(0);
        else
        {
            Eigen::Matrix3d R_t1_t2 = get_local_rotation_b2f();
            Eigen::AngleAxisd ang_axis(R_t1_t2);
            double _delta_time = eventBundle.time_delta[eventBundle.time_delta.rows()-1]; 
            ang_axis.angle() /= _delta_time;  // get angular velocity
            gt_angleAxis = ang_axis.axis() * ang_axis.angle();

            // display gt
            getWarpedEventImage(ang_axis.axis() * ang_axis.angle(), 
                event_warpped_Bundle_gt).convertTo(curr_warpped_event_image, CV_32F);
        }
    }

    /* get local bundle sharper using self derived iteration CM method */ 
    // EstimateMotion_kim();
    // getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32F);

    /* get local bundle sharper using time residual */
    est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 
    // cv::waitKey(0);
    EstimateMotion_ransca_once(0.95, 1, 0.05);
    // cv::waitKey(0);
    // EstimateMotion_ransca_once(0.95, 1, 0.05);
    // cv::waitKey(0);
    // EstimateMotion_ransca_once(0.95, 1, 0.05);
    // cv::waitKey(0);
    // EstimateMotion_ransca_once(0.95, 1, 0.05);

    // EstimateMotion_ransca_once(0.9, 0.7, 0.05  );
    // cv::waitKey(0);
    // EstimateMotion_ransca_once(0.9, 1,   0.05);
    // cv::waitKey(0);
    // EstimateMotion_ransca_once(0.7, 1,   0.05);
    // cv::waitKey(0);
    // EstimateMotion_ransca_once(0.7, 1,   0.01);
    // cv::waitKey(0);


    getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32F);


    // save gt date 
    save_velocity();

    /* get global maps */ 
    // getMapImage();

    // visualize 
    visualize(); 

    // clear event bundle 
    // que_vec_eventData.pop();
    eventBundle.Clear();
    cout << "-------sucess run thread -------" << endl;

}



/**
* \brief save event velocity(t2->t1), add minus to convert it to t1->t2 .
*/
void System::save_velocity()
{
    // for velocity 
    double timestamp =  (eventBundle.last_tstamp - beginTS).toSec();
    double delta_time = (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec(); 

    double angle = (est_angleAxis * delta_time).norm();
    Eigen::AngleAxisd ag_pos =  Eigen::AngleAxisd(angle, - (est_angleAxis * delta_time) / angle);
    Eigen::Quaterniond q = Eigen::Quaterniond(ag_pos);

    // minus means from t1->t2. 
    Eigen::Vector3d euler_position = - toEulerAngles(q) / delta_time; // back to velocity
    est_velocity_file << timestamp <<" " << euler_position.transpose() << endl;
    // cout << "time "  << time << " " << euler_position.transpose()  <<  endl; 

}

/**
* \brief input evene vector from ros msg, according to time interval.
*/
void System::pushEventData(const std::vector<dvs_msgs::Event>& ros_vec_event)
{
    // que_vec_eventData.push(ros_vec_event); 
    vec_vec_eventData.push_back(ros_vec_event);
    // cout << "push to vec_vec_eventData " << endl;  
    
    Run(); 
}

void System::setBeginTime(ros::Time begin)
{
    beginTS = begin; 
}


/**
* \brief input evene vector from ros msg.
* \param[in] ImageData self defined imagedata.
*/
void System::pushimageData(const ImageData& imageData)
{

    // can be save in vector 

    // update current image 
    curr_imageData = imageData;  // copy construct 
    curr_raw_image = imageData.image.clone(); 

    // undistort image 
    cv::remap(curr_raw_image, curr_undis_image, undist_mesh_x, undist_mesh_y, cv::INTER_LINEAR );
}

void System::pushPoseData(const PoseData& poseData)
{

    // Eigen::Vector3d v_2 = poseData.quat.toRotationMatrix().eulerAngles(2,1,0);
    Eigen::Vector3d curr_pos = toEulerAngles(poseData.quat);

    int loop = 6; 
    if(vec_gt_poseData.size() > 12)
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

        Eigen::Vector3d velocity_zerobased(velocity(0),velocity(2),-velocity(1)); 

        // Eigen::Vector3d velocity(0,0,0); 
        // double delta_time = poseData.time_stamp - vec_gt_poseData.back().time_stamp;
        // Eigen::Vector3d theta_1 = toEulerAngles(poseData.quat);
        // Eigen::Vector3d theta_2 = toEulerAngles(vec_gt_poseData.back().quat);
        // Eigen::Vector3d delta_theta = theta_1 - theta_2; 
        // // cout << "theta 1: " << theta_1.transpose() <<"\ntheta 2: " << theta_2.transpose() << "\ndelta: " << delta_theta.transpose() << endl; 
        // velocity = delta_theta / delta_time;
    
        // cout << "  final velocity " << (velocity.array()/3.14*180).transpose() << endl;
        gt_velocity_file << poseData.time_stamp <<" " << velocity_zerobased.transpose() << endl;
        
        vec_angular_velocity.push_back(velocity_zerobased);
        vec_curr_time.push_back(poseData.time_stamp);
    }

    double time = (poseData.time_stamp_ros - beginTS).toSec();
    // cout << "time " <<time  << ", " << poseData.pose.transpose() << "," << curr_pos.transpose() << endl;
    gt_theta_file << time <<" " << curr_pos.transpose() << endl;

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
        cv::imshow("curr_warpped_event_image", curr_warpped_event_image * 0.3);
        cv::imshow("curr_warpped_event_image_gt", curr_warpped_event_image_gt);

        // cv::imshow("curr_map_image", curr_map_image);
        cv::imshow("hot_image_C3", hot_image_C3);

        cv::waitKey(10);

}


