
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
    // cv::namedWindow("curr_raw_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_undis_image", cv::WINDOW_NORMAL);

    // cv::namedWindow("curr_event_image", cv::WINDOW_NORMAL);
    cv::namedWindow("curr_undis_event_image", cv::WINDOW_NORMAL);
    cv::namedWindow("curr_warpped_event_image", cv::WINDOW_NORMAL);

    // cv::namedWindow("curr_map_image", cv::WINDOW_NORMAL);


    curr_undis_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_raw_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_event_image = cv::Mat(camera.height,camera.width, CV_8UC3);
    curr_undis_event_image = cv::Mat(camera.height,camera.width, CV_8UC3);
    curr_map_image = cv::Mat(camera.height,camera.width, CV_8UC3);
    curr_warpped_event_image = cv::Mat(camera.height,camera.width, CV_8UC3);

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
    cout << "------undisotrt eventBundle cols " << eventBundle.coord.rows() << "," << eventBundle.coord.cols()  <<  endl;
    

    vector<cv::Point2f> raw_event_points(point_size), undis_event_points(point_size);

    for(size_t i=0; i<point_size; ++i)
        raw_event_points[i] = cv::Point2f(eventBundle.x[i],eventBundle.y[i]);
    
    // cv::undistortPoints(raw_event_points, undis_event_points, 
    //         camera.cameraMatrix, camera.distCoeffs, cv::noArray(), camera.cameraMatrix);
    cv::Mat temp_map = cv::Mat::eye(3,3, CV_32F);
    cv::undistortPoints(raw_event_points, undis_event_points, 
            temp_map, {0,0}, cv::noArray(), temp_map);
    
    // convert points to mat 
    cv::Mat temp_mat = cv::Mat(undis_event_points); 
    temp_mat = temp_mat.reshape(1,point_size); // channel 1, row = 2
    // cout << "points " << temp.rows << "," << temp.cols <<", "<< temp.channels() << endl;
    // cout << temp.rowRange(0,5) << endl;
    cv::transpose(temp_mat, temp_mat);
    // cout << "cv: points row, col, channel" << temp_mat.rows << "," << temp_mat.cols <<", "<< temp_mat.channels() << endl;
    // cout << temp_mat.colRange(0,5) << endl;
    
    // conver cv2eigen 
    // event_undis_Bundle.size = point_size; 
    // event_undis_Bundle.coord = Eigen::MatrixXf(2,point_size);
    event_undis_Bundle.CopySize(eventBundle); 
    cv::cv2eigen(temp_mat, event_undis_Bundle.coord); 

    // cout <<"eigen: row, col " << event_undis_Bundle.coord.rows() << ","<<event_undis_Bundle.coord.cols() << endl;
    // cout << event_undis_Bundle.coord.topLeftCorner(2,5) << endl;
    
    event_undis_Bundle.x.resize(point_size);
    event_undis_Bundle.y.resize(point_size);
    Eigen::VectorXf::Map(&event_undis_Bundle.x[0], point_size) = Eigen::VectorXf(event_undis_Bundle.coord.row(0));
    Eigen::VectorXf::Map(&event_undis_Bundle.y[0], point_size) = Eigen::VectorXf(event_undis_Bundle.coord.row(1));
    
    // store 3d data
    event_undis_Bundle.InverseProjection(camera.eg_cameraMatrix); 

    // vector<float> temp_vecx(event_undis_Bundle.x.begin(),event_undis_Bundle.x.begin()+5);
    // vector<float> temp_vecy(event_undis_Bundle.y.begin(),event_undis_Bundle.y.begin()+5);
    // cout <<"eigen vector: \n"; 
    // for(int i =0 ; i< 5; i++)
    //     cout << temp_vecx[i] <<"," << temp_vecy[i] << endl;
    
    // store inner 
    event_undis_Bundle.DiscriminateInner(camera.width, camera.height);

    // generate the event image 
    getImageFromBundle(eventBundle, curr_event_image);
    getImageFromBundle(event_undis_Bundle, curr_undis_event_image);

    // curr_event_image = cv::Scalar(0,0,0); // clear first 
    // for(int i=0; i<5; i++)
    // {
    //     // bgr
    //     // int bgr = eventBundle.polar[i] ? 2 : 0;
    //     int bgr = 1;  
    //     cout << "cols " <<i<<"," <<eventBundle.coord.col(i)[0] << ","<< eventBundle.coord.col(i)[1] << endl;
    //     int x = eventBundle.coord.col(i)[0];
    //     int y = eventBundle.coord.col(i)[1]; 
    //     cv::Point2i point_temp(x,y);
    //     curr_event_image.at<cv::Vec3b>(point_temp)[bgr] = 255;
    // }

}

void System::getImageFromBundle(EventBundle& cur_event_bundle, cv::Mat& image, const PlotOption& option)
{
    cout << "getImageFromBundle " << cur_event_bundle.coord.cols() << endl;
    image = cv::Scalar(0,0,0); // clear first 

    for(int i=0; i<cur_event_bundle.coord.cols(); i++)
    {
        // bgr
        int bgr = eventBundle.polar[i] ? 2 : 0; 
        int x = cur_event_bundle.coord.col(i)[0];
        int y = cur_event_bundle.coord.col(i)[1]; 

        // TODO make inner to remove these points 
        // if(x > 239) 
        //     cout << "x out of range 240 " << i << "," << x <<","<< y << endl;
        // if(x < 0) 
        //     cout << "x out of range 0 " << i << "," << x <<","<< y << endl;
        

        x = x > 239 ? 239 : x; 
        y = y > 179 ? 179 : y; 
        
        cv::Point2i point_temp(x,y);
        image.at<cv::Vec3b>(point_temp)[bgr] = 255;
    }
    // cout << "success" << endl;
}


/**
* \brief input evene vector from ros msg.
* \param[in] eventData event array of dvs_msg::Event .
*/
void System::pushEventData(EventData& eventData)
{
    // save in vector eventData and event bundle (local events)
    vec_eventData.push_back(eventData);  // in system 
    eventBundle.Append(eventData);       // in event bundle 

    // check the time interval is match 
    double time_inteval = (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec();

    if(time_inteval > delta_time || eventBundle.x.size() > max_store_count)
    {
        cout << "----processing event bundle" << endl; 

        // undistort events 
        undistortEvents();

        // estimat current motion 

        if(vec_angular_velocity.empty())
        {
            cout << "no engough pose" << endl;
            return;
        }
        else
        {
            cout << "engough pose" << endl;
        }


        // find velocty according to time stamps 

        Eigen::Vector3f cur_ang_vel(0,0,0); 
        size_t temp_size = vec_angular_velocity.size()>50 ? 50 : vec_angular_velocity.size();
            double interval = 1e5; 
            int target_pos = 0;
            for(size_t i =0; i< temp_size; i++)
            {
                if(abs(target_pos - eventBundle.abs_tstamp))
                {
                    target_pos = i;
                    interval = abs(vec_curr_time[i] - eventBundle.abs_tstamp);
                }
            }
        cur_ang_vel = vec_angular_velocity[target_pos];

        cout << "event time " << eventBundle.abs_tstamp <<  "pose time "<< vec_curr_time[target_pos]<<endl;
        cout << "choose vel" << cur_ang_vel.transpose() << endl;


        getWarpedEventImage(cur_ang_vel);
        // getWarpedEventPoints(cur_ang_vel);


        // show event image 

        visualize();     
        // clear event bundle 
        eventBundle.Clear();
    }

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
    
        cout << "  final velocity " << (velocity.array()/3.14*180).transpose() << endl;
        gt_velocity_file << poseData.time_stamp <<" " << velocity.transpose() << endl;
        
        vec_angular_velocity.push_back(velocity.cast<float>() );
        vec_curr_time.push_back(poseData.time_stamp);

        gt_theta_file << (*(vec_gt_poseData.end()-loop)).time_stamp <<" " << curr_pos.transpose() << endl;
    }


    vec_gt_poseData.push_back(poseData);

}


void System::visualize()
{

    // TODO update all images 

    // cv::imshow("curr_raw_image", curr_raw_image);
    // cv::imshow("curr_undis_image", curr_undis_image);
    // cv::imshow("curr_event_image", curr_event_image);
    cv::imshow("curr_undis_event_image", curr_undis_event_image);
    cv::imshow("curr_warpped_event_image", curr_warpped_event_image);
    // cv::imshow("map_image", curr_map_image);

    cv::waitKey(100);

}


