
#include "system.hpp"


using namespace std;

System::System(const string& yaml)
{
    cout << "init system" << endl;

    cv::FileStorage fSettings(yaml, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        ROS_ERROR("counld not open file %s", yaml.c_str());
    }

    string calib_dir = fSettings["calib_dir"];
    camera = CameraPara(calib_dir);

    yaml_iter_num = fSettings["yaml_iter_num"];
    yaml_ts_start = fSettings["yaml_ts_start"];
    yaml_ts_end = fSettings["yaml_ts_end"];
    yaml_sample_count = fSettings["yaml_sample_count"];
    yaml_ceres_iter_thread = fSettings["yaml_ceres_iter_thread"];
    yaml_ceres_iter_num = fSettings["yaml_ceres_iter_num"];
    yaml_gaussian_size = fSettings["yaml_gaussian_size"];
    yaml_gaussian_size_sigma = fSettings["yaml_gaussian_size_sigma"];
    yaml_denoise_num = fSettings["yaml_denoise_num"];
    yaml_default_value_factor = fSettings["yaml_default_value_factor"];

    time_length_ = fSettings["yaml_time_length"]; 
    batch_length_ = fSettings["yaml_batch_length"]; 
    time_length_ /= float(1000); // convert to seconds
    outlier_ratio_ = fSettings["yaml_outlier_ratio"];  
    outlier_ratio_ /= float(100); // convert to float
    
    cout << "time_length_" << time_length_ << 
        ", batch_length_ init " << batch_length_ << 
        ", outlier_ratio_ init " << outlier_ratio_ << endl;

    // undistore data 
    undist_mesh_x, undist_mesh_y;  
    cv::initUndistortRectifyMap(camera.cameraMatrix, camera.distCoeffs, 
                cv::Mat::eye(3,3,CV_32FC1), camera.cameraMatrix, cv::Size(camera.width, camera.height), 
                CV_32FC1, undist_mesh_x, undist_mesh_y);

    using_gt = false;
    vec_vec_eventData_iter = 0;
    seq_count = 1;

    // ros msg 
    // vec_last_event_idx = 0;

    // visualize 
    // before processing 
    // cv::namedWindow("curr_raw_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_undis_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_event_image", cv::WINDOW_NORMAL);

    // after processing 
    cv::namedWindow("curr_undis_event_image", cv::WINDOW_NORMAL);
    cv::namedWindow("curr_warpped_event_image", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_warpped_event_image_gt", cv::WINDOW_NORMAL);
    // cv::namedWindow("curr_map_image", cv::WINDOW_NORMAL);
    //  cv::namedWindow("hot_image_C3", cv::WINDOW_NORMAL);
    //  cv::namedWindow("timesurface_early", cv::WINDOW_NORMAL);
    //  cv::namedWindow("timesurface_later", cv::WINDOW_NORMAL);
    //  cv::namedWindow("opti", cv::WINDOW_NORMAL);

    // before processing 
    curr_undis_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_raw_image = cv::Mat(camera.height,camera.width, CV_8U);
    curr_event_image = cv::Mat(camera.height,camera.width, CV_32FC3);
    hot_image_C1 = cv::Mat(camera.height,camera.width, CV_8UC1);
    hot_image_C3 = cv::Mat(camera.height,camera.width, CV_8UC3);

    // optimizeing 
    est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 
    // int dims[] = {180,240,20};   // row, col, channels            // useless
    // cv_3D_surface_index = cv::Mat(3, dims, CV_32S);               // useless
    // cv_3D_surface_index_count = cv::Mat(180, 240, CV_32S);        // useless
    
    // after processing 
    curr_undis_event_image = cv::Mat(camera.height,camera.width, CV_32F);
    curr_map_image = cv::Mat(camera.height_map,camera.width_map, CV_32F);
    curr_warpped_event_image = cv::Mat(camera.height,camera.width, CV_32F);
    curr_warpped_event_image_gt = cv::Mat(camera.height,camera.width, CV_32F); 

    // output file 
    output_dir = string(fSettings["output_dir"]);
    string output_path = output_dir + std::to_string(yaml_sample_count) + "_outlier" + std::to_string(int(outlier_ratio_*100)) + "_batchtime" + std::to_string(int(time_length_*1000)) + "ms"+
                  "_batchlength" + std::to_string(batch_length_) + "_timerange(0." + std::to_string(int(yaml_ts_start * 10)) + "-0." + std::to_string(int(yaml_ts_end * 10)) + ")" +
                  "_iter" + std::to_string(yaml_iter_num) + "_ceres" + std::to_string(yaml_ceres_iter_num) +
                  "_gaussan" + std::to_string(yaml_gaussian_size) + "_sigma" + std::to_string(int(yaml_gaussian_size_sigma)) + "." + std::to_string(int(yaml_gaussian_size_sigma * 10) % 10) +
                  "_denoise" + std::to_string(yaml_denoise_num) +
                  ".txt";
    cout << "open file " << output_path << endl; 


    if(!fstream(output_path, ios::in).is_open())
    {
        cout << "creating file " << endl;
        est_velocity_file = fstream(output_path, ios::out);
    }

    // est_velocity_file_quat = fstream("/home/hxy/Desktop/hxy-rotation/data/evo_data/ransac_velocity.txt", ios::out);


    // thread in background 
    // thread_view = new thread(&System::visualize, this);
    // thread_run = new thread(&System::Run, this);


    // init 
    total_evaluate_time = 0;
    total_visual_time = 0;
    total_undistort_time = 0;
    total_timesurface_time = 0;
    total_ceres_time = 0;
    total_eventbundle_time = 0;
    total_readevents_time = 0;
    total_warpevents_time = 0;

    est_angleAxis = Eigen::Vector3d(0.01,0.01,0.01);       // estimated anglar anxis from t2->t1.  = theta / delta_time 
    est_trans_velocity = Eigen::Vector3d(0.01,0.01,0.01);  // estimated anglar anxis from t2->t1, translation velocity, need mul by delta_time
    last_est_var << 0.01,0.01,0.01,0.01,0.01,0.01;

}

System::~System()
{
    cout << "saving files " << endl;

    // delete thread_run; 
    cv::destroyAllWindows();

    est_velocity_file.close(); 
    // est_velocity_file_quat.close();

}


/**
* \brief undistr events, and save the data to event_undis_Bundle (2d and 3d).
*/
void System::undistortEvents()
{
    int point_size = eventBundle.size;
    // cout << "------unditort events num:" << point_size <<  endl;
    // cout << "------undisotrt eventBundle cols " << eventBundle.coord.rows() << "," << eventBundle.coord.cols()  <<  endl;
    
    // vector<cv::Point2f> raw_event_points(point_size), undis_event_points(point_size);

    // for(size_t i=0; i<point_size; ++i)
    //     raw_event_points[i] = cv::Point2f(eventBundle.coord(0,i),eventBundle.coord(1,i));
    
    // using gt camera param 
    {
        // cv::undistortPoints(raw_event_points, undis_event_points, 
        //         camera.cameraMatrix, camera.distCoeffs, cv::noArray(), camera.cameraMatrix);
    }
        
    // using (0,0,0,0)  camera param  
    // {
    //     cv::Mat undis = (cv::Mat_<double>(1,4) << 0, 0, 0 ,0 );
    //     cv::undistortPoints(raw_event_points, undis_event_points, 
    //             camera.cameraMatrix,undis , cv::noArray(), camera.cameraMatrix);
    // }
    
    // convert points to cv_mat 
    // cv::Mat temp_mat = cv::Mat(undis_event_points); 
    // temp_mat = temp_mat.reshape(1,point_size); // channel 1, row = 2
    // cv::transpose(temp_mat, temp_mat);
    
    // // convert cv2eigen 
    // event_undis_Bundle.CopySize(eventBundle); 
    // cv::cv2eigen(temp_mat, event_undis_Bundle.coord); 
    

    event_undis_Bundle.CopySize(eventBundle); 
    event_undis_Bundle.coord = eventBundle.coord;
    // store 3d data
    // cout << event_undis_Bundle.coord.topLeftCorner(2,5) << endl;

    event_undis_Bundle.InverseProjection(camera.eg_cameraMatrix, eventBundle.coord_3d);  
    // event_undis_Bundle.coord_3d = eventBundle.coord_3d;
    
    // store inner 
    event_undis_Bundle.DiscriminateInner(camera.width, camera.height);

    getImageFromBundle(event_undis_Bundle, PlotOption::U16C3_EVNET_IMAGE_COLOR, false).convertTo(curr_undis_event_image, CV_32F);
    // cout << "success undistort events " << endl;
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
        
        for(int i = cur_event_bundle.coord.cols()-1; i>0; i--)
        // for(int i=0; i<cur_event_bundle.coord.cols(); i++)
        {
            // bgr
            int bgr = eventBundle.polar[i] ? 2 : 0; 
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1]; 

            // descriminate inner 
            if(cur_event_bundle.isInner(i) < 1)
                continue;

            if(x >= width  ||  x < 0 || y >= height|| y < 0 ) {
                cout << "x, y" << x << "," << y << endl;
                continue;
            }
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

            if(x >= width  ||  x < 0 || y >= height || y < 0 ) 
                cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x,y);
            image.at<unsigned short>(point_temp) += 1;
        }
        break;
    case PlotOption::U8C1_EVNET_IMAGE:
        // cout << "enter case U16C1_EVNET_IMAGE" << endl;
        image = cv::Mat(height, width, CV_8UC1);
        image = cv::Scalar(0);

        for(int i=0; i<cur_event_bundle.coord.cols(); i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1]; 

            if(cur_event_bundle.isInner(i) < 1) continue;

            // if(x >= width  ||  x < 0 || y >= height || y < 0 ) 
            //     cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x,y);
            image.at<unsigned char>(point_temp) += 1;
        }
        break;
    case PlotOption::TIME_SURFACE:
        // cout << "build time surface " << endl;
        // cv_3D_surface_index.setTo(0); cv_3D_surface_index_count.setTo(0); 
        image = cv::Mat(height, width, CV_32FC1);
        image = cv::Scalar(0);

        for(int i=0; i<cur_event_bundle.size; i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1];

            if(cur_event_bundle.isInner(i) < 1) continue;

            if(x >= width  ||  x < 0 || y >= height || y < 0 ) 
                cout << "x, y" << x << "," << y << endl;

            image.at<float>(y,x) = 0.1 - eventBundle.time_delta(i);  // only for visualization 

            // cv_3D_surface_index.at<int>(y,x,cv_3D_surface_index_count.at<int>(y,x)) = i;
            // cv_3D_surface_index_count.at<int>(y,x) += 1; 
            // max_count = std::max(max_count,  cv_3D_surface_index_count.at<int>(y,x));

            // cout << eventBundle.time_delta(i)<< endl;
            // img_surface_index.at<unsigned short>(y,x) = i;
            // cout << eventBundle.time_delta(i) << endl;
        }

        // cout << "max_count channels " << max_count << endl;
        // cout << "size " << eventBundle.time_delta.size() << endl; 
        // cout << "size " << cur_event_bundle.coord.cols() << endl; 
        break; 
    case PlotOption::F32C1_EVENT_COUNT: 
        image = cv::Mat(height, width, CV_32FC1);
        image = cv::Scalar(0);

        for(int i=0; i<cur_event_bundle.size; i++)
        {

            int x = std::floor(cur_event_bundle.coord.col(i)[0]);
            int y = std::floor(cur_event_bundle.coord.col(i)[1]);
            float dx = float(cur_event_bundle.coord.col(i)[0]) - float(x);
            float dy = float(cur_event_bundle.coord.col(i)[1]) - float(y);

            if(cur_event_bundle.isInner(i) < 1) continue;
            // if(x >= width-1  ||  x < 0 || y >= height-1 || y < 0 ) 
            //     cout << "x, y" << x << "," << y << endl;

            image.at<float>(y,x)     += (1-dx)*(1-dy);
            image.at<float>(y,x+1)   += (dx)*(1-dy);
            image.at<float>(y+1,x)   += (1-dx)*(dy);
            image.at<float>(y+1,x+1) += (dx)*(dy);
        }

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
    ros::Time t1, t2;

    // check current eventsize or event interval
    // double time_interval = (vec_vec_eventData[vec_vec_eventData_iter].back().ts - vec_vec_eventData[vec_vec_eventData_iter].front().ts).toSec();
    // if (time_interval < 0.0001 || vec_vec_eventData[vec_vec_eventData_iter].size() < 3000)
    // {
    //     // 确保有足够的基础数 得到可靠的速度，然后再判断重叠率 
    //     cout << "no enough interval or num: " << time_interval << ", " << vec_vec_eventData[vec_vec_eventData_iter].size() << endl;
    //     eventBundle.Clear();
    //     vec_vec_eventData_iter++;
    //     return;
    // }

    if (eventBundle.size < int(20e3) || eventBundle.time_delta.tail(1).value() < 0.01)
    {
        // 执行正常的一个流程 预计是1w
        /* update eventBundle */
        t1 = ros::Time::now();
        eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter], vec_vec_eventDepth[vec_vec_eventData_iter]);
        vec_vec_eventData_iter++;
        t2 = ros::Time::now();
        total_eventbundle_time += (t2 - t1).toSec();
        // cout << "----processing event bundle------ size: " <<eventBundle.size  <<
        // ", vec leave:" <<vec_vec_eventData.size() - vec_vec_eventData_iter << endl;
        last_merge_ratio_ = outlier_ratio_;
        // est_angleAxis.setZero();
        cout << "no enough interval or num: " << eventBundle.time_delta.tail(1) << ", " << eventBundle.size << endl;
    }
    else {
        /* undistort events */
        t1 = ros::Time::now();
        undistortEvents();
        t2 = ros::Time::now();
        total_undistort_time += (t2 - t1).toSec();
        // cout << "undistortEvents time " <<total_undistort_time<< ", " << (t2-t1).toSec() << endl;  // 0.00691187 s

        // 初始化eventbundle里面的速度 保证里面一定有1w个事件
        t1 = ros::Time::now();
        EstimateMotion_ransca_ceres(yaml_ts_start, yaml_ts_end, yaml_sample_count, yaml_iter_num);
        t2 = ros::Time::now();
        total_evaluate_time += (t2 - t1).toSec();
        // cout << "EstimateMotion_ransca_ceres time " <<total_evaluate_time<< ", " << (t2-t1).toSec() << endl;  // 0.00691187 s
        
        // t_threshold_ = eventBundle.time_delta.tail(0).value();

        // 测试 下一个batch是否在界内
        EventBundle latest_bundle, undis_latest_bundle;
        latest_bundle.Append(vec_vec_eventData[vec_vec_eventData_iter], vec_vec_eventDepth[vec_vec_eventData_iter], eventBundle.first_tstamp);
        undistortEvents(latest_bundle, undis_latest_bundle);

        double merge_ratio = GetInsideRatioSingle(undis_latest_bundle);
        // double merge_ratio = GetInsideRatioDouble(undis_latest_bundle);

        double cur_time_length = eventBundle.time_delta.tail(1).value();
        if (merge_ratio > 1.1 * last_merge_ratio_ || cur_time_length > time_length_ || eventBundle.size/1000 > batch_length_)
        {

            // finetune 不在界内，匀速假设不成立，则保留当前batch值，iter位置不变
            // int temp_iter = yaml_iter_num;
            // yaml_iter_num = yaml_iter_num_final;
            // t1 = ros::Time::now();
            // EstimateMotion_ransca_ceres(yaml_ts_start, yaml_ts_end, yaml_sample_count, yaml_iter_num);
            // t2 = ros::Time::now();
            // total_evaluate_time += (t2 - t1).toSec();
            // yaml_iter_num = temp_iter; // 再恢复过来 原本的迭代次数
            
            total_processing_events += eventBundle.size;

            save_velocity();
            if (merge_ratio > last_merge_ratio_) {
                invalid_merge_count++;
                cout << "匀速假设不成立, outiler上升，当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
            } else if (eventBundle.size / 1000 > batch_length_) {
                invalid_batch_count++;
                cout << "匀速假设不成立, eventBundle太大 size " << eventBundle.size / 1000 << "k" << ", batch_length_ " << batch_length_ << ", time " << cur_time_length << endl;
            }
            else {
                cout << "匀速假设不成立, 超时，当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
                invalid_time_count++;
            }
            eventBundle.Clear();
            last_merge_ratio_ = outlier_ratio_;

            // 界内界外共享这段函数，开始merge，并更新速度
            eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter], vec_vec_eventDepth[vec_vec_eventData_iter]);
            // t_threshold_ = eventBundle.time_delta.tail(0).value();
            vec_vec_eventData_iter++;
            undistortEvents();
            // EstimateMotion_ransca_ceres();
        } else {
            last_merge_ratio_ = merge_ratio;
            valid_merge_count++;
            cout << "merge, 匀速假设成立, 当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
   
            // 界内界外共享这段函数，开始merge，并更新速度 这时最低就有1.5w了
            eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter],  vec_vec_eventDepth[vec_vec_eventData_iter]);
            vec_vec_eventData_iter++;
            undistortEvents();
            t1 = ros::Time::now();
            EstimateMotion_ransca_ceres(yaml_ts_start, yaml_ts_end, yaml_sample_count, yaml_iter_num);
            t2 = ros::Time::now();
            total_evaluate_time += (t2 - t1).toSec();
       }        
    }
    visualize();
    cout << "invalid_merge_count " << invalid_merge_count << ", invalid time count " << invalid_time_count << 
        ", invalid batch count " << invalid_batch_count << ", valid merge count " << valid_merge_count << endl;
}

void System::undistortEvents(EventBundle &ebin, EventBundle &ebout) {
    ebout.CopySize(eventBundle); 
    ebout.coord = eventBundle.coord;
    ebout.time_delta = eventBundle.time_delta;
    // store 3d data
    // cout << event_undis_Bundle.coord.topLeftCorner(2,5) << endl;

    ebout.InverseProjection(camera.eg_cameraMatrix, eventBundle.coord_3d);  
    // event_undis_Bundle.coord_3d = eventBundle.coord_3d;
    
    // store inner 
    ebout.DiscriminateInner(camera.width, camera.height);
}

double System::GetInsideRatioSingle(EventBundle &evin)
{
    assert(!cv_early_timesurface_float_.empty());
    Eigen::Vector3d cur_ang_vel = est_angleAxis;
    Eigen::Vector3d cur_trans_vel = est_trans_velocity;
    EventBundle my_warpped_Bundle;
    // getWarpedEvent(evin, est_angleAxis, my_warpped_Bundle);
    getWarpedEventPoints(evin, my_warpped_Bundle, cur_ang_vel, cur_trans_vel, -1, false); 
    my_warpped_Bundle.Projection(camera.eg_cameraMatrix);
    my_warpped_Bundle.DiscriminateInner(camera.width, camera.height);
    // add more residual s

    cv::Mat temp_img(camera.height, camera.width, CV_8U);
    temp_img.setTo(0);
    
    int outlier = 0, outlier_bound = 0, outlier_time = 0;
    for (int loop_temp = 0; loop_temp < evin.size; loop_temp++)
    {
        int idx = loop_temp;
        int sampled_x = std::round(my_warpped_Bundle.coord.col(idx)[0]);
        int sampled_y = std::round(my_warpped_Bundle.coord.col(idx)[1]);
        if (my_warpped_Bundle.isInner[idx] < 1) // outlier
            outlier_bound++;
        else
        {
            // if (cv_earlier_timesurface_mask_.at<u_char>(sampled_y, sampled_x) == 0)
            //     outlier_time++;
            int count = 0;
            for (int j = -1; j < 2; j++)
                for (int k = -1; k < 2; k++)
                    count += (cv_early_timesurface_float_.at<float>(sampled_y + j, sampled_x + k) < eventBundle.time_delta(eventBundle.size / 2));
                    // count += (cv_earlier_timesurface_float_.at<float>(sampled_y + j, sampled_x + k) < t_threshold_);
            
            // if (count > 0)
            //     temp_img.at<u_char>(sampled_y, sampled_x) = 255;
            
            if (count < 1) 
                outlier_time++;
        }

        outlier = outlier_bound + outlier_time;
        // if (loop_temp % 3000 == 0)
        // {
        //     double ratio = 100 * outlier_time / float(evin.time_delta.size());
        //     cout << "0-" << loop_temp / float(evin.time_delta.size()) << " sample, " << evin.time_delta.size() << ", outlier bound " << outlier_bound << ", ourlier time " << outlier_time << ", outlier_time rate " << ratio << "%" << endl;
        // }
    }

    double ratio = outlier / float(evin.time_delta.size());
    cout << "0-1 sample " << evin.time_delta.size() << ", outlier " << outlier << ", rate " << ratio*100 << "%" << endl;


    // cv::imshow("tempimg", temp_img);
    // cv::waitKey(10);
    return ratio;
}


/**
* \brief save event velocity(t2->t1), add minus to convert it to t1->t2 .
*/
void System::save_velocity()
{
    // for velocity 
    double delta_time = (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec(); 

    // minus means from t1->t2. 
    // double angle = (est_angleAxis * delta_time).norm();
    // Eigen::AngleAxisd ag_pos =  Eigen::AngleAxisd(angle, (est_angleAxis * delta_time) / angle);
    // Eigen::Quaterniond q = Eigen::Quaterniond(ag_pos);
    // Eigen::Vector3d euler_position = toEulerAngles(q) / delta_time; // back to velocity
    
    // WARNING, you should use ros timestamps not double (cout for double is 6 valid numbers)
    // est_velocity_file << seq_count++ <<" " << eventBundle.first_tstamp << " " << eventBundle.last_tstamp << " " << euler_position.transpose() << endl;

    est_velocity_file << seq_count++ <<" " << eventBundle.first_tstamp << " " << 
                        eventBundle.last_tstamp << " " << est_angleAxis.transpose() << " " <<
                        est_trans_velocity.transpose() << " " << eventBundle.size << " " << eventBundle.size << endl;

}

/**
* \brief input evene vector from ros msg, according to time interval.
*/
void System::pushEventData(const std::vector<dvs_msgs::Event>& ros_vec_event)
{
    // que_vec_eventData.push(ros_vec_event); 
    vec_vec_eventData.push_back(ros_vec_event);
    // cout << " to vec_vec_eventData " << endl;  
    
    Run(); 
}

void System::pushEventData(const std::vector<dvs_msgs::Event>& ros_vec_event, const std::vector<double>& vec_depth)
{
    // que_vec_eventData.push(ros_vec_event); 
    vec_vec_eventData.push_back(ros_vec_event);
    // cout << " to vec_vec_eventData " << endl;  
    vec_vec_eventDepth.push_back(vec_depth);
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



/** useless
* \brief average 6 pose data from euler anglers to compute angular velocity.
*/
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
        // gt_velocity_file << poseData.time_stamp <<" " << velocity_zerobased.transpose() << endl;
        
        vec_angular_velocity.push_back(velocity_zerobased);
        vec_curr_time.push_back(poseData.time_stamp);
    }

    double time = (poseData.time_stamp_ros - beginTS).toSec();
    // cout << "time " <<time  << ", " << poseData.pose.transpose() << "," << curr_pos.transpose() << endl;
    // gt_theta_file << time <<" " << curr_pos.transpose() << endl;

    vec_gt_poseData.push_back(poseData);
    // cout << "push pose to system " << endl;
}


void System::visualize()
{
        // cout << "visualize" << endl; 
            
        // TODO update all images 

        // cv::imshow("curr_raw_image", curr_raw_image);
        // cv::imshow("curr_undis_image", curr_undis_image);
        // cv::imshow("curr_event_image", curr_event_image);

        // cout << "channels " << curr_undis_event_image.channels() << 
            // "types " << curr_undis_event_image.type() << endl;
        cv::imshow("curr_undis_event_image", curr_undis_event_image);

        // getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);
        cv::imshow("curr_warpped_event_image", curr_warpped_event_image);
        // cv::imshow("curr_warpped_event_image_gt", curr_warpped_event_image_gt);
        // cv::imshow("curr_map_image", curr_map_image);
        // cv::imshow("hot_image_C3", hot_image_C3);

        // output data
        // cv::normalize(curr_undis_event_image, curr_undis_event_image, 0,255, cv::NORM_MINMAX, CV_8U);
        // cv::normalize(curr_warpped_event_image, curr_warpped_event_image, 0,255, cv::NORM_MINMAX, CV_8U);
        // cv::threshold(curr_warpped_event_image, curr_warpped_event_image, 0.1, 255, CV_8U);
        // cv::threshold(curr_undis_event_image, curr_undis_event_image, 0.1, 255, CV_8U);
        // cv::imwrite(output_dir + std::to_string(seq_count) + "_undis.png", curr_undis_event_image);
        // cv::imwrite(output_dir + std::to_string(seq_count) + "_warp.png", curr_warpped_event_image);
        

        cv::waitKey(1);
}


