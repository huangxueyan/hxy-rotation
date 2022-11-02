
#include "system.hpp"

using namespace std;

System::System(const string &yaml)
{
    cout << "init system" << endl;

    cv::FileStorage fSettings(yaml, cv::FileStorage::READ);
    if (!fSettings.isOpened())
    {
        ROS_ERROR("counld not open file %s", yaml.c_str());
    }

    yaml_iter_num = fSettings["yaml_iter_num"];
    yaml_iter_num_final = fSettings["yaml_iter_num_final"];
    yaml_ts_start = fSettings["yaml_ts_start"];
    yaml_ts_end = fSettings["yaml_ts_end"];
    yaml_sample_count = fSettings["yaml_sample_count"];
    yaml_ceres_iter_num = fSettings["yaml_ceres_iter_num"];
    yaml_gaussian_size = fSettings["yaml_gaussian_size"];
    yaml_gaussian_size_sigma = fSettings["yaml_gaussian_size_sigma"];
    yaml_denoise_num = fSettings["yaml_denoise_num"];
    yaml_default_value_factor = fSettings["yaml_default_value_factor"];
    yaml_ceres_iter_thread = fSettings["yaml_ceres_iter_thread"];
    yaml_ros_starttime = fSettings["yaml_ros_starttime"];
    
    
    time_length_ = fSettings["yaml_time_length"]; 
    batch_length_ = fSettings["yaml_batch_length"]; 
    time_length_ /= float(1000); // convert to seconds
    outlier_ratio_ = fSettings["yaml_outlier_ratio"];  
    outlier_ratio_ /= float(100); // convert to float
    // undistore data for map
    // cv::initUndistortRectifyMap(camera.cameraMatrix, camera.distCoeffs,
    //             cv::Mat::eye(3,3,CV_32FC1), camera.cameraMatrix, cv::Size(camera.width, camera.height),
    //             CV_32FC1, undist_mesh_x, undist_mesh_y);

    { // init undistort map for events
        std::vector<cv::Point2f> distort_points(camera.width * camera.height), undistort_points;

        for (int y = 0; y < camera.height; ++y)
        {
            const int yy = y * camera.width;
            for (int x = 0; x < camera.width; ++x)
            {
                distort_points[yy + x] = cv::Point2f(x, y);
            }
        }

        cv::undistortPoints(distort_points, undistort_points,
                            camera.cameraMatrix, camera.distCoeffs, cv::noArray(), camera.cameraMatrix);

        undist_mesh_x = cv::Mat(camera.height, camera.width, CV_32FC1, cv::Scalar(0));
        undist_mesh_y = cv::Mat(camera.height, camera.width, CV_32FC1, cv::Scalar(0));
        for (int y = 0; y < camera.height; ++y)
        {
            const int yy = y * camera.width;
            for (int x = 0; x < camera.width; ++x)
            {
                undist_mesh_x.at<float>(y, x) = undistort_points[yy + x].x;
                undist_mesh_y.at<float>(y, x) = undistort_points[yy + x].y;
            }
        }
    }

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
    curr_undis_image = cv::Mat(camera.height, camera.width, CV_8U);
    curr_raw_image = cv::Mat(camera.height, camera.width, CV_8U);
    curr_event_image = cv::Mat(camera.height, camera.width, CV_32FC3);
    hot_image_C1 = cv::Mat(camera.height, camera.width, CV_8UC1);
    hot_image_C3 = cv::Mat(camera.height, camera.width, CV_8UC3);

    // optimizeing
    est_angleAxis = Eigen::Vector3d(0, 0, 0);       // set to 0.
    int dims[] = {camera.height, camera.width, 20}; // row, col, channels
    cv_3D_surface_index = cv::Mat(3, dims, CV_32S);
    cv_3D_surface_index_count = cv::Mat(camera.height, camera.width, CV_32S);

    // after processing
    curr_undis_event_image = cv::Mat(camera.height, camera.width, CV_32F);
    curr_map_image = cv::Mat(camera.height_map, camera.width_map, CV_32F);
    curr_warpped_event_image = cv::Mat(camera.height, camera.width, CV_32F);
    curr_warpped_event_image_gt = cv::Mat(camera.height, camera.width, CV_32F);

    // output file
    string output_dir = fSettings["output_dir"];
    output_dir += std::to_string(yaml_sample_count) + "_outlier" + std::to_string(int(outlier_ratio_*100)) + "_batchtime" + std::to_string(int(time_length_*1000)) + "ms"+
                  "_batchlength" + std::to_string(batch_length_) + "_timerange(0." + std::to_string(int(yaml_ts_start * 10)) + "-0." + std::to_string(int(yaml_ts_end * 10)) + ")" +
                  "_iter" + std::to_string(yaml_iter_num) + "_ceres" + std::to_string(yaml_ceres_iter_num) +
                  "_gaussan" + std::to_string(yaml_gaussian_size) + "_sigma" + std::to_string(int(yaml_gaussian_size_sigma)) + "." + std::to_string(int(yaml_gaussian_size_sigma * 10) % 10) +
                  "_denoise" + std::to_string(yaml_denoise_num) +
                  ".txt";
    cout << "open file " << output_dir << endl;

    if (!fstream(output_dir, ios::in).is_open())
    {
        cout << "creating file " << endl;
        est_velocity_file = fstream(output_dir, ios::out);
    }

    // est_velocity_file_quat = fstream("/home/hxy/Desktop/hxy-rotation/data/evo_data/ransac_velocity.txt", ios::out);

    // thread in background
    // thread_view = new thread(&System::visualize, this);
    // thread_run = new thread(&System::Run, this);

    est_angleAxis = Eigen::Vector3d(0.01, 0.01, 0.01); // estimated anglar anxis from t2->t1.  = theta / delta_time
    last_est_var << 0.01, 0.01, 0.01;

    total_evaluate_time = 0;
    total_visual_time = 0;
    total_undistort_time = 0;
    total_timesurface_time = 0;
    total_ceres_time = 0;
    total_eventbundle_time = 0;
    total_readevents_time = 0;
    total_warpevents_time = 0;
    // cout << "COUNT " << seq_count <<", last est " << last_est_var << endl;
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
void System::undistortEvents(EventBundle &ebin, EventBundle &ebout)
{
    int point_size = ebin.size;
    // cout << "------unditort events num:" << point_size <<  endl;
    // cout << "------undisotrt eventBundle cols " << eventBundle.coord.rows() << "," << eventBundle.coord.cols()  <<  endl;

    ebout.CopySize(ebin);
    for (size_t i = 0; i < point_size; ++i)
    {
        int x = int(ebin.coord(0, i)), y = int(ebin.coord(1, i));
        ebout.coord(0, i) = undist_mesh_x.at<float>(y, x);
        ebout.coord(1, i) = undist_mesh_y.at<float>(y, x);
    }

    // store 3d data
    ebout.InverseProjection(camera.eg_cameraMatrix);

    // store inner
    ebout.DiscriminateInner(camera.width, camera.height);
    ebout.time_delta = ebin.time_delta;
}

/**
 * \brief undistr events, and save the data to event_undis_Bundle (2d and 3d).
 */
void System::undistortEvents()
{
    int point_size = eventBundle.size;
    // cout << "------unditort events num:" << point_size <<  endl;
    // cout << "------undisotrt eventBundle cols " << eventBundle.coord.rows() << "," << eventBundle.coord.cols()  <<  endl;

    {
        event_undis_Bundle.CopySize(eventBundle);
        for (size_t i = 0; i < point_size; ++i)
        {
            int x = int(eventBundle.coord(0, i)), y = int(eventBundle.coord(1, i));
            event_undis_Bundle.coord(0, i) = undist_mesh_x.at<float>(y, x);
            event_undis_Bundle.coord(1, i) = undist_mesh_y.at<float>(y, x);

            // if(i<5)
            //     cout << "undist 2 " << undist_mesh_x.at<float>(y, x) << "," << undist_mesh_y.at<float>(y, x) <<endl;
        }
    }
    // t3 = ros::Time::now();
    // cout <<"undistor 1 " << (t2-t1).toSec() <<" undistort2 " << (t3-t2).toSec()<< endl;

    // store 3d data
    event_undis_Bundle.InverseProjection(camera.eg_cameraMatrix);

    // store inner
    event_undis_Bundle.DiscriminateInner(camera.width, camera.height);

    getImageFromBundle(event_undis_Bundle, PlotOption::U16C3_EVNET_IMAGE_COLOR).convertTo(curr_undis_event_image, CV_32F);
    // cout << "success undistort events " << endl;
}

/**
 * \brief Constructor.
 * \param is_mapping means using K_map image size.
 * \param cv_3D_surface_index store index (height, width, channel)
 * \param cv_3D_surface_index_count store index count (height, width, count)
 */
cv::Mat System::getImageFromBundle(EventBundle &cur_event_bundle, const PlotOption option, float timerange)
{

    // cout << "getImageFromBundle " << cur_event_bundle.coord.cols() << ", is_mapping "<< is_mapping << endl;
    // cout << "enter for interval " << "cols " << cur_event_bundle.isInner.rows()<< endl;

    cv::Mat image;
    // cv::Mat img_surface_index; // store time index

    int max_count = 0;

    int width = camera.width, height = camera.height;

    // cout << "  image size (h,w) = " << height << "," << width << endl;

    switch (option)
    {
    case PlotOption::U16C3_EVNET_IMAGE_COLOR:
        // cout << "  choosing U16C3_EVNET_IMAGE_COLOR" << endl;
        image = cv::Mat(height, width, CV_16UC3);
        image = cv::Scalar(0, 0, 0); // clear first

        // #pragma omp parallel for
        for (int i = cur_event_bundle.coord.cols() - 1; i > 0; i--)
        // for(int i=0; i<cur_event_bundle.coord.cols(); i++)
        {
            // bgr
            int bgr = eventBundle.polar[i] ? 2 : 0;
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1];

            // descriminate inner
            if (cur_event_bundle.isInner(i) < 1)
                continue;

            if (x >= width || x < 0 || y >= height || y < 0)
                cout << "x, y" << x << "," << y << endl;

            // cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x, y);

            image.at<cv::Vec3w>(point_temp) += eventBundle.polar[i] > 0 ? cv::Vec3w(0, 0, 1) : cv::Vec3w(1, 0, 0);
        }

        // cout << "image size"  << image.size() <<endl;
        break;

    case PlotOption::U16C1_EVNET_IMAGE:
        // cout << "enter case U16C1_EVNET_IMAGE" << endl;
        image = cv::Mat(height, width, CV_16UC1);
        image = cv::Scalar(0);

        for (int i = 0; i < cur_event_bundle.coord.cols(); i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1];

            if (cur_event_bundle.isInner(i) < 1)
                continue;

            // if(x >= width  ||  x < 0 || y >= height || y < 0 )
            //     cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x, y);
            image.at<unsigned short>(point_temp) += 1;
        }
        break;
    case PlotOption::U8C1_EVNET_IMAGE:
        // cout << "enter case U16C1_EVNET_IMAGE" << endl;
        image = cv::Mat(height, width, CV_8UC1);
        image = cv::Scalar(0);

        for (int i = 0; i < cur_event_bundle.coord.cols(); i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1];

            if (cur_event_bundle.isInner(i) < 1)
                continue;

            // if(x >= width  ||  x < 0 || y >= height || y < 0 )
            //     cout << "x, y" << x << "," << y << endl;

            cv::Point2i point_temp(x, y);
            image.at<unsigned char>(point_temp) += 1;
        }
        break;
    case PlotOption::TIME_SURFACE:
        // cout << "build time surface " << endl;
        // cv_3D_surface_index.setTo(0); cv_3D_surface_index_count.setTo(0);
        image = cv::Mat(height, width, CV_32FC1);
        image = cv::Scalar(0);

        for (int i = 0; i < cur_event_bundle.size; i++)
        {
            int x = cur_event_bundle.coord.col(i)[0];
            int y = cur_event_bundle.coord.col(i)[1];

            if (cur_event_bundle.isInner(i) < 1)
                continue;

            if (x >= width || x < 0 || y >= height || y < 0)
                cout << "x, y" << x << "," << y << endl;

            image.at<float>(y, x) = 0.1 - eventBundle.time_delta(i); // only for visualization

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

        for (int i = 0; i < cur_event_bundle.size; i++)
        {

            int x = std::floor(cur_event_bundle.coord.col(i)[0]);
            int y = std::floor(cur_event_bundle.coord.col(i)[1]);
            float dx = float(cur_event_bundle.coord.col(i)[0]) - float(x);
            float dy = float(cur_event_bundle.coord.col(i)[1]) - float(y);

            if (cur_event_bundle.isInner(i) < 1)
                continue;
            // if(x >= width-1  ||  x < 0 || y >= height-1 || y < 0 )
            //     cout << "x, y" << x << "," << y << endl;

            image.at<float>(y, x) += (1 - dx) * (1 - dy);
            image.at<float>(y, x + 1) += (dx) * (1 - dy);
            image.at<float>(y + 1, x) += (1 - dx) * (dy);
            image.at<float>(y + 1, x + 1) += (dx) * (dy);
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

    return R1.transpose() * R2;
}

/**
 * \brief get_local_rotation_b2f using current eventbundle, return the rotation matrix from t2(end) to t1(start).
 * \param reverse from t1->t2. as intuision.
 */
Eigen::Matrix3d System::get_local_rotation_b2f(bool inverse)
{
    int target1_pos, target2_pos;
    size_t start_pos = vec_gt_poseData.size() > 50 ? vec_gt_poseData.size() - 50 : 0;

    double interval_1 = 1e5, interval_2 = 1e5;

    for (size_t i = start_pos; i < vec_gt_poseData.size(); i++)
    {
        // cout << "ros time " <<  std::to_string(vec_gt_poseData[i].time_stamp_ros.toSec()) << endl;
        if (abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.first_tstamp).toSec()) < interval_1)
        {
            target1_pos = i;
            interval_1 = abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.first_tstamp).toSec());
        }

        if (abs((vec_gt_poseData[i].time_stamp_ros - eventBundle.last_tstamp).toSec()) < interval_2)
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
    if (inverse)
        return R2.transpose() * R1;

    // from t2->t1
    return R1.transpose() * R2;
}


double System::GetInsideRatioDouble(EventBundle &evin)
{
    // todo using early mask and later mask 
    EventBundle my_warpped_Bundle_early, my_warpped_Bundle_later;
    getWarpedEventImage(evin, est_angleAxis, my_warpped_Bundle_early);
    getWarpedEventImage(evin, est_angleAxis, my_warpped_Bundle_later, true); // backwarp
    // add more residual s

    cv::Mat temp_img(camera.height, camera.width, CV_8U);
    temp_img.setTo(0);
    
    int outlier = 0, outlier_bound = 0, outlier_time = 0;
    for (int loop_temp = 0; loop_temp < evin.size; loop_temp++)
    {
        int idx = loop_temp;
        int sampled_x_early = std::round(my_warpped_Bundle_early.coord.col(idx)[0]);
        int sampled_y_early = std::round(my_warpped_Bundle_early.coord.col(idx)[1]);
        int sampled_x_later = std::round(my_warpped_Bundle_later.coord.col(idx)[0]);
        int sampled_y_later = std::round(my_warpped_Bundle_later.coord.col(idx)[1]);

        if (my_warpped_Bundle_early.isInner[idx] < 1 || my_warpped_Bundle_later.isInner[idx] < 1) // outlier
            outlier_bound++;
        else
        {
            // if (cv_earlier_timesurface_mask_.at<u_char>(sampled_y, sampled_x) == 0)
            //     outlier_time++;
            int count_early = 0;
            for (int j = -1; j < 2; j++)
                for (int k = -1; k < 2; k++)
                    count_early += (cv_early_timesurface_float_.at<float>(sampled_y_early + j, sampled_x_early + k) < eventBundle.time_delta(eventBundle.size / 2));
                    // count += (cv_earlier_timesurface_float_.at<float>(sampled_y + j, sampled_x + k) < t_threshold_);
            
            int count_later = 0;
            for (int j = -1; j < 2; j++)
                for (int k = -1; k < 2; k++)
                    count_later += (cv_later_timesurface_float_.at<float>(sampled_y_later + j, sampled_x_later + k) > eventBundle.time_delta(eventBundle.size / 2));
                    // count += (cv_earlier_timesurface_float_.at<float>(sampled_y + j, sampled_x + k) < t_threshold_);
            
            if (count_early > 0 && count_later > 0)
                temp_img.at<u_char>(sampled_y_early, sampled_x_early) = 255;
            
            if (count_early < 1 || count_later < 1) 
                outlier_time++;
        }

        outlier = outlier_bound + outlier_time;
        if (loop_temp % 3000 == 0)
        {
            double ratio = 100 * outlier_time / float(evin.time_delta.size());
            cout << "0-" << loop_temp / float(evin.time_delta.size()) << " sample, " << evin.time_delta.size() << ", outlier bound " << outlier_bound << ", ourlier time " << outlier_time << ", outlier_time rate " << ratio << "%" << endl;
        }
    }

    double ratio = outlier / float(evin.time_delta.size());
    cout << "0-1 sample " << evin.time_delta.size() << ", outlier " << outlier << ", rate " << ratio*100 << "%" << endl;


    cv::imshow("new apped warp img", temp_img);
    // cv::imshow("mask", cv_earlier_timesurface_mask_);
    cv::waitKey(10);
    return ratio;
}

double System::GetInsideRatioSingle(EventBundle &evin)
{

    EventBundle my_warpped_Bundle;
    getWarpedEventImage(evin, est_angleAxis, my_warpped_Bundle);
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
            
            if (count > 0)
                temp_img.at<u_char>(sampled_y, sampled_x) = 255;
            
            if (count < 1) 
                outlier_time++;
        }

        outlier = outlier_bound + outlier_time;
        if (loop_temp % 3000 == 0)
        {
            double ratio = 100 * outlier_time / float(evin.time_delta.size());
            cout << "0-" << loop_temp / float(evin.time_delta.size()) << " sample, " << evin.time_delta.size() << ", outlier bound " << outlier_bound << ", ourlier time " << outlier_time << ", outlier_time rate " << ratio << "%" << endl;
        }
    }

    double ratio = outlier / float(evin.time_delta.size());
    cout << "0-1 sample " << evin.time_delta.size() << ", outlier " << outlier << ", rate " << ratio*100 << "%" << endl;


    cv::imshow("tempimg", temp_img);
    cv::waitKey(10);
    return ratio;
}

/**
 * \brief run in back ground, avoid to affect the ros call back function.
 */
void System::Run()
{
    ros::Time t1, t2;

    // check current eventsize or event interval
    double time_interval = (vec_vec_eventData[vec_vec_eventData_iter].back().ts - vec_vec_eventData[vec_vec_eventData_iter].front().ts).toSec();
    if (time_interval < 0.0001 || vec_vec_eventData[vec_vec_eventData_iter].size() < 3000)
    {
        cout << "no enough interval or num: " << time_interval << ", " << vec_vec_eventData[vec_vec_eventData_iter].size() << endl;
        eventBundle.Clear();
        vec_vec_eventData_iter++;
        return;
    }

    if (eventBundle.size < int(6e3) || eventBundle.time_delta.tail(1).value() < 0.01)
    {
        // 执行正常的一个流程 预计是1w
        /* update eventBundle */
        t1 = ros::Time::now();
        eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter]);
        vec_vec_eventData_iter++;
        t2 = ros::Time::now();
        total_eventbundle_time += (t2 - t1).toSec();
        // cout << "----processing event bundle------ size: " <<eventBundle.size  <<
        // ", vec leave:" <<vec_vec_eventData.size() - vec_vec_eventData_iter << endl;
        last_merge_ratio_ = outlier_ratio_;
        est_angleAxis.setZero();
    }
    else {
        /* undistort events */
        t1 = ros::Time::now();
        undistortEvents();
        t2 = ros::Time::now();
        total_undistort_time += (t2 - t1).toSec();
        // cout << "undistortEvents time " <<total_undistort_time<< ", " << (t2-t1).toSec() << endl;  // 0.00691187 s

        // 初始化eventbundle里面的速度 保证里面一定有1w个事件
        EstimateMotion_ransca_ceres();
        // t_threshold_ = eventBundle.time_delta.tail(0).value();

        // 测试 下一个batch是否在界内
        EventBundle latest_bundle, undis_latest_bundle;
        latest_bundle.Append(vec_vec_eventData[vec_vec_eventData_iter], eventBundle.first_tstamp);
        undistortEvents(latest_bundle, undis_latest_bundle);

        double merge_ratio = GetInsideRatioSingle(undis_latest_bundle);
        // double merge_ratio = GetInsideRatioDouble(undis_latest_bundle);

        double cur_time_length = eventBundle.time_delta.tail(1).value();
        if (merge_ratio > 1.1 * last_merge_ratio_ || cur_time_length > time_length_ || eventBundle.size/1000 > batch_length_)
        {
            // 不在界内，匀速假设不成立，则保留当前batch值，iter位置不变
            int temp_iter = yaml_iter_num;
            yaml_iter_num = yaml_iter_num_final;
            EstimateMotion_ransca_ceres(); // 自己全体再执行一次
            yaml_iter_num = temp_iter; // 再恢复过来

            save_velocity();
            if (merge_ratio > last_merge_ratio_) {
                invalid_merge_count++;
                cout << "匀速假设不成立, outiler上升，当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
            } else if (eventBundle.size > batch_length_) {
                invalid_batch_count++;
                cout << "匀速假设不成立, outiler上升，当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
            }
            else {
                cout << "匀速假设不成立, 超时，当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
                invalid_time_count++;
            }
            eventBundle.Clear();
            last_merge_ratio_ = outlier_ratio_;

            // 界内界外共享这段函数，开始merge，并更新速度
            eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter]);
            // t_threshold_ = eventBundle.time_delta.tail(0).value();
            vec_vec_eventData_iter++;
            undistortEvents();
            EstimateMotion_ransca_ceres();
        } else {
            last_merge_ratio_ = merge_ratio;
            valid_merge_count++;
            cout << "merge, 匀速假设成立, 当前 size " << eventBundle.size / 1000 << "k" << ", merge_ratio " << merge_ratio << ", time " << cur_time_length << endl;
   
            // 界内界外共享这段函数，开始merge，并更新速度 这时最低就有1.5w了
            eventBundle.Append(vec_vec_eventData[vec_vec_eventData_iter]);
            vec_vec_eventData_iter++;
            undistortEvents();
            EstimateMotion_ransca_ceres();
       }        
    }
    visualize();
    cout << "invalid_merge_count " << invalid_merge_count << ", invalid time count " << invalid_time_count << 
        ", invalid batch count " << invalid_batch_count << ", valid merge count " << valid_merge_count << endl;
}

/**
 * \brief save event velocity(t2->t1), add minus to convert it to t1->t2 .
 */
void System::save_velocity()
{
    // for velocity
    // double delta_time = (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec();
    // minus means from t1->t2.
    // double angle = (est_angleAxis * delta_time).norm();
    // Eigen::AngleAxisd ag_pos =  Eigen::AngleAxisd(angle, (est_angleAxis * delta_time) / angle);
    // Eigen::Quaterniond q = Eigen::Quaterniond(ag_pos);
    // Eigen::Vector3d euler_position = toEulerAngles(q) / delta_time; // back to velocity

    // WARNING, you should use ros timestamps not double (cout for double is 6 valid numbers)
    // est_velocity_file << seq_count++ <<" " << eventBundle.first_tstamp << " " << eventBundle.last_tstamp << " " << euler_position.transpose() << endl;

    // est_velocity_file << seq_count++ <<" " << eventBundle.first_tstamp << " " << eventBundle.last_tstamp << " " << est_angleAxis.transpose() << endl;
    est_velocity_file << seq_count++ << " "
                      << eventBundle.first_tstamp.toSec() - yaml_ros_starttime << " "
                      << eventBundle.last_tstamp.toSec() - yaml_ros_starttime << " " << est_angleAxis.transpose() <<" " <<eventBundle.size/1000 << endl;
}

/**
 * \brief input evene vector from ros msg, according to time interval.
 */
void System::pushEventData(const std::vector<dvs_msgs::Event> &ros_vec_event)
{
    // que_vec_eventData.push(ros_vec_event);
    ros::Time t1 = ros::Time::now();
    vec_vec_eventData.push_back(ros_vec_event);
    // cout << " to vec_vec_eventData " << endl;

    total_readevents_time += (ros::Time::now() - t1).toSec();

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
void System::pushimageData(const ImageData &imageData)
{

    // can be save in vector

    // update current image
    curr_imageData = imageData; // copy construct
    curr_raw_image = imageData.image.clone();

    // undistort image
    cv::remap(curr_raw_image, curr_undis_image, undist_mesh_x, undist_mesh_y, cv::INTER_LINEAR);
}

/** useless
 * \brief average 6 pose data from euler anglers to compute angular velocity.
 */
void System::pushPoseData(const PoseData &poseData)
{

    // Eigen::Vector3d v_2 = poseData.quat.toRotationMatrix().eulerAngles(2,1,0);
    Eigen::Vector3d curr_pos = toEulerAngles(poseData.quat);

    int loop = 6;
    if (vec_gt_poseData.size() > 12)
    {
        // [* * * target * * *] to get target velocity.
        vector<PoseData>::iterator it = vec_gt_poseData.end() - loop - 1;

        Eigen::Vector3d velocity(0, 0, 0);

        for (int k = 1; k <= loop; ++k)
        {
            // FIXME rpy: zyx, so v_1=(theta_z,y,x)
            // Eigen::Vector3d v_1 = (*it).quat.toRotationMatrix().eulerAngles(2,1,0);
            // Eigen::Vector3d v_2 = (*(it-loop)).quat.toRotationMatrix().eulerAngles(2,1,0);

            Eigen::Vector3d v_1 = toEulerAngles((*(it - loop - 1 + k)).quat);
            Eigen::Vector3d v_2 = toEulerAngles((*(it + k)).quat);

            double delta_time = (*(it + k)).time_stamp - (*(it - loop - 1 + k)).time_stamp;

            Eigen::Vector3d delta_theta = v_2 - v_1;
            velocity += delta_theta / delta_time;

            // cout<< "loop " << k << " delta_t: " <<delta_time
            //     << ", delta_theta: " << delta_theta.transpose() <<", vel: " << (delta_theta / delta_time).transpose() << endl;
            // cout << "pose delta time " << delta_time << endl;
        }

        velocity = velocity.array() / loop;

        Eigen::Vector3d velocity_zerobased(velocity(0), velocity(2), -velocity(1));

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
    std::cout << "showing " << std::to_string(seq_count) << endl;
    // cv::imshow("curr_warpped_event_image_gt", curr_warpped_event_image_gt);

    // cv::imshow("curr_map_image", curr_map_image);
    // cv::imshow("hot_image_C3", hot_image_C3);

    cv::waitKey(1);
}
