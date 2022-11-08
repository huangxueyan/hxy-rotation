
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>
#include <algorithm>
#include <ceres/cubic_interpolation.h>
using namespace std;



/**
* \brief used for ceres to implement CM methods, automatie version. .
*/
struct ResidualCostFunction
{

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ResidualCostFunction(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const Eigen::Matrix3d& K, 
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_early_ptr):
            points_(points), delta_time_early_(delta_time_early), 
            intrisic_(K), interpolator_early_ptr_(interpolator_early_ptr)
    {
    }

    // operator 
    template<typename T> 
    bool operator()(const T* const ag, T* residual) const
    {
        
        int count = 0; 
        Eigen::Matrix<T, 3, 1> delta_points_T, delta_second_points_T;
        Eigen::Matrix<T, 3, 1> points_early_T, points_later_T;
        Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;


        // taylor first : points * skew_matrix

        {   // first order version 
            delta_points_T(0) = -ag[2]*T(points_(1)) + ag[1]*T(points_(2));  
            delta_points_T(1) =  ag[2]*T(points_(0)) - ag[0]*T(points_(2));
            delta_points_T(2) = -ag[1]*T(points_(0)) + ag[0]*T(points_(1));
            points_early_T(0) = T(points_(0)) + delta_points_T(0)*T(delta_time_early_) ;
            points_early_T(1) = T(points_(1)) + delta_points_T(1)*T(delta_time_early_) ;
            points_early_T(2) = T(points_(2)) + delta_points_T(2)*T(delta_time_early_) ;
        }

        // taylor second : points * skew_matrix * skew_matrix
        {   // second order version
            // delta_points_T(0) = -ag[2]*T(points_(1)) + ag[1]*T(points_(2));  
            // delta_points_T(1) =  ag[2]*T(points_(0)) - ag[0]*T(points_(2));
            // delta_points_T(2) = -ag[1]*T(points_(0)) + ag[0]*T(points_(1));
            // delta_second_points_T(0) = -ag[2]*delta_points_T(1) + ag[1]*delta_points_T(2);
            // delta_second_points_T(1) =  ag[2]*delta_points_T(0) - ag[0]*delta_points_T(2);
            // delta_second_points_T(2) = -ag[1]*delta_points_T(0) + ag[0]*delta_points_T(1);

            // points_early_T(0) = T(points_(0)) + delta_points_T(0)*T(delta_time_early_) + delta_second_points_T(0)*T(0.5*delta_time_early_*delta_time_early_);
            // points_early_T(1) = T(points_(1)) + delta_points_T(1)*T(delta_time_early_) + delta_second_points_T(1)*T(0.5*delta_time_early_*delta_time_early_);
            // points_early_T(2) = T(points_(2)) + delta_points_T(2)*T(delta_time_early_) + delta_second_points_T(2)*T(0.5*delta_time_early_*delta_time_early_);
        }
   
        {  // exactly version 
            // Eigen::Matrix<T, 3,1> points = {T(points_(0)), T(points_(1)), T(points_(2))};
            // Eigen::Matrix<T, 3,1> angaxis = {ag[0], ag[1], ag[2]};
            // angaxis = angaxis * T(delta_time_early_);
            // ceres::AngleAxisRotatePoint(&angaxis(0), &points(0), &points_early_T(0));
        }


        // cout << "points "<< points(0) << ", "<< points(1) << endl;
        points_2D_early_T(0) = points_early_T(0)/points_early_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_early_T(1) = points_early_T(1)/points_early_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));
        

        // cout << "points "<< points(0) << ", "<< points(1) << endl;
        

        /* ceres interpolate version  */
        {
            T early_loss = T(0);
            interpolator_early_ptr_->Evaluate(points_2D_early_T(1), points_2D_early_T(0), &early_loss);

            // cout << "intered " << early_loss << ", later " << later_loss << endl;
            residual[0] = early_loss; 
        }

        return true;
    }

    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const Eigen::Matrix3d& K,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_early_ptr)
        {
            return new ceres::AutoDiffCostFunction<ResidualCostFunction,1, 3>(
                new ResidualCostFunction(points, delta_time_early, K, 
                    interpolator_early_ptr));
        }

    // inputs     
    Eigen::Vector3d points_;
    double delta_time_early_;
    Eigen::Matrix3d intrisic_; 

    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_early_ptr_;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};




/**
* \brief using time as distance, self boosting, using ceres as optimizer, const gradient for each optimizing.
  \param [ts_start, ts_end]: time_range to form timesurface template  
  \param sample_num: the sample count beyond the time_range  
*/
void System::EstimateMotion_ransca_ceres()
{
    double ts_start = yaml_ts_start, ts_end = yaml_ts_end;
    int sample_num = yaml_sample_count, total_iter_num = yaml_iter_num;

    cout <<seq_count<< " time "<< eventBundle.first_tstamp.toSec() <<  ", total "
            << eventBundle.size << ", duration " << (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec() 
            << ", sample " << sample_num << ", ts_start " << ts_start<<"~"<< ts_end << ", iter " << total_iter_num <<endl;

    bool show_time_info = false;
    // measure time 
    ros::Time t1, t2; 


    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    Eigen::Vector3d last_est_angleAxis = est_angleAxis;
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        // double timesurface_range = iter_/50.0 + 0.2;  // original 
        // double timesurface_range = (iter_)/60.0 + 0.2;
        
    t1 = ros::Time::now();
        // get t0 time surface of warpped image using latest angleAxis
        // getWarpedEventImage(eg_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
        getWarpedEvent(event_undis_Bundle, eg_angleAxis, event_warpped_Bundle, false);
    t2 = ros::Time::now();
    total_warpevents_time += (t2-t1).toSec(); 

    t1 = ros::Time::now();
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(camera.height, camera.width, CV_32FC1); 

        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);
        // cout << "default early " << default_value << endl; 

        // visual optimizng process 
        // if(iter_ % 2  == 0) 
        // {
        //     cv::Mat temp_img;
        //     getWarpedEventImage(eg_angleAxis, event_warpped_Bundle).convertTo(temp_img, CV_32FC3);
        //     cv::imshow("opti", temp_img);
        //     // cv::Mat temp_img_char;
        //     // cv::threshold(temp_img, temp_img_char, 0.1, 255, cv::THRESH_BINARY);
        //     // cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/optimize/opti_" + std::to_string(iter_) + ".png", temp_img_char);
        //     cv::waitKey(30);
        // }


        // get early timesurface
        t1 = ros::Time::now();
        for(int i = event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
            float* row_ptr = cv_earlier_timesurface.ptr<float>(sampled_y);
            row_ptr[sampled_x] = eventBundle.time_delta(i);
            // cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 
        // need for dynamic size 
        cv_early_timesurface_float_ = cv_earlier_timesurface.clone();


        // add gaussian on cv_earlier_timesurface
        cv::Mat cv_earlier_timesurface_blur;
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);

    t2 = ros::Time::now(); 
    total_timesurface_time += (t2-t1).toSec();


    t1 = ros::Time::now();
        // get timesurface in ceres 
        int pixel_count = camera.height * camera.width;
        vector<float> line_grid_early; line_grid_early.assign((float*)cv_earlier_timesurface_blur.data, (float*)cv_earlier_timesurface_blur.data + pixel_count);

        ceres::Grid2D<float,1> grid_early(line_grid_early.data(), 0, camera.height, 0, camera.width);
        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_early);

        // sample events 
        // select 100 random points, and warp delta_t < min(t_point_delta_t). 
        // accumulate all time difference before and after warpped points. 
        std::vector<int> vec_sampled_idx; 
        // int samples_count = std::min(sample_num, int(eventBundle.size)); 
        int samples_count = std::min(sample_num, int(eventBundle.size * 0.8)); 
        getSampledVec(vec_sampled_idx, samples_count, 0, 1);

        // init problem 
        ceres::Problem problem; 
        // add residual 
        for(int loop_temp =0; loop_temp < vec_sampled_idx.size(); loop_temp++)
        {
            size_t sample_idx = vec_sampled_idx[loop_temp];
            double early_time =  eventBundle.time_delta(sample_idx); // positive 

            ceres::CostFunction* cost_function = ResidualCostFunction::Create(
                                                    event_undis_Bundle.coord_3d.col(sample_idx),
                                                    early_time, 
                                                    camera.eg_cameraMatrix,
                                                    interpolator_early_ptr);

            problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = yaml_ceres_iter_thread;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.use_nonmonotonic_steps = true;
        options.max_num_iterations = yaml_ceres_iter_num;
        // options.initial_trust_region_radius = 1;
        problem.SetParameterLowerBound(&angleAxis[0],0,-20);
        problem.SetParameterLowerBound(&angleAxis[0],1,-20);
        problem.SetParameterLowerBound(&angleAxis[0],2,-20);
        problem.SetParameterUpperBound(&angleAxis[0],0, 20);
        problem.SetParameterUpperBound(&angleAxis[0],1, 20);
        problem.SetParameterUpperBound(&angleAxis[0],2, 20);

        ceres::Solver::Summary summary; 

        // evaluate: choose init velocity, test whether using last_est or {0,0,0},
        // if(iter_ == 1)
        if (false)
        {
            double cost = 0;
            vector<double> residual_vec; 
            // previous old velocity  
            angleAxis[0] = est_angleAxis(0); angleAxis[1] = est_angleAxis(1); angleAxis[1] = est_angleAxis(2); 
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
            double residual_sum_old = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);
            // 0 init 
            angleAxis[0] = 0; angleAxis[1] = 0; angleAxis[2] = 0;  
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
            double residual_sum_0 = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);

            if(residual_sum_old < residual_sum_0)
            {
                angleAxis[0] = est_angleAxis(0); 
                angleAxis[1] = est_angleAxis(1);
                angleAxis[1] = est_angleAxis(2);  
            }
            // cout << "using " << angleAxis[0] << "," << angleAxis[1] << "," << angleAxis[2] 
            // << ", residual size " << residual_vec.size() << ", sum_0: "<<  residual_sum_0 << ", sum_old: " <<residual_sum_old << endl; 
        }

        ceres::Solve(options, &problem, &summary);
    t2 = ros::Time::now();
    total_ceres_time += (t2-t1).toSec();


        // Eigen::Vector3d temp_vel = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
        // if (last_est_angleAxis.norm()!=0 && (last_est_angleAxis - temp_vel).norm() > 1.0) {
        //     cout << "failuer solved" << endl;
        //     angleAxis[0] = 0;
        //     angleAxis[1] = 0;
        //     angleAxis[2] = 0;
        //     last_est_angleAxis.setZero();
        // }
        // last_est_angleAxis = temp_vel;

        // if (iter_ == total_iter_num - 1) {
        if (false) {
            // get gt velocity 
            double residual_sum = 0, cost = 0;
            int samples_count = 3000;

            Eigen::Vector3d eg_angleAxis(angleAxis);
            
            EventBundle my_warpped_Bundle;
            cv::Mat IMG;
            getWarpedEventImage(eg_angleAxis, my_warpped_Bundle).convertTo(IMG, CV_32F);
            // cv::imwrite("/home/hxy/Desktop/ECCV22-all/hxy-rotation/devel_release/hxy_" + std::to_string(seq_count) + "_warp.png", myimg);
    
            // add more residual s  
            {
                std::vector<int> vec_sampled; 
                getSampledVec(vec_sampled, samples_count, 0, 0.3);
                int outlier = 0;
                for(int loop_temp =0; loop_temp < vec_sampled.size(); loop_temp++)
                {
                    int idx = vec_sampled[loop_temp];
                    int sampled_x = std::round(my_warpped_Bundle.coord.col(idx)[0]);
                    int sampled_y = std::round(my_warpped_Bundle.coord.col(idx)[1]); 
                    if(my_warpped_Bundle.isInner[idx] < 1)               // outlier 
                        outlier++;    
                    else if (cv_earlier_timesurface.at<float>(sampled_y, sampled_x) - 0.0001 > eventBundle.time_delta(idx)) 
                        outlier++;                      
                }

                cout << "0.-0.3 sample " << samples_count << ", outlier count " << outlier << ", rate " << 100 * outlier / float(samples_count) << "%"<< endl;
             }

            // add more residual s 
            {
                std::vector<int> vec_sampled; 
                getSampledVec(vec_sampled, samples_count, 0.3, 0.5);
                int outlier = 0;
                for(int loop_temp =0; loop_temp < vec_sampled.size(); loop_temp++)
                {
                    int idx = vec_sampled[loop_temp];
                    int sampled_x = std::round(my_warpped_Bundle.coord.col(idx)[0]);
                    int sampled_y = std::round(my_warpped_Bundle.coord.col(idx)[1]); 
                    if(my_warpped_Bundle.isInner[idx] < 1)               // outlier 
                        outlier++;    
                    else if (cv_earlier_timesurface.at<float>(sampled_y, sampled_x) - 0.0001 > eventBundle.time_delta(idx)) 
                        outlier++;                        
                }

                cout << "0.3-0.5 sample " << samples_count << ", outlier count " << outlier << ", rate " << 100 * outlier / float(samples_count) << "%"<< endl;
            }

            // add more residual s 
            {
                std::vector<int> vec_sampled; 
                getSampledVec(vec_sampled, samples_count, 0.5, 0.8);
                int outlier = 0;
                for(int loop_temp =0; loop_temp < vec_sampled.size(); loop_temp++)
                {
                    int idx = vec_sampled[loop_temp];
                    int sampled_x = std::round(my_warpped_Bundle.coord.col(idx)[0]);
                    int sampled_y = std::round(my_warpped_Bundle.coord.col(idx)[1]); 
                    if(my_warpped_Bundle.isInner[idx] < 1)               // outlier 
                        outlier++;    
                    else if (cv_earlier_timesurface.at<float>(sampled_y, sampled_x) - 0.0001 > eventBundle.time_delta(idx)) 
                        outlier++;                       
                }

                cout << "0.5-0.8 sample " << samples_count << ", outlier count " << outlier << ", rate " << 100 * outlier / float(samples_count) << "%"<< endl;
            }

            // add more residual s 
            {
                std::vector<int> vec_sampled; 
                getSampledVec(vec_sampled, samples_count, 0.8, 1.0);
                int outlier = 0;
                for(int loop_temp =0; loop_temp < vec_sampled.size(); loop_temp++)
                {
                    int idx = vec_sampled[loop_temp];
                    int sampled_x = std::round(my_warpped_Bundle.coord.col(idx)[0]);
                    int sampled_y = std::round(my_warpped_Bundle.coord.col(idx)[1]); 
                    if(my_warpped_Bundle.isInner[idx] < 1)               // outlier 
                        outlier++;    
                    else if (cv_earlier_timesurface.at<float>(sampled_y, sampled_x) - 0.0001 > eventBundle.time_delta(idx)) 
                        outlier++;                          
                }

                cout << "0.8-1.0 sample " << samples_count << ", outlier count " << outlier << ", rate " << 100 * outlier / float(samples_count) << "%"<< endl;
            }
        }


    }

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);


    // cout << "Loss: " << 0 << ", est_angleAxis " << est_angleAxis.transpose() << endl;

}

