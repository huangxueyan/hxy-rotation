
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
        Eigen::Matrix<T, 3, 1> points_early_T, points_later_T;
        Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;


        {   // first order version 
            points_early_T(0) = T(points_(0)) + ag[0]*T(delta_time_early_) ;
            points_early_T(1) = T(points_(1)) + ag[1]*T(delta_time_early_) ;
        }       


        // cout << "points "<< points(0) << ", "<< points(1) << endl;
        points_2D_early_T(0) = points_early_T(0)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_early_T(1) = points_early_T(1)*T(intrisic_(1,1)) + T(intrisic_(1,2));

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
            return new ceres::AutoDiffCostFunction<ResidualCostFunction,1, 2>(
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

void System::EstimateMotion_ransca_ceres_evaluate()
{

    double ts_start = yaml_ts_start, ts_end = yaml_ts_end;
    int sample_num = yaml_sample_count, total_iter_num = yaml_iter_num;

    cout <<seq_count<< " time "<< eventBundle.first_tstamp.toSec() <<  ", total "
            << eventBundle.size << ", duration " << (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec() 
            << ", sample " << sample_num << ", ts_start " << ts_start<<"~"<< ts_end << ", iter " << total_iter_num <<endl;

    double residuals; 
    // est_trans_vel = Eigen::Vector2d::Zero();
    double trans_vel[2] = {est_trans_vel(0), est_trans_vel(1)}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector2d eg_trans_vel(trans_vel[0],trans_vel[1]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 

        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);
        // cout << "default early " << default_value << endl; 

        
        getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
        
        // get early timesurface
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 

            /* visualize timesurface */  
            // {
                cv::Mat cv_earlier_timesurface_8U, cv_earlier_timesurface_color; 
                cv::normalize(cv_earlier_timesurface, cv_earlier_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
                // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
                cv::applyColorMap(cv_earlier_timesurface_8U, cv_earlier_timesurface_color, cv::COLORMAP_JET);
                cv::imshow("timesurface_early", cv_earlier_timesurface_color);
                cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/timesurface_Single__early_" +std::to_string(iter_)+".png", cv_earlier_timesurface_color);
                cv::waitKey(10);
            // }

        // add gaussian on cv_earlier_timesurface
        cv::Mat cv_earlier_timesurface_blur;
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        // get timesurface in ceres 
        vector<float> line_grid_early; line_grid_early.assign((float*)cv_earlier_timesurface_blur.data, (float*)cv_earlier_timesurface_blur.data + 180*240);

        ceres::Grid2D<float,1> grid_early(line_grid_early.data(), 0, 180, 0, 240);
        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_early);

        // sample events 
        // select 100 random points, and warp delta_t < min(t_point_delta_t). 
        // accumulate all time difference before and after warpped points. 
        std::vector<int> vec_sampled_idx; 
        int samples_count = std::min(sample_num, int(eventBundle.size)); 
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

            problem.AddResidualBlock(cost_function, nullptr, &trans_vel[0]);
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = yaml_ceres_iter_thread;
        options.max_num_iterations = yaml_ceres_iter_num;
        // problem.SetParameterLowerBound(&trans_vel[0],0,-20);
        // problem.SetParameterLowerBound(&trans_vel[0],1,-20);
        // problem.SetParameterUpperBound(&trans_vel[0],0, 20);
        // problem.SetParameterUpperBound(&trans_vel[0],1, 20);

        ceres::Solver::Summary summary; 

        // evaluate: choose init velocity, test whether using last_est or {0,0,0},
        
            double cost = 0;
            vector<double> residual_vec; 
            // previous old velocity  
            double pre_trans_vel[2] = {trans_vel[0], trans_vel[1]};
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
            double residual_sum_est = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);

            cout << "estimated residual " << residual_sum_est << endl;

            int right_x_range = 20, left_x_range = -90;
            int right_y_range = 70, left_y_range = -40;
            cv::Mat loss_img(right_y_range-left_y_range, right_x_range-left_x_range, CV_32F, cv::Scalar(0));
            
            double best_var = 1e8;
            int best_x = 0, best_y = 0;

            for(int _y = left_y_range ; _y < right_y_range; _y++)
            for(int _x = left_x_range ; _x < right_x_range; _x++)
            {
                trans_vel[0] = 0.01 * _x; trans_vel[1] = 0.01 * _y; 
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
                double residual_sum = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0); 

                loss_img.at<float>(_y-left_y_range, _x - left_x_range) = residual_sum; 

                if(residual_sum < best_var)
                {
                    best_var = residual_sum;
                    best_x = _x; best_y = _y;
                    // getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
                    // cv::normalize(curr_warpped_event_image, curr_warpped_event_image, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
                    // cv::threshold(curr_warpped_event_image, curr_warpped_event_image, 0.1, 255, cv::THRESH_BINARY);
                    // cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/iwe_Single_" + std::to_string(iter_) + "_" 
                    //     + std::to_string(best_var) + "_("+std::to_string(_x) + "," + std::to_string(_y) + ").png", curr_warpped_event_image);
                }

            }  

            {
                Eigen::Vector2d eg_trans_vel = {0.01 * best_x, 0.01 * best_y};
                getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
                cv::normalize(curr_warpped_event_image, curr_warpped_event_image, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
                cv::threshold(curr_warpped_event_image, curr_warpped_event_image, 0.1, 255, cv::THRESH_BINARY);
                cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/iwe_Single_" + std::to_string(iter_) + "_" 
                    + std::to_string(best_var) + "_best("+std::to_string(best_x) + "," + std::to_string(best_y) + ").png", curr_warpped_event_image);

            }

            // restore previous value 
            trans_vel[0] = pre_trans_vel[0]; 
            trans_vel[1] = pre_trans_vel[1];     

            // visualize loss
            cv::Mat image_color;
            cv::normalize(loss_img, image_color, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
            cv::applyColorMap(image_color, image_color, cv::COLORMAP_JET);
            cv::namedWindow("loss", cv::WINDOW_NORMAL); 
            cv::imshow("loss", image_color);
            cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/loss_Single" +std::to_string(iter_) + ".png", image_color);
            cv::waitKey(10);

        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << endl;

    }

    est_trans_vel = Eigen::Vector2d(trans_vel[0],trans_vel[1]);
    cout << "est_trans_vel " << est_trans_vel.transpose() << endl;


}



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
    // est_trans_vel = Eigen::Vector2d::Zero();
    double trans_vel[2] = {est_trans_vel(0), est_trans_vel(1)}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector2d eg_trans_vel(trans_vel[0],trans_vel[1]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        // double timesurface_range = iter_/50.0 + 0.2;  // original 
        // double timesurface_range = (iter_)/60.0 + 0.2;
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 

        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);
        // cout << "default early " << default_value << endl; 

        
        t1 = ros::Time::now();
        getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
        t2 = ros::Time::now();
        if(show_time_info)
            cout << "getWarpedEventImage time " << (t2-t1).toSec() * 2 << endl;  // 0.00691187 s
        
        // visual optimizng process 
        // if(iter_ % 2  == 0) 
        // {
        //     cv::Mat temp_img;
        //     getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(temp_img, CV_32FC3);
        //     cv::imshow("opti", temp_img);
        //     // cv::Mat temp_img_char;
        //     // cv::threshold(temp_img, temp_img_char, 0.1, 255, cv::THRESH_BINARY);
        //     // cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/optimize/opti_" + std::to_string(iter_) + ".png", temp_img_char);
        //     cv::waitKey(30);
        // }


    t1 = ros::Time::now();
        // get early timesurface
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 
    t2 = ros::Time::now();
    if(show_time_info)
        cout << "cv_earlier_timesurface time " << (t2-t1).toSec() * 2 << endl; // 0.000106088

            /* visualize timesurface */  
            // {
                
            //     cv::Mat cv_earlier_timesurface_8U, cv_earlier_timesurface_color; 
            //     cv::normalize(cv_earlier_timesurface, cv_earlier_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
            //     // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
            //     cv::applyColorMap(cv_earlier_timesurface_8U, cv_earlier_timesurface_color, cv::COLORMAP_JET);
            //     cv::imshow("timesurface_early", cv_earlier_timesurface_color);
            //     cv::waitKey(10);
            // }

        // add gaussian on cv_earlier_timesurface
        cv::Mat cv_earlier_timesurface_blur;
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        // get timesurface in ceres 
        vector<float> line_grid_early; line_grid_early.assign((float*)cv_earlier_timesurface_blur.data, (float*)cv_earlier_timesurface_blur.data + 180*240);

        ceres::Grid2D<float,1> grid_early(line_grid_early.data(), 0, 180, 0, 240);
        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_early);

        // sample events 
        // select 100 random points, and warp delta_t < min(t_point_delta_t). 
        // accumulate all time difference before and after warpped points. 
    t1 = ros::Time::now();
        std::vector<int> vec_sampled_idx; 
        int samples_count = std::min(sample_num, int(eventBundle.size)); 
        getSampledVec(vec_sampled_idx, samples_count, 0, 1);
    t2 = ros::Time::now();
    if(show_time_info)
        cout << "getSampledVec time " << (t2-t1).toSec() << endl; // 0.000473817

    t1 = ros::Time::now();  
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

            problem.AddResidualBlock(cost_function, nullptr, &trans_vel[0]);
        }
    t2 = ros::Time::now();
    if(show_time_info)
        cout << "add residual time " << (t2-t1).toSec() << endl;  // 0.00168042

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = yaml_ceres_iter_thread;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.use_nonmonotonic_steps = true;
        options.max_num_iterations = yaml_ceres_iter_num;
        // options.initial_trust_region_radius = 1;
        problem.SetParameterLowerBound(&trans_vel[0],0,-20);
        problem.SetParameterLowerBound(&trans_vel[0],1,-20);
        problem.SetParameterUpperBound(&trans_vel[0],0, 20);
        problem.SetParameterUpperBound(&trans_vel[0],1, 20);

        ceres::Solver::Summary summary; 

        // evaluate: choose init velocity, test whether using last_est or {0,0,0},
        // if(iter_ == 1)
        // {
        //     double cost = 0;
        //     vector<double> residual_vec; 
        //     // previous old velocity  
        //     angleAxis[0] = est_trans_vel(0); angleAxis[1] = est_trans_vel(1); angleAxis[1] = est_trans_vel(2); 
        //     problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
        //     double residual_sum_old = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);
        //     // 0 init 
        //     angleAxis[0] = 0; angleAxis[1] = 0; angleAxis[2] = 0;  
        //     problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
        //     double residual_sum_0 = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);

        //     if(residual_sum_old < residual_sum_0)
        //     {
        //         angleAxis[0] = est_trans_vel(0); 
        //         angleAxis[1] = est_trans_vel(1);
        //         angleAxis[1] = est_trans_vel(2);  
        //     }
            
        //     // cout << "using " << angleAxis[0] << "," << angleAxis[1] << "," << angleAxis[2] 
        //     // << ", residual size " << residual_vec.size() << ", sum_0: "<<  residual_sum_0 << ", sum_old: " <<residual_sum_old << endl; 
        // }


        t1 = ros::Time::now();
        ceres::Solve(options, &problem, &summary);
        t2 = ros::Time::now();
        if(show_time_info)
            cout << "ceres time " << (t2-t1).toSec() << endl;  // 0.00383356 
        
        // cout << summary.BriefReport() << endl;

        // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;
        // if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
        // {
        //     cout << "   iter " << iter_ <<  ", No convergence Event bundle time " << eventBundle.first_tstamp.toSec() <<", size " << eventBundle.size << endl;
        //     // cout << summary.BriefReport() << endl;
        // }

        // {   // visualize hot map 

        //     // cout << "using angle axis " << est_trans_vel.transpose() << endl;
        //     // visualize warp events and get a timesurface image with indexs 
        //     {
        //         cv::Mat cv_timesurface; 
        //         cv_timesurface = getImageFromBundle(event_undis_Bundle, PlotOption::TIME_SURFACE);  
        //         cv::normalize(cv_timesurface, hot_image_C1, 0,255, cv::NORM_MINMAX, CV_8UC1);
        //         cv::cvtColor(hot_image_C1, hot_image_C3, cv::COLOR_GRAY2BGR);
        //     }

        //     int sample_green = 1;
        //     for(int i=0; i<vec_sampled_idx.size() ; i++)
        //     {
        //         // if(i<10) cout << "sampling " << sample_idx << endl;

        //         // viusal sample 
        //         if(i%sample_green != 0) continue;
        //         int x = int(event_undis_Bundle.coord(0, vec_sampled_idx[i]));
        //         int y = int(event_undis_Bundle.coord(1, vec_sampled_idx[i]));
        //         if(x>239 || y>179 || x<0 ||y<0) continue;
        //         hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,255,0);   // green of events before warp 
        //     }

        //     // compare with gt
        //     // cout << "estimated angleAxis " <<  est_trans_vel.transpose() << endl;   
            
        //     // plot earlier timestamps oringe 22，G：07，B：201
        //     for(int i=0; i<eventBundle.size/15 ; i++)
        //     {
        //         // viusal sample 
        //         // cout << "hello " << endl;
        //         if(i%sample_green != 0) continue;
        //         int x = int(event_undis_Bundle.coord(0, i));
        //         int y = int(event_undis_Bundle.coord(1, i));
        //         if(x>239 || y>179 || x<0 ||y<0) continue;
        //         hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 165, 255);   // bottom events earlier  
        //     }

        //     // visualize est
        //     int sample_red = 1;
        //     for(int i=0; i<vec_sampled_idx.size(); i++)
        //     {
        //         // viusal sample 
        //         if(i%sample_red != 0) continue;
        //         int x = int(event_warpped_Bundle.coord(0, vec_sampled_idx[i])); 
        //         int y = int(event_warpped_Bundle.coord(1, vec_sampled_idx[i]));
        //         if(x>239 || y>179 || x<0 ||y<0) continue;
        //         hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);   // red after warp
        //         // cout << "all inlier red" << endl;
        //     }
        // }


    }

    est_trans_vel = Eigen::Vector2d(trans_vel[0],trans_vel[1]);
    cout << "est_trans_vel " << est_trans_vel.transpose() << endl;

}

