
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
        const double delta_time_later, 
        const Eigen::Matrix3d& K, 
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_later_ptr,
        Eigen::Vector3d& last_est_N_norm, double yaml_regulization_factor):
            points_(points), 
            delta_time_early_(delta_time_early), 
            delta_time_later_(delta_time_later), 
            intrisic_(K), 
            interpolator_early_ptr_(interpolator_early_ptr),
            interpolator_later_ptr_(interpolator_later_ptr),
            last_N_norm_(last_est_N_norm), yaml_regulization_factor_(yaml_regulization_factor)
    {

    }

    // operator 
    template<typename T> 
    bool operator()(const T* const ag_v_ang, T* residual) const  // ag_v_ang 8 vars of homography 
    {
        // homography https://blog.csdn.net/heyijia0327/article/details/53782094/  H = R - dt*v*N_normed
        Eigen::Matrix<T, 3, 1> points_T;
        points_T(0) = T(points_(0));
        points_T(1) = T(points_(1));
        points_T(2) = T(points_(2));
        
        // WARNING TODO 由于 N和T不是简单取负值, 所以double warp的策略不成立
        Eigen::Matrix<T, 3, 1> delta_points_T, delta_second_points_T;
        Eigen::Matrix<T, 3, 1> points_early_T, points_later_T;
        Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;

       
        Eigen::Matrix<T, 3, 1> vel; // translation velocity
        vel(0) = ag_v_ang[3];
        vel(1) = ag_v_ang[4];
        vel(2) = ag_v_ang[5];
        

        Eigen::Matrix<T, 3, 1> N_vec = {ag_v_ang[6], ag_v_ang[7], ag_v_ang[8]}; 
        Eigen::Matrix<T, 3, 1> N_norm = N_vec.array() / N_vec.norm(); // avoid div zero!!


        // exactly method 
        {
            Eigen::Matrix<T, 3, 3> Rotation_early;
            Eigen::Matrix<T, 3, 1> ag_early = {ag_v_ang[0], ag_v_ang[1], ag_v_ang[2]};
            // ceres rotation
            ag_early = ag_early * T(delta_time_early_);         
            ceres::AngleAxisToRotationMatrix(&ag_early(0), &Rotation_early(0));
            points_early_T = (Rotation_early + vel * N_norm.transpose() * T(delta_time_early_)).inverse().matrix() * points_T;
        
            Eigen::Matrix<T, 3, 3> Rotation_later;
            Eigen::Matrix<T, 3, 1> ag_later = {ag_v_ang[0], ag_v_ang[1], ag_v_ang[2]};
            // ceres rotation
            ag_later = ag_later * T(delta_time_later_);         
            ceres::AngleAxisToRotationMatrix(&ag_later(0), &Rotation_later(0));
            points_later_T = (Rotation_later + vel * N_norm.transpose() * T(delta_time_later_)).matrix() * points_early_T;
    
        }  

        // approx method 
        // {
        //     // from t2->t1, so Pt1 = (I+skew)*Pt1 + trans 
        //     delta_points_T(0) = -ag_v_ang[2]*T(points_(1)) + ag_v_ang[1]*T(points_(2));  
        //     delta_points_T(1) =  ag_v_ang[2]*T(points_(0)) - ag_v_ang[0]*T(points_(2));
        //     delta_points_T(2) = -ag_v_ang[1]*T(points_(0)) + ag_v_ang[0]*T(points_(1));

        //     points_early_T(0) = T(points_(0)) + delta_points_T(0)*T(delta_time_early_) ;// + delta_second_points_T(0)*T(0.5*delta_time_early_*delta_time_early_);
        //     points_early_T(1) = T(points_(1)) + delta_points_T(1)*T(delta_time_early_) ;// + delta_second_points_T(1)*T(0.5*delta_time_early_*delta_time_early_);
        //     points_early_T(2) = T(points_(2)) + delta_points_T(2)*T(delta_time_early_) ;// + delta_second_points_T(2)*T(0.5*delta_time_early_*delta_time_early_);

        //     // the homography is not simply set negative time, but using mattrix inverse !!!  
        //     points_early_T(0) += T(delta_time_early_) * vel(0) * N_norm.transpose() * points_T;
        //     points_early_T(1) += T(delta_time_early_) * vel(1) * N_norm.transpose() * points_T;
        //     points_early_T(2) += T(delta_time_early_) * vel(2) * N_norm.transpose() * points_T;
        // }

        points_2D_early_T(0) = points_early_T(0)/points_early_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_early_T(1) = points_early_T(1)/points_early_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));

        points_2D_later_T(0) = points_later_T(0)/points_later_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_later_T(1) = points_later_T(1)/points_later_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));
                

        // cout << "points "<< points(0) << ", "<< points(1) << endl;

        /* ceres interpolate version  */
        {
            T early_loss = T(0), later_loss = T(0);
            interpolator_early_ptr_->Evaluate(points_2D_early_T(1), points_2D_early_T(0), &early_loss);
            interpolator_later_ptr_->Evaluate(points_2D_later_T(1), points_2D_later_T(0), &later_loss);

            // cout << "intered " << early_loss << ", later " << later_loss << endl;
            residual[0] = early_loss + later_loss; 
            residual[1] = T(yaml_regulization_factor_)*
                                (ceres::pow(T(last_N_norm_(0)) - ag_v_ang[6], 2) + 
                                ceres::pow(T(last_N_norm_(1)) - ag_v_ang[7], 2) + 
                                ceres::pow(T(last_N_norm_(2)) - ag_v_ang[8], 2));

        }

        return true;
    }

    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const double delta_time_later, 
        const Eigen::Matrix3d& K,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_later_ptr,
        Eigen::Vector3d& last_est_N_norm, double yaml_regulization_factor)
        {
            // Eigen::Matrix<double, 3, 3> Rotation_later; 
            // Eigen::Matrix<double, 3, 1> ag_later = {ag_v_ang[0], ag_v_ang[1], ag_v_ang[2]};
            // ag_later = ag_later * delta_time_later;       
            // ceres::AngleAxisToRotationMatrix(&ag_later(0), &Rotation_later(0));
            // vel * N_norm.transpose() delta_time_later_;  
            
            return new ceres::AutoDiffCostFunction<ResidualCostFunction, 2, 9>(
                new ResidualCostFunction(points, delta_time_early, delta_time_later, K, 
                    interpolator_early_ptr,
                    interpolator_later_ptr,
                    last_est_N_norm, yaml_regulization_factor));
        }

    // inputs     
    Eigen::Vector3d points_;
    double delta_time_early_, delta_time_later_;
    Eigen::Matrix3d intrisic_; 

    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_early_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_later_ptr_;
    cv::Mat cv_earlier_timesurface_;
    cv::Mat cv_later_timesurface_;
    Eigen::Vector3d last_N_norm_;
    double yaml_regulization_factor_;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};




/**
* \brief using time as distance, self boosting, using ceres as optimizer, const gradient for each optimizing.
  \param [ts_start, ts_end]: time_range to form timesurface template  
  \param sample_num: the sample count beyond the time_range  
*/
void System::EstimateMotion_ransca_ceres(double ts_start, double ts_end, int sample_num, int total_iter_num)
{
    cout <<seq_count<< " time "<< eventBundle.first_tstamp.toSec() <<  ", total "
            << eventBundle.size << ", duration " << (eventBundle.last_tstamp - eventBundle.first_tstamp).toSec() 
            << ", sample " << sample_num << ", ts_start " << ts_start<<"~"<< ts_end << ", iter " << total_iter_num <<endl;

    bool show_time_info = false;
    // measure time 


    double residuals; 
    double ag_v_ang[9] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2), 
                          est_trans_velocity(0), est_trans_velocity(1), est_trans_velocity(2),
                        est_N_norm(0), est_N_norm(1), est_N_norm(2)};
    // double ag_v_ang[9] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.1 , 0.1, 0.999545};

    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(ag_v_ang[0],ag_v_ang[1],ag_v_ang[2]);
        Eigen::Vector3d eg_trans_vel(ag_v_ang[3],ag_v_ang[4],ag_v_ang[5]);
        Eigen::Vector3d eg_N_norm(ag_v_ang[6],ag_v_ang[7], ag_v_ang[8]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        // double timesurface_range = iter_/50.0 + 0.2;  // original 
        // double timesurface_range = (iter_)/60.0 + 0.2;
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 
        cv::Mat cv_later_timesurface = cv::Mat(180,240, CV_32FC1); 


        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);
        // cout << "default early " << default_value << endl; 
        float later_default_value = eventBundle.time_delta(eventBundle.size-1) - eventBundle.time_delta(int(eventBundle.size*(1-timesurface_range)));
        cv_later_timesurface.setTo(later_default_value * yaml_default_value_factor);
        // cout << "default later " << default_value << endl; 

        // get t0 time surface of warpped image using latest angleAxis
        
        getWarpedEventImage(eg_angleAxis, eg_trans_vel, eg_N_norm, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
                
        // get early timesurface
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 
        cv_early_timesurface_float_ = cv_earlier_timesurface.clone();

        // get t1 time surface of warpped image
        getWarpedEventImage(eg_angleAxis,eg_trans_vel, eg_N_norm, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE, true);
        // // get later timesurface
        for(int i= event_warpped_Bundle.size*(1-timesurface_range); i< event_warpped_Bundle.size; i++)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
                cv_later_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(event_warpped_Bundle.size-1) - eventBundle.time_delta(i);  
        } 

            /* visualize timesurface */  
            // {
                
            //     cv::Mat cv_earlier_timesurface_8U, cv_earlier_timesurface_color; 
            //     cv::normalize(cv_earlier_timesurface, cv_earlier_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
            //     // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
            //     cv::applyColorMap(cv_earlier_timesurface_8U, cv_earlier_timesurface_color, cv::COLORMAP_JET);
            //     cv::imshow("timesurface_early", cv_earlier_timesurface_color);
            //     cv::waitKey(10);
            // }
            // {
                // visualize timesurface 
                // cv::Mat cv_later_timesurface_8U, cv_later_timesurface_color; 
                // cv::normalize(cv_later_timesurface, cv_later_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
                // // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
                // cv::applyColorMap(cv_later_timesurface_8U, cv_later_timesurface_color, cv::COLORMAP_JET);
                // cv::imshow("timesurface_later", cv_later_timesurface_color);
                // cv::waitKey(100);
            // }
        // add gaussian on cv_earlier_timesurface
        cv::Mat cv_earlier_timesurface_blur, cv_later_timesurface_blur;
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);
        cv::GaussianBlur(cv_later_timesurface, cv_later_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        // get timesurface in ceres 
        vector<float> line_grid_early; line_grid_early.assign((float*)cv_earlier_timesurface_blur.data, (float*)cv_earlier_timesurface_blur.data + 180*240);
        vector<float> line_grid_later; line_grid_later.assign((float*)cv_later_timesurface_blur.data, (float*)cv_later_timesurface_blur.data + 180*240);

        ceres::Grid2D<float,1> grid_early(line_grid_early.data(), 0, 180, 0, 240);
        ceres::Grid2D<float,1> grid_later(line_grid_later.data(), 0, 180, 0, 240);

        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_early);
        auto* interpolator_later_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_later);  



        // sample events 
        // select 100 random points, and warp delta_t < min(t_point_delta_t). 
        // accumulate all time difference before and after warpped points. 
        std::vector<int> vec_sampled_idx; 
        int samples_count = std::min(sample_num, int(eventBundle.size)); 
        getSampledVec(vec_sampled_idx, samples_count, 0, 1);

        ceres::Problem problem; 
        // add residual 
        for(int loop_temp =0; loop_temp < vec_sampled_idx.size(); loop_temp++)
        {
            size_t sample_idx = vec_sampled_idx[loop_temp];
            double early_time =  eventBundle.time_delta(sample_idx);
            double later_time =  eventBundle.time_delta(event_warpped_Bundle.size-1) - eventBundle.time_delta(0);

            ceres::CostFunction* cost_function = ResidualCostFunction::Create(
                                                    event_undis_Bundle.coord_3d.col(sample_idx),
                                                    early_time, later_time, 
                                                    camera.eg_cameraMatrix,
                                                    interpolator_early_ptr, interpolator_later_ptr,
                                                    last_est_N_norm, yaml_regulization_factor);

            problem.AddResidualBlock(cost_function, nullptr, &ag_v_ang[0]);

            // set norm constant 
            // if(iter_%2==0)
            // {
            //     std::vector<int> constant_translation = {6,7,8}; // Norm vec 
            //     ceres::SubsetParameterization* constant_transform_parameterization =
            //                         new ceres::SubsetParameterization(9, constant_translation);
            //     problem.SetParameterization(&ag_v_ang[0],
            //                       constant_transform_parameterization);
            // }
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = yaml_ceres_iter_thread;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.use_nonmonotonic_steps = true;
        options.max_num_iterations = yaml_ceres_iter_num;
        // options.initial_trust_region_radius = 1;
        problem.SetParameterLowerBound(&ag_v_ang[0],0,-8);
        problem.SetParameterUpperBound(&ag_v_ang[0],0, 8);
        problem.SetParameterLowerBound(&ag_v_ang[0],1,-6);
        problem.SetParameterUpperBound(&ag_v_ang[0],1, 6);
        problem.SetParameterLowerBound(&ag_v_ang[0],2,-15);
        problem.SetParameterUpperBound(&ag_v_ang[0],2, 15);

        problem.SetParameterLowerBound(&ag_v_ang[0],6, -1);
        problem.SetParameterUpperBound(&ag_v_ang[0],6, +1);
        problem.SetParameterLowerBound(&ag_v_ang[0],7, -1);
        problem.SetParameterUpperBound(&ag_v_ang[0],7, +1);
        problem.SetParameterLowerBound(&ag_v_ang[0],8, -1);
        problem.SetParameterUpperBound(&ag_v_ang[0],8, +1);
        ceres::Solver::Summary summary; 

        // evaluate: choose init velocity, test whether using last_est or {0,0,0},
        // if(iter_ == 1)
        if(false)
        {
            double cost = 0;
            vector<double> residual_vec; 
            // previous old velocity  
            ag_v_ang[0] = est_angleAxis(0); 
            ag_v_ang[1] = est_angleAxis(1); 
            ag_v_ang[2] = est_angleAxis(2); 
            ag_v_ang[3] = est_trans_velocity(0); 
            ag_v_ang[4] = est_trans_velocity(1); 
            ag_v_ang[5] = est_trans_velocity(2); 
            ag_v_ang[6] = est_N_norm(0); 
            ag_v_ang[7] = est_N_norm(1);  
            ag_v_ang[8] = est_N_norm(2);  

            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
            double residual_sum_old = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);
            // 0 init 
            memset(ag_v_ang, 0.1, sizeof(double)*6); // skip the norm

            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
            double residual_sum_0 = std::accumulate(residual_vec.begin(), residual_vec.end(), 0.0);

            if(residual_sum_old < residual_sum_0)
            {
                ag_v_ang[0] = est_angleAxis(0); 
                ag_v_ang[1] = est_angleAxis(1); 
                ag_v_ang[2] = est_angleAxis(2); 
                ag_v_ang[3] = est_trans_velocity(0); 
                ag_v_ang[4] = est_trans_velocity(1); 
                ag_v_ang[5] = est_trans_velocity(2); 
                ag_v_ang[6] = est_N_norm(0); 
                ag_v_ang[7] = est_N_norm(1);
                ag_v_ang[8] = est_N_norm(2);
            }
            
            // cout << "using ang" << ag_v_ang[0] << "," << ag_v_ang[1] << "," << ag_v_ang[2] <<
            // " trans " << ag_v_ang[3] << "," << ag_v_ang[4] << "," << ag_v_ang[5] <<
            // " norm " << ag_v_ang[6] << "," << ag_v_ang[7] << "," <<  ag_v_ang[8] << 
            // ", residual size " << residual_vec.size() << ", sum_0: "<<  residual_sum_0 << ", sum_old: " <<residual_sum_old << endl; 
        }

        ros::Time t1 = ros::Time::now();
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << endl;
        ros::Time t2 = ros::Time::now();
        // cout << "ceres time " << (t2-t1).toSec() << endl;

    }


    // output data 
    est_angleAxis = Eigen::Vector3d(ag_v_ang[0],ag_v_ang[1],ag_v_ang[2]);
    est_trans_velocity = Eigen::Vector3d(ag_v_ang[3],ag_v_ang[4],ag_v_ang[5]);
    est_N_norm = Eigen::Vector3d(ag_v_ang[6],ag_v_ang[7], ag_v_ang[8]);

    last_est_N_norm = est_N_norm; 

    // cout << "Loss: " << 0 << ", est_angleAxis " << est_angleAxis.transpose() << 
    // ", trans " << est_trans_velocity.transpose() << ", norm " << est_N_norm.transpose() << endl;

}

