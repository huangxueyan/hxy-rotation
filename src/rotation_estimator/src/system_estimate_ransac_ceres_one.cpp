
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>
#include <algorithm>
#include <ceres/cubic_interpolation.h>
using namespace std;


struct ResidualCostFunction
{
    ResidualCostFunction(
        const Eigen::Matrix3Xd& coord_3d, 
        const Eigen::VectorXd& delta_time, 
        const Eigen::Matrix3d& K,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_early_ptr)
        :_coord_3d(coord_3d), _delta_time(delta_time), _intrisic(K), interpolator_early_ptr_(interpolator_early_ptr)
    {
        // cout << "  _coord_3d \n" << _coord_3d.rows() << ", " << _coord_3d.cols() << endl;
        // cout << "  delta_time \n" << delta_time.rows() << ", " << delta_time.cols() << endl;
        // cout << "  _delta_time \n" << _delta_time.rows() << ", " << _delta_time.cols() << endl;
        // cout << "  _intrisic \n" << _intrisic << endl;
        N = coord_3d.cols();
        // cout << "CM loss init: size " << N << endl;

    }

    // operator 
    template<typename T> 
    bool operator()(const T* ag, T* residual) const
    {
        Eigen::Matrix<T, 3, -1> ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x, coord_3d_T;
        ang_vel_hat_mul_x.resize(3,_coord_3d.cols());
        ang_vel_hat_sqr_mul_x.resize(3,_coord_3d.cols());
        coord_3d_T.resize(3,_coord_3d.cols());
        
        coord_3d_T = _coord_3d.template cast<T>();
        Eigen::Matrix<T, -1, 1> delta_time_T = _delta_time.template cast<T>();
        
        // equation 11 
        ang_vel_hat_mul_x.row(0) = -ag[2]*coord_3d_T.row(1) + ag[1]*coord_3d_T.row(2);
        ang_vel_hat_mul_x.row(1) =  ag[2]*coord_3d_T.row(0) - ag[0]*coord_3d_T.row(2);
        ang_vel_hat_mul_x.row(2) = -ag[1]*coord_3d_T.row(0) + ag[0]*coord_3d_T.row(1);

        Eigen::Matrix<T,3,-1> new_coord_3d = coord_3d_T.array() + 
                                    ang_vel_hat_mul_x.array().rowwise() * delta_time_T.transpose().array();

        /* second order version
            ang_vel_hat_sqr_mul_x.row(0) = -ag[2]*ang_vel_hat_mul_x.row(1) + ag[1]*ang_vel_hat_mul_x.row(2);
            ang_vel_hat_sqr_mul_x.row(1) =  ag[2]*ang_vel_hat_mul_x.row(0) - ag[0]*ang_vel_hat_mul_x.row(2);
            ang_vel_hat_sqr_mul_x.row(2) = -ag[1]*ang_vel_hat_mul_x.row(0) + ag[0]*ang_vel_hat_mul_x.row(1);

            Eigen::Matrix<T,3,-1> new_coord_3d = coord_3d_T
                                            + Eigen::Matrix<T,3,-1>
                                            (   ang_vel_hat_mul_x.array().rowwise() 
                                                * (delta_time_T.transpose().array())
                                                + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                                * (T(0.5) * delta_time_T.transpose().array().square()) );
        */ 

        // project and store in a image 
        new_coord_3d.row(0) = new_coord_3d.row(0).array() / new_coord_3d.row(2).array() * T(_intrisic(0,0)) + T(_intrisic(0,2));
        new_coord_3d.row(1) = new_coord_3d.row(1).array() / new_coord_3d.row(2).array() * T(_intrisic(1,1)) + T(_intrisic(1,2));
        
        for(int i=0; i<N; i++)
        {
            interpolator_early_ptr_->Evaluate(new_coord_3d(1, i), new_coord_3d(0,i), &residual[i]);
        }

        return true;
    }
    
    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Matrix3Xd& coord_3d, 
        const Eigen::VectorXd& delta_time, 
        const Eigen::Matrix3d& K,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>* interpolator_early_ptr)
        {
            return new ceres::AutoDiffCostFunction<ResidualCostFunction, ceres::DYNAMIC, 3>(
                new ResidualCostFunction(coord_3d, delta_time, K, interpolator_early_ptr), coord_3d.cols());  // fun, residual num
        }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // inputs     
    Eigen::Matrix3d _intrisic; 
    Eigen::Matrix3Xd _coord_3d; 
    Eigen::VectorXd _delta_time;
    int N; 
    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_early_ptr_;

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
    ros::Time t1, t2; 


    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        // double timesurface_range = iter_/50.0 + 0.2;  // original 
        // double timesurface_range = (iter_)/60.0 + 0.2;
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 

        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);
        // cout << "default early " << default_value << endl; 

        // get t0 time surface of warpped image using latest angleAxis
        
        t1 = ros::Time::now();
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
        t2 = ros::Time::now();
        if(show_time_info)
            cout << "getWarpedEventImage time " << (t2-t1).toSec() * 2 << endl;  // 0.00691187 s
        

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
        ceres::CostFunction* cost_function = ResidualCostFunction::Create(
                                                    event_undis_Bundle.coord_3d,
                                                    eventBundle.time_delta, 
                                                    camera.eg_cameraMatrix,
                                                    interpolator_early_ptr);
        problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
    t2 = ros::Time::now();
    if(show_time_info)
        cout << "add residual time " << (t2-t1).toSec() << endl;  // 0.00168042

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 2;
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
        if(iter_ == 1)
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
            
            cout << "using " << angleAxis[0] << "," << angleAxis[1] << "," << angleAxis[2] 
            << ", residual size " << residual_vec.size() << ", sum_0: "<<  residual_sum_0 << ", sum_old: " <<residual_sum_old << endl; 
        }


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

        //     // cout << "using angle axis " << est_angleAxis.transpose() << endl;
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
        //     // cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;   
            
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

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
    cout << "Loss: " << 0 << ", est_angleAxis " << est_angleAxis.transpose() << endl;




}

