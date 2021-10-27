
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

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
        const double delta_time, 
        const double delta_time_ratio,
        const Eigen::Matrix3d& K, 
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_ptr):
            points_(points), delta_time_(delta_time), intrisic_(K), interpolator_ptr_(interpolator_ptr)
    {
        delta_time_ratio_ = delta_time_ratio;
    }

    // operator 
    template<typename T> 
    bool operator()(const T* ag, T* residual) const
    {
        
        // Eigen::Matrix<T, 3, 1> delta_points_T, delta_second_points_T;
        // Eigen::Matrix<T, 3, 1> points_T;
        // Eigen::Matrix<T, 2, 1> points_2D_T;
        // // taylor first 
        // delta_points_T(0) = -ag[2]*T(points_(1)) + ag[1]*T(points_(2));
        // delta_points_T(1) =  ag[2]*T(points_(0)) - ag[0]*T(points_(2));
        // delta_points_T(2) = -ag[1]*T(points_(0)) + ag[0]*T(points_(1));

        // // taylor second 
        // delta_second_points_T(0) = -ag[2]*delta_points_T(1) + ag[1]*delta_points_T(2);
        // delta_second_points_T(1) =  ag[2]*delta_points_T(0) - ag[0]*delta_points_T(2);
        // delta_second_points_T(2) = -ag[1]*delta_points_T(0) + ag[0]*delta_points_T(1);

        // points_T(0) = T(points_(0)) + delta_points_T(0)*T(delta_time_) + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
        // points_T(1) = T(points_(1)) + delta_points_T(1)*T(delta_time_) + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
        // points_T(2) = T(points_(2)) + delta_points_T(2)*T(delta_time_) + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);
        // points_2D_T(0) = points_T(0)/points_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        // points_2D_T(1) = points_T(1)/points_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));
        // interpolator_ptr_->Evaluate(points_2D_T(1), points_2D_T(0),  &residual[0]);

        // reference https://zh.wikipedia.org/wiki/%E7%BD%97%E5%BE%B7%E9%87%8C%E6%A0%BC%E6%97%8B%E8%BD%AC%E5%85%AC%E5%BC%8F

        T norm = ceres::sqrt(ag[0]*ag[0]+ag[1]*ag[1]+ag[2]*ag[2]);

        if(norm < T(0.0001)) norm = T(0.0001); // prevent zero 
        Eigen::Matrix<T,3,1> axis = {ag[0]/norm, ag[1]/norm, ag[2]/norm};
        Eigen::Matrix<T,3,1> original = {T(points_(0)), T(points_(1)), T(points_(2))};

        T angle = - norm * delta_time_ratio_; // rotation for each points, minus means t2->t1, 
        Eigen::Matrix<T,3,1> first = original * ceres::cos(angle);
        Eigen::Matrix<T,3,1> second; 
            ceres::CrossProduct(axis.data(), original.data(), second.data());
            second = second * ceres::sin(angle);
    
        Eigen::Matrix<T,3,1> third = axis * ceres::DotProduct(axis.data(),  original.data()) * (T(1)-ceres::cos(angle));
        Eigen::Matrix<T,3,1> final = first + second + third;
        // cout << "rodrigues " << final << endl;


        Eigen::Matrix<T,2,1> final_2d; 
        final_2d(0) = final(0)/final(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        final_2d(1) = final(1)/final(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));

        interpolator_ptr_->Evaluate(final_2d(1), final_2d(0),  &residual[0]);
        
        // residual[0] = T(0);
        // T value;
        // for(int i=0; i<4; i++) // row
        // for(int j=0; j<4; j++) // col
        // {
        //     interpolator_ptr_->Evaluate(points_2D_T(1)+T(i-2), points_2D_T(0) + T(j-2),  &value);
        //     residual[0] += value;
        // }

        // cout << " residual " << residual[0] << ", ag " <<ag[0] << ", "<< ag[1] << ", " <<ag[2] << endl;

        return true;
    }

    // TODO FIXME reference to CVPR2019 for gaussian smoother. 
    

    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Vector3d& points, 
        const double delta_time, 
        const double delta_time_ratio,
        const Eigen::Matrix3d& K,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_ptr)
        {
            return new ceres::AutoDiffCostFunction<ResidualCostFunction,1, 3>(
                new ResidualCostFunction(points, delta_time, delta_time_ratio, K, interpolator_ptr));
        }

    // inputs     
    Eigen::Vector3d points_;
    double delta_time_;
    double delta_time_ratio_; 
    Eigen::Matrix3d intrisic_; 

    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_ptr_;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};




/**
* \brief using time as distance, using ceres as optimizer, const gradient for each optimizing.
*/
void System::EstimateMotion_ransca_ceres(double sample_start, double sample_end)
{
    
    cout << "------------- time "<< eventBundle.first_tstamp.toSec() <<  ", total " << eventBundle.size << ", sample ratio " << sample_start<<"~"<< sample_end <<" ----" <<endl;
    
    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx; 
    int samples_count = std::min(20000,int(event_undis_Bundle.size * (sample_end-sample_start)));
    getSampledVec(vec_sampled_idx, samples_count, sample_start, sample_end);

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    double bundle_total_time = (eventBundle.last_tstamp -eventBundle.first_tstamp).toSec();
    
    for(int iter_=0; iter_< 10; iter_++)
    {
        // get timesurface earlier         
        double timesurface_range = iter_/20.0 + 0.2;
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 
        cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        double default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(default_value);
        // using event_warpped_Bundle not eventBundle !!! 
        // get t0 time surface of warpped image !!

        {
            // visualize 
            cv::Mat cv_earlier_timesurface_8U, cv_earlier_timesurface_color; 
            cv::normalize(cv_earlier_timesurface, cv_earlier_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
            // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
            cv::applyColorMap(cv_earlier_timesurface_8U, cv_earlier_timesurface_color, cv::COLORMAP_JET);
            cv::imshow("timesurface", cv_earlier_timesurface_color);
            cv::waitKey(1);
        }

        est_angleAxis = Eigen::Vector3d(angleAxis[0]/bundle_total_time, angleAxis[1]/bundle_total_time, angleAxis[2]/bundle_total_time);
        getWarpedEventImage(est_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE); // init timesurface  

        for(int i=0; i<(event_warpped_Bundle.size*timesurface_range); i++)
        {
            
            int sampled_x = event_warpped_Bundle.coord.col(i)[0], sampled_y = event_warpped_Bundle.coord.col(i)[1]; 
            // int sampled_x = eventBundle.coord.col(i)[0], sampled_y = eventBundle.coord.col(i)[1]; 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            if(visited_map.at<uchar>(sampled_y, sampled_x) > 0) continue;   // visited 
            visited_map.at<uchar>(sampled_y, sampled_x) = 1;
            // linear 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);              
        } 

        // get timesurface in ceres 
        vector<double> line_grid(180*240, 0);
        for (int row = 0; row < 179; row++)
        {
            for (int col = 0; col < 239; col++)
            {
                    line_grid[row * 240 + col] =
                        cv_earlier_timesurface.at<float>(row, col); 

                    if(col % 10 == 0 && row %20 == 0)
                    {
                        // cout << "timesurface (" << row << ", " << col << ") = " << line_grid[row * 240 + col] << endl;
                    }
            }
        }
        ceres::Grid2D<double,1> grid(line_grid.data(), 0, 180, 0, 240);
        auto* interpolator_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid);  
        

        // init problem 
        ceres::Problem problem; 

        for(int loop_temp =0; loop_temp <vec_sampled_idx.size(); loop_temp++)
        {
            size_t sample_idx = vec_sampled_idx[loop_temp];
            ceres::CostFunction* cost_function = ResidualCostFunction::Create(
                                                    event_undis_Bundle.coord_3d.col(sample_idx),
                                                    eventBundle.time_delta(sample_idx), 
                                                    eventBundle.time_delta(sample_idx) / bundle_total_time, 
                                                    camera.eg_cameraMatrix,
                                                    interpolator_ptr);

            problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
        }
    
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.use_nonmonotonic_steps = true;
        options.max_num_iterations = 10 + iter_;
        // options.initial_trust_region_radius = 1;

        ceres::Solver::Summary summary; 
        ceres::Solve(options, &problem, &summary);

        // cout << summary.BriefReport() << endl;
        // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;

        if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
        {
            cout <<"Iter " << iter_ << " No convergence Event bundle time " << eventBundle.first_tstamp.toSec() << endl;
            cout << summary.BriefReport() << endl;
        }


    }

    est_angleAxis = Eigen::Vector3d(angleAxis[0]/bundle_total_time,angleAxis[1]/bundle_total_time,angleAxis[2]/bundle_total_time);


    bool visual_hot_c3 = false; 
    if(visual_hot_c3)
    {

        {   // show timesurface 
            cv::Mat cv_timesurface; 
            cv_timesurface = getImageFromBundle(event_undis_Bundle, PlotOption::TIME_SURFACE);  
            cv::normalize(cv_timesurface, hot_image_C1, 0,255, cv::NORM_MINMAX, CV_8UC1);
            cv::cvtColor(hot_image_C1, hot_image_C3, cv::COLOR_GRAY2BGR);
        }

        int sample_green = 1;
        for(int i=0; i<vec_sampled_idx.size() ; i++)
        {
            // if(i<10) cout << "sampling " << sample_idx << endl;

            // viusal sample 
            if(i%sample_green != 0) continue;
            int x = int(event_undis_Bundle.coord(0, vec_sampled_idx[i]));
            int y = int(event_undis_Bundle.coord(1, vec_sampled_idx[i]));
            if(x>239 || y>179 || x<0 ||y<0) continue;
            hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,255,0);   // green of events before warp 
        }

        // compare with gt
        // cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;   
        
        // plot earlier timestamps oringe 22，G：07，B：201
        for(int i=0; i<eventBundle.size/15 ; i++)
        {
            // viusal sample 
            // cout << "hello " << endl;
            if(i%sample_green != 0) continue;
            int x = int(event_undis_Bundle.coord(0, i));
            int y = int(event_undis_Bundle.coord(1, i));
            if(x>239 || y>179 || x<0 ||y<0) continue;
            hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 165, 255);   // bottom events earlier  
        }

        // visualize est
        int sample_red = 1;
        for(int i=0; i<vec_sampled_idx.size(); i++)
        {
            // viusal sample 
            if(i%sample_red != 0) continue;
            int x = int(event_warpped_Bundle.coord(0, vec_sampled_idx[i])); 
            int y = int(event_warpped_Bundle.coord(1, vec_sampled_idx[i]));
            if(x>239 || y>179 || x<0 ||y<0) continue;
            hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);   // red after warp
            // cout << "all inlier red" << endl;
        }
    }

}

