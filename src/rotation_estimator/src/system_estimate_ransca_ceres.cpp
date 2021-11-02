
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
        const double delta_time_early, 
        const double delta_time_later, 
        const Eigen::Matrix3d& K, 
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_later_ptr):
            points_(points), delta_time_early_(delta_time_early), delta_time_later_(delta_time_later), 
            intrisic_(K), interpolator_early_ptr_(interpolator_early_ptr), interpolator_later_ptr_(interpolator_later_ptr)
    {
        // cout << "CM loss init :" << endl;
        
        // init timesurface and its surface 
        // Eigen::Matrix<double,3,-1> new_coord_3d;
        // warp<double>(ag_vec.data(), new_coord_3d);

        // cv::Mat image = cv::Mat::zeros({240,180}, CV_32FC1);
        // for(int i=0; i< 180; i++)
        // for(int j=0; j< 240; j++)
        // {
        //     double value = 0;
        //     interpolator_ptr_->Evaluate(i, j,  &value);
        //     image.at<float>(i,j) = value;
        // }
        // cv::Mat image_color;
        // cv::normalize(image, image_color, 255, 0,   cv::NORM_MINMAX, CV_8UC1);  
        // cv::applyColorMap(image_color, image_color, cv::COLORMAP_JET);
        // cv::imshow("mage " , image_color);
        // cv::waitKey(2000);

    }

    // operator 
    template<typename T> 
    bool operator()(const T* ag, T* residual) const
    {
        
        int count = 0; 
        Eigen::Matrix<T, 3, 1> delta_points_T, delta_second_points_T;
        Eigen::Matrix<T, 3, 1> points_early_T, points_later_T;
        Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;


        // taylor first 
        delta_points_T(0) = -ag[2]*T(points_(1)) + ag[1]*T(points_(2));
        delta_points_T(1) =  ag[2]*T(points_(0)) - ag[0]*T(points_(2));
        delta_points_T(2) = -ag[1]*T(points_(0)) + ag[0]*T(points_(1));

        // taylor second 
        delta_second_points_T(0) = -ag[2]*delta_points_T(1) + ag[1]*delta_points_T(2);
        delta_second_points_T(1) =  ag[2]*delta_points_T(0) - ag[0]*delta_points_T(2);
        delta_second_points_T(2) = -ag[1]*delta_points_T(0) + ag[0]*delta_points_T(1);

        points_early_T(0) = T(points_(0)) - delta_points_T(0)*T(delta_time_early_); // + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
        points_early_T(1) = T(points_(1)) - delta_points_T(1)*T(delta_time_early_); // + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
        points_early_T(2) = T(points_(2)) - delta_points_T(2)*T(delta_time_early_); // + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);

        points_later_T(0) = T(points_(0)) - delta_points_T(0)*T(delta_time_later_); // + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
        points_later_T(1) = T(points_(1)) - delta_points_T(1)*T(delta_time_later_); // + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
        points_later_T(2) = T(points_(2)) - delta_points_T(2)*T(delta_time_later_); // + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);
             
        // cout << "points "<< points(0) << ", "<< points(1) << endl;
        
        points_2D_early_T(0) = points_early_T(0)/points_early_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_early_T(1) = points_early_T(1)/points_early_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));
        points_2D_later_T(0) = points_later_T(0)/points_later_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_later_T(1) = points_later_T(1)/points_later_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));

        // cout << "points "<< points(0) << ", "<< points(1) << endl;

        // residual[0] = T(0);
        // T value;
        // for(int i=0; i<4; i++) // row
        // for(int j=0; j<4; j++) // col
        // {
        //     interpolator_ptr_->Evaluate(points_2D_T(1)+T(i-2), points_2D_T(0) + T(j-2),  &value);
        //     residual[0] += value;
        // }

        T early_loss = T(0), later_loss = T(0);

        interpolator_early_ptr_->Evaluate(points_2D_early_T(1), points_2D_early_T(0), &early_loss);
        interpolator_later_ptr_->Evaluate(points_2D_later_T(1), points_2D_later_T(0), &later_loss);
        
        residual[0] = early_loss + later_loss; 
        // cout << " residual " << residual[0] << ", ag " <<ag[0] << ", "<< ag[1] << ", " <<ag[2] << endl;

        return true;
    }

    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const double delta_time_later, 
        const Eigen::Matrix3d& K,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_later_ptr)
        {
            return new ceres::AutoDiffCostFunction<ResidualCostFunction,1, 3>(
                new ResidualCostFunction(points, delta_time_early, delta_time_later, K, interpolator_early_ptr, interpolator_later_ptr));
        }

    // inputs     
    Eigen::Vector3d points_;
    double delta_time_early_, delta_time_later_;
    Eigen::Matrix3d intrisic_; 

    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_early_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_later_ptr_;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};




/**
* \brief using time as distance, using ceres as optimizer, const gradient for each optimizing.
*/
void System::EstimateMotion_ransca_ceres(double sample_start, double sample_end)
{
    cout << "time "<< eventBundle.first_tstamp.toSec() <<  ", total " << 
            eventBundle.size << ", sample ratio " << sample_start<<"~"<< sample_end <<endl;

    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx; 
    int samples_count = std::min(20000,int(event_undis_Bundle.size * (sample_end-sample_start)));
    getSampledVec(vec_sampled_idx, samples_count, sample_start, sample_end);

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    
    for(int iter_=0; iter_< 50; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        double timesurface_range = iter_/100.0 + 0.2;
        // double timesurface_range = 0.2;
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 
        cv::Mat cv_later_timesurface = cv::Mat(180,240, CV_32FC1); 


        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        double default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(default_value);
        default_value = eventBundle.time_delta(eventBundle.size-1) - eventBundle.time_delta(int(eventBundle.size*(1-timesurface_range)));
        cv_later_timesurface.setTo(default_value);

        // get t0 time surface of warpped image using latest angleAxis
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle); 
        // get early timesurface
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = event_warpped_Bundle.coord.col(i)[0], sampled_y = event_warpped_Bundle.coord.col(i)[1]; 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 

        // get t1 time surface of warpped image
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE, true);
        // get later timesurface
        for(int i= event_warpped_Bundle.size*(1-timesurface_range); i< event_warpped_Bundle.size; i++)
        {
            
            int sampled_x = event_warpped_Bundle.coord.col(i)[0], sampled_y = event_warpped_Bundle.coord.col(i)[1]; 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // linear add TODO improve to module 
                cv_later_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(event_warpped_Bundle.size-1) - eventBundle.time_delta(i);  
        } 
        // ceres::Grid2D<double,1> grid(line_grid.data(), 0, 180, 0, 240);
        // auto* interpolator_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid);  
            
            // {
            //     // visualize timesurface 
            //     cv::Mat cv_earlier_timesurface_8U, cv_earlier_timesurface_color; 
            //     cv::normalize(cv_earlier_timesurface, cv_earlier_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
            //     // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
            //     cv::applyColorMap(cv_earlier_timesurface_8U, cv_earlier_timesurface_color, cv::COLORMAP_JET);
            //     cv::imshow("timesurface_early", cv_earlier_timesurface_color);
            //     cv::waitKey(1);
            // }
            // {
            //     // visualize timesurface 
            //     cv::Mat cv_later_timesurface_8U, cv_later_timesurface_color; 
            //     cv::normalize(cv_later_timesurface, cv_later_timesurface_8U, 255, 0, cv::NORM_MINMAX , CV_8UC1 );
            //     // cv_earlier_timesurface.convertTo(cv_earlier_timesurface_8U, CV_8UC1);
            //     cv::applyColorMap(cv_later_timesurface_8U, cv_later_timesurface_color, cv::COLORMAP_JET);
            //     cv::imshow("timesurface_later", cv_later_timesurface_color);
            //     cv::waitKey(1);
            // }

        // TODO add gaussian on cv_earlier_timesurface

        // get timesurface in ceres 
        vector<double> line_grid_early(180*240, 0), line_grid_later(180*240, 0);
        for (int row = 0; row < 179; row++)
        {
            for (int col = 0; col < 239; col++)
            {
                    line_grid_early[row * 240 + col] = cv_earlier_timesurface.at<float>(row, col); 
                    line_grid_later[row * 240 + col] = cv_later_timesurface.at<float>(row, col); 
                    if(col % 10 == 0 && row %20 == 0)
                    {
                        // cout << "timesurface (" << row << ", " << col << ") = " << line_grid[row * 240 + col] << endl;
                    }
            }
        }
        ceres::Grid2D<double,1> grid_early(line_grid_early.data(), 0, 180, 0, 240);
        ceres::Grid2D<double,1> grid_later(line_grid_later.data(), 0, 180, 0, 240);
        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_early);
        auto* interpolator_later_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_later);  

        
        // init problem 
        ceres::Problem problem; 
        // add residual 
        for(int loop_temp =0; loop_temp <vec_sampled_idx.size(); loop_temp++)
        {
            size_t sample_idx = vec_sampled_idx[loop_temp];
            double early_time =  eventBundle.time_delta(sample_idx);
            double later_time =  eventBundle.time_delta(sample_idx) - eventBundle.time_delta(event_warpped_Bundle.size-1);

            ceres::CostFunction* cost_function = ResidualCostFunction::Create(
                                                    event_undis_Bundle.coord_3d.col(sample_idx),
                                                    early_time, later_time, 
                                                    camera.eg_cameraMatrix,
                                                    interpolator_early_ptr, interpolator_later_ptr);

            problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
        }
    
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 14;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.use_nonmonotonic_steps = true;
        options.max_num_iterations = 10;
        // options.initial_trust_region_radius = 1;

        ceres::Solver::Summary summary; 
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << endl;

        // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;
        if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
        {
            cout << "   iter " << iter_ <<  ", No convergence Event bundle time " << eventBundle.first_tstamp.toSec() <<", size " << eventBundle.size << endl;
            // cout << summary.BriefReport() << endl;
        }
    }

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
    cout << "Loss: " << 0 << ", est_angleAxis " <<est_angleAxis.transpose() << endl;


    bool visual_hot_c3 = false; 
    if(visual_hot_c3)
    {

        // cout << "using angle axis " << est_angleAxis.transpose() << endl;
        // visualize warp events and get a timesurface image with indexs 
        {
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

