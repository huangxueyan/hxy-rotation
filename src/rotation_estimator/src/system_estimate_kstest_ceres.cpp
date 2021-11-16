
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

#include <ceres/cubic_interpolation.h>
using namespace std;



/**
* \brief used for ceres to implement CM methods, automatie version. .
*/
struct KsCostFunction
{

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KsCostFunction(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const double delta_time_later, 
        const Eigen::Matrix3d& K, 
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator_later_ptr):
            points_(points), delta_time_early_(delta_time_early), delta_time_later_(delta_time_later), 
            intrisic_(K), interpolator_early_ptr_(interpolator_early_ptr), interpolator_later_ptr_(interpolator_later_ptr)
    {
    }

    // operator 
    template<typename T> 
    bool operator()(const T* ag, T* residual) const
    {
        // warp unditort points 
        int count = 0; 
        Eigen::Matrix<T, 3, -1> coord_3d_T = coord_3d_.cast<T>();
        Eigen::Matrix<T, -1, 1> delta_time_T = vec_delta_time_.cast<T>();
        
        Eigen::Matrix<T, 3, -1> delta_points_T, delta_second_points_T, warped_3D_T;

        Eigen::Matrix<T, 2, -1> points_2D_T;

        delta_points_T.resize(3, coord_3d_.cols());
        delta_second_points_T.resize(3, coord_3d_.cols());
        warped_3D_T.resize(3, coord_3d_.cols());
        points_2D_T.resize(3, coord_3d_.cols());


        // taylor first 
        delta_points_T.row(0) = -ag[2]*coord_3d_T.row(1) + ag[1]*coord_3d_T.row(2);
        delta_points_T.row(1) =  ag[2]*coord_3d_T.row(0) - ag[0]*coord_3d_T.row(2);
        delta_points_T.row(2) = -ag[1]*coord_3d_T.row(0) + ag[0]*coord_3d_T.row(1);

        // taylor second 
        delta_second_points_T.row(0) = -ag[2]*delta_points_T.row(1) + ag[1]*delta_points_T.row(2);
        delta_second_points_T.row(1) =  ag[2]*delta_points_T.row(0) - ag[0]*delta_points_T.row(2);
        delta_second_points_T.row(2) = -ag[1]*delta_points_T.row(0) + ag[0]*delta_points_T.row(1);


        warped_3D_T.row(0) = - delta_points_T.row(0).array()*delta_time_T.transpose().array(); // + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
        warped_3D_T.row(0) += coord_3d_T.row(0);
        warped_3D_T.row(1) = - delta_points_T.row(1).array()*delta_time_T.transpose().array(); // + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
        warped_3D_T.row(1) += coord_3d_T.row(1);
        warped_3D_T.row(2) = - delta_points_T.row(2).array()*delta_time_T.transpose().array(); // + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);
        warped_3D_T.row(2) += coord_3d_T.row(2);
        

        // warped_3D_T.row(0) = coord_3d_T.row(0).colwise() - delta_points_T.row(0).array()*delta_time_T.transpose().array(); // + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
        // warped_3D_T.row(1) = coord_3d_T.row(1) - delta_points_T.row(1).array()*delta_time_T.transpose().array(); // + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
        // warped_3D_T.row(2) = coord_3d_T.row(2) - delta_points_T.row(2).array()*delta_time_T.transpose().array(); // + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);
                
        points_2D_T.row(0) = warped_3D_T.row(0).array()/warped_3D_T.row(2).array()*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_T.row(1) = warped_3D_T.row(1).array()/warped_3D_T.row(2).array()*T(intrisic_(1,1)) + T(intrisic_(1,2));

        // cout << " points_2D_T \n " << points_2D_T.topLeftCorner(2,3)<<endl;


        // // accumulate && select inlier 
        Eigen::Matrix<T, -1, -1> warpped_images = cur_iwe.cast<T>();  
        // warpped_images.resize(180, 240);
        for(int i=0; i<points_2D_T.cols(); i++)
        {
            T x = points_2D_T(i,0), y = points_2D_T(i,1);

            int x_int = 0, y_int = 0;
            if constexpr(std::is_same<T, double>::value)
            {
                x_int = ceres::floor(x);
                y_int = ceres::floor(y);
            }
            else
            {
                x_int = ceres::floor(x.a);
                y_int = ceres::floor(y.a);
            }
             
            if(x_int<0 || x_int>237 || y_int<0 || y_int>177) continue;
            
            T dx = x - T(x_int);
            T dy = y - T(y_int);

            warpped_images(y_int, x_int)     += (T(1)-dx) * (T(1)-dy);
            warpped_images(y_int+1, x_int)   += (T(1)-dx) * dy;
            warpped_images(y_int, x_int+1)   += dx * (T(1)-dy);
            warpped_images(y_int+1, x_int+1) += dx*dy;
        }

        residual[0] = T(0); 
        for(int i=0; i<180; i++)
        for(int j=0; j<240; j++)
        {
            if()
            iwe(i,j)
        }
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
            return new ceres::AutoDiffCostFunction<KsCostFunction,1, 3>(
                new KsCostFunction(points, delta_time_early, delta_time_later, K, interpolator_early_ptr, interpolator_later_ptr));
        }

    // inputs     
    Eigen::Vector3d points_;
    double delta_time_early_, delta_time_later_;
    Eigen::Matrix3d intrisic_; 
    Eigen::Matrix<double, 180,240> cur_iwe; 
    Eigen::Matrix<double, 180,240> iwe; 


    // ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_early_ptr_;
    // ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_later_ptr_;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};




/**
* \brief using ks in cumulative probability against uniform distribution .
*/
void System::EstimateMotion_KS_ceres()
{
    cout << "time "<< eventBundle.first_tstamp.toSec() <<  ", total " << eventBundle.size << endl;

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    

    for(int iter_=0; iter_< 50; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        // get t0 time surface of warpped image using latest angleAxis
        cv::Mat event_count_mat = getWarpedEventImage(eg_angleAxis, event_warpped_Bundle, PlotOption::F32C1_EVENT_COUNT); 
        


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

            ceres::CostFunction* cost_function = KsCostFunction::Create(
                                                    event_undis_Bundle.coord_3d.col(sample_idx),
                                                    early_time, later_time, 
                                                    camera.eg_cameraMatrix,
                                                    interpolator_early_ptr, interpolator_later_ptr);

            problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
        }
    
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 4;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.use_nonmonotonic_steps = true;
        options.max_num_iterations = 10;
        // options.initial_trust_region_radius = 1;
        problem.SetParameterLowerBound(&angleAxis[0],0,-20);
        problem.SetParameterLowerBound(&angleAxis[0],1,-20);
        problem.SetParameterLowerBound(&angleAxis[0],2,-20);
        problem.SetParameterUpperBound(&angleAxis[0],0, 20);
        problem.SetParameterUpperBound(&angleAxis[0],1, 20);
        problem.SetParameterUpperBound(&angleAxis[0],2, 20);

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

