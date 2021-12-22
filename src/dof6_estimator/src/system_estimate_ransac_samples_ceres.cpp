
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>
#include <algorithm>
#include <ceres/cubic_interpolation.h>
using namespace std;



/**
* \brief used for ceres to implement CM methods, automatie version. .
*/
struct ResidualDistanceCostFunction
{

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ResidualDistanceCostFunction(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const double delta_time_later, 
        const Eigen::Matrix3d& K, 
        cv::Mat& cv_earlier_mat,
        cv::Mat& cv_later_mat):
            points_(points), delta_time_early_(delta_time_early), delta_time_later_(delta_time_later), 
            intrisic_(K), cv_earlier_mat_(cv_earlier_mat), cv_later_mat_(cv_later_mat)
    {
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

        /* cv mat version  */
        {
            int early_x_int = 0, early_y_int = 0, later_x_int = 0, later_y_int = 0;
            if constexpr(std::is_same<T, double>::value)
            {
                early_x_int = ceres::floor(points_2D_early_T(0));
                early_y_int = ceres::floor(points_2D_early_T(1));

                later_x_int = ceres::floor(points_2D_later_T(0));
                later_y_int = ceres::floor(points_2D_later_T(1));
            }
            else
            {
                early_x_int = ceres::floor(points_2D_early_T(0).a);
                early_y_int = ceres::floor(points_2D_early_T(1).a);

                later_x_int = ceres::floor(points_2D_later_T(0).a);
                later_y_int = ceres::floor(points_2D_later_T(1).a);
            }
            
            // init loss 
            residual[0] = T(0); residual[1] = T(0);
            residual[2] = T(0); residual[3] = T(0);
            
            // inlier test 
            if(early_x_int < 3 || early_x_int > 235 || early_y_int < 1 || early_y_int > 175 
                || later_x_int < 3 || later_x_int > 235 || later_y_int < 1 || later_y_int > 175) return true;

            // make sure got target value, search all 9 neighbors 
            {
                // outside target, search 8 neighbors  
                T early_dis_x = T(10), early_dis_y = T(10), later_dis_x = T(10), later_dis_y = T(10);
                for(int i=-2; i<3; i++)
                for(int j=-2; j<3; j++)
                {
                    early_dis_x = ceres::fmin(
                        ceres::abs(points_2D_early_T(0) - T(cv_earlier_mat_.at<cv::Vec2f>(early_y_int + i, early_x_int + j)[0])), early_dis_x);
                    early_dis_y = ceres::fmin(
                        ceres::abs(points_2D_early_T(1) - T(cv_earlier_mat_.at<cv::Vec2f>(early_y_int + i, early_x_int + j)[1])), early_dis_y);
                    later_dis_x = ceres::fmin(
                        ceres::abs(points_2D_later_T(0) - T(cv_later_mat_.at<cv::Vec2f>(later_y_int + i, later_x_int + j)[0])), later_dis_x);
                    later_dis_y = ceres::fmin(
                        ceres::abs(points_2D_later_T(1) - T(cv_later_mat_.at<cv::Vec2f>(later_y_int + i, later_x_int + j)[1])), later_dis_y);
                }

                residual[0] = early_dis_x;           
                residual[1] = early_dis_y;           
                residual[2] = later_dis_x;           
                residual[3] = later_dis_y;           
                
                // cout << "points_2D early int " << early_x_int << ", " << early_y_int << endl;
                // cout << "points_2D later int " << later_x_int << ", " << later_y_int << endl;
                // cout << "cv_earlier_mat_ " << cv_earlier_mat_.at<cv::Vec2f>(early_y_int, early_x_int)[0] << ", " << cv_earlier_mat_.at<cv::Vec2f>(early_y_int, early_x_int)[1] << endl;                
                // cout << "cv_later_mat_ " << cv_later_mat_.at<cv::Vec2f>(later_y_int, later_x_int)[0] << ", " << cv_later_mat_.at<cv::Vec2f>(later_y_int, later_x_int)[1] << endl;                
            }
        }   

        // cout << "ag " <<ag[0] << ", "<< ag[1] << ", " <<ag[2] << endl;
        // cout << " residual " << residual[0] << endl;
        // cout << " residual " << residual[1] << endl;
        // cout << " residual " << residual[2] << endl;
        // cout << " residual " << residual[3] << endl;

        return true;
    }

    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Vector3d& points, 
        const double delta_time_early, 
        const double delta_time_later, 
        const Eigen::Matrix3d& K,
        cv::Mat& cv_earlier_mat,
        cv::Mat& cv_later_mat)
        {
            return new ceres::AutoDiffCostFunction<ResidualDistanceCostFunction, 4, 3>(
                new ResidualDistanceCostFunction(points, delta_time_early, delta_time_later, K, 
                    cv_earlier_mat, cv_later_mat));
        }

    // inputs     
    Eigen::Vector3d points_;
    double delta_time_early_, delta_time_later_;
    Eigen::Matrix3d intrisic_; 

    cv::Mat cv_earlier_mat_;
    cv::Mat cv_later_mat_;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};




/**
* \brief using earliest pixels matching problem, using ceres as optimizer, const gradient for each optimizing.
*/
void System::EstimateMotion_ransca_samples_ceres(double sample_start, double sample_end)
{
    cout << "time "<< eventBundle.first_tstamp.toSec() <<  ", total " << 
            eventBundle.size << ", sample ratio " << sample_start<<"~"<< sample_end <<endl;

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    
    // choose init velocity, test whether using last_est or {0,0,0}, 
    // TODO add function to calculate residual 
    {
        // double residual1 = 0, residual2 = 0;
        // DeriveTimeErrAnalyticRansacBottom(est_angleAxis, vec_sampled_idx, residual1);
        // DeriveTimeErrAnalyticRansacBottom(Eigen::Vector3d(0,0,0), vec_sampled_idx, residual2);
        // if(residual1 > residual2)
        // {
        //     cout << "at time " <<eventBundle.first_tstamp.toSec() << ", using {0,0,0} "<< endl;
        //     est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 
        // }
        cout << "need resudial function" << endl;
        // CM, timeresidual, confidence template, python is better choice rosl
    }


    for(int iter_=0; iter_< 50; iter_++)
    {

        // sample events 
        std::vector<int> vec_sampled_idx; 
        // int samples_count = std::min(20000, int(event_undis_Bundle.size * (sample_end-sample_start)));
        int samples_count = 20000;
        getSampledVec(vec_sampled_idx, samples_count, sample_start, sample_end);

        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        // cout << "before angleaxis " << eg_angleAxis.transpose() << endl;
         
        double timesurface_range = 0.3; 
        // double timesurface_range = 0.2;

        cv::Mat cv_earlier_mat = cv::Mat(camera.height, camera.width, CV_32FC2); // store earliest events (x, y) 
        cv::Mat cv_later_mat =   cv::Mat(camera.height, camera.width, CV_32FC2); 
        cv_earlier_mat = cv::Vec2f(0,0); // init very important 
        cv_later_mat = cv::Vec2f(0,0);

        // get t0 time surface of warpped image using latest angleAxis
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle);

        // DELETE later 
        cv::Mat temp_img;
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle).convertTo(temp_img, CV_32FC3);
        cv::imshow("opti", temp_img);
        cv::waitKey(500);

        // get early timesurface
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = event_warpped_Bundle.coord.col(i)[0], sampled_y = event_warpped_Bundle.coord.col(i)[1]; 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // TODO weight with other events
            cv_earlier_mat.at<cv::Vec2f>(sampled_y, sampled_x)[0] = event_warpped_Bundle.coord.col(i)[0];  
            cv_earlier_mat.at<cv::Vec2f>(sampled_y, sampled_x)[1] = event_warpped_Bundle.coord.col(i)[1];  

            // if(i % 10 == 0) // test well
            // {
            //     cout << "ori " <<  event_warpped_Bundle.coord.col(i)[0] << ", " << event_warpped_Bundle.coord.col(i)[1] << endl;
            //     cout << "int " <<  sampled_x << ", " << sampled_y << endl;
            //     cout << "cv " <<  cv_earlier_mat.at<cv::Vec2f>(sampled_y, sampled_x)[0] << ", " << cv_earlier_mat.at<cv::Vec2f>(sampled_y, sampled_x)[1] << endl;
            // }
        } 

        // get t1 time surface of warpped image
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE, true);
        // get later timesurface
        for(int i= event_warpped_Bundle.size*(1-timesurface_range); i< event_warpped_Bundle.size; i++)
        {
            
            int sampled_x = event_warpped_Bundle.coord.col(i)[0], sampled_y = event_warpped_Bundle.coord.col(i)[1]; 

            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
            // TODO weight with other events
            cv_later_mat.at<cv::Vec2f>(sampled_y, sampled_x)[0] = event_warpped_Bundle.coord.col(i)[0];  
            cv_later_mat.at<cv::Vec2f>(sampled_y, sampled_x)[1] = event_warpped_Bundle.coord.col(i)[1];  
        } 

        // TODO add gaussian on cv_earlier_timesurface

        
        // init problem 
        ceres::Problem problem; 
        // add residual 
        for(int loop_temp =0; loop_temp <vec_sampled_idx.size(); loop_temp++)
        {
            size_t sample_idx = vec_sampled_idx[loop_temp];
            double early_time =  eventBundle.time_delta(sample_idx);
            double later_time =  eventBundle.time_delta(sample_idx) - eventBundle.time_delta(event_warpped_Bundle.size-1);

            ceres::CostFunction* cost_function = ResidualDistanceCostFunction::Create(
                                                    event_undis_Bundle.coord_3d.col(sample_idx),
                                                    early_time, later_time, 
                                                    camera.eg_cameraMatrix,
                                                    cv_earlier_mat, cv_later_mat);

            problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
        }
    
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.use_nonmonotonic_steps = true;
        options.max_num_iterations = 1;
        // options.initial_trust_region_radius = 1;
        problem.SetParameterLowerBound(&angleAxis[0],0,-20);
        problem.SetParameterLowerBound(&angleAxis[0],1,-20);
        problem.SetParameterLowerBound(&angleAxis[0],2,-20);
        problem.SetParameterUpperBound(&angleAxis[0],0, 20);
        problem.SetParameterUpperBound(&angleAxis[0],1, 20);
        problem.SetParameterUpperBound(&angleAxis[0],2, 20);

        {
            double cost = 0;
            vector<double> residual_vec; 
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec, nullptr, nullptr);
            cout << "using " << angleAxis[0] << "," << angleAxis[1] << "," << angleAxis[2] 
            << ", residual size " << residual_vec.size() << ", sum " << std::accumulate(residual_vec.begin(), residual_vec.end(), 0) << endl; 
        }

        ceres::Solver::Summary summary; 
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << endl;

        // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;
        // if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
        // {
        //     cout << "   iter " << iter_ <<  ", No convergence Event bundle time " << eventBundle.first_tstamp.toSec() <<", size " << eventBundle.size << endl;
        //     // cout << summary.BriefReport() << endl;
        // }

    }

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
    cout << "Loss: " << 0 << ", est_angleAxis " <<est_angleAxis.transpose() << endl;

    // bool visual_hot_c3 = false; 
    // if(visual_hot_c3)
    // {

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

