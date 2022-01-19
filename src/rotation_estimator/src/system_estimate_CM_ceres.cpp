
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>
#include <ceres/cubic_interpolation.h>
using namespace std;



/**
* \brief used for ceres to implement CM methods, automatie version. .
*/
struct ContrastCostFunction
{

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ContrastCostFunction(
            const Eigen::Matrix3Xd& coord_3d, 
            const Eigen::VectorXd& vec_delta_time, 
            const Eigen::Matrix3d& K):
            coord_3d_(coord_3d), vec_delta_time_(vec_delta_time), intrisic_(K)
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
        points_2D_T.resize(2, coord_3d_.cols());


        // taylor first 
        delta_points_T.row(0) = -ag[2]*coord_3d_T.row(1) + ag[1]*coord_3d_T.row(2);
        delta_points_T.row(1) =  ag[2]*coord_3d_T.row(0) - ag[0]*coord_3d_T.row(2);
        delta_points_T.row(2) = -ag[1]*coord_3d_T.row(0) + ag[0]*coord_3d_T.row(1);


        // taylor second 
        delta_second_points_T.row(0) = -ag[2]*delta_points_T.row(1) + ag[1]*delta_points_T.row(2);
        delta_second_points_T.row(1) =  ag[2]*delta_points_T.row(0) - ag[0]*delta_points_T.row(2);
        delta_second_points_T.row(2) = -ag[1]*delta_points_T.row(0) + ag[0]*delta_points_T.row(1);


        warped_3D_T.row(0) = coord_3d_T.row(0).array() - delta_points_T.row(0).array()*delta_time_T.transpose().array();// + delta_second_points_T.row(0).array()*0.5*delta_time_T.transpose().array()*delta_time_T.transpose().array();
        warped_3D_T.row(1) = coord_3d_T.row(1).array() - delta_points_T.row(1).array()*delta_time_T.transpose().array();// + delta_second_points_T.row(1).array()*0.5*delta_time_T.transpose().array()*delta_time_T.transpose().array();
        warped_3D_T.row(2) = coord_3d_T.row(2).array() - delta_points_T.row(2).array()*delta_time_T.transpose().array();// + delta_second_points_T.row(2).array()*0.5*delta_time_T.transpose().array()*delta_time_T.transpose().array();
        
        // cout << "coord_3d_T " << coord_3d_T.topLeftCorner(3,5) << endl;
        // cout << "delta_points_T  " << delta_points_T.topLeftCorner(3,5) << endl;
        // cout << "delta time " << delta_time_T.topLeftCorner(5,1).transpose()<< endl;
        // cout << "taylor first  " << warped_3D_T.topLeftCorner(3,5)<< endl;
                
        points_2D_T.row(0) = warped_3D_T.row(0).array()/warped_3D_T.row(2).array()*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_T.row(1) = warped_3D_T.row(1).array()/warped_3D_T.row(2).array()*T(intrisic_(1,1)) + T(intrisic_(1,2));

        // cout << " points_2D_T \n " << points_2D_T.topLeftCorner(2,3)<<endl;
        // cout << "intrinct " << intrisic_ << endl;

        // // accumulate && select inlier 
        Eigen::Matrix<T, -1, 240> warpped_images;  
        warpped_images.resize(180, 240);
        warpped_images.fill(T(0));

        // cout << "orignal mean " << warpped_images.mean() << endl;

        for(int i=0; i<points_2D_T.cols(); i++)
        {
            T x = points_2D_T(0,i), y = points_2D_T(1,i);

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

        // gaussian_blur(warpped_images);
        {
            int templates[25] = { 1, 4, 7, 4, 1,   
                                4, 16, 26, 16, 4,   
                                7, 26, 41, 26, 7,  
                                4, 16, 26, 16, 4,   
                                1, 4, 7, 4, 1 };        
            int height = 180, width = 240;
            for (int j=2;j<height-2;j++)  
            {  
                for (int i=2;i<width-2;i++)  
                {  
                    T sum = T(0);  
                    int index = 0;  
                    for ( int m=j-2; m<j+3; m++)  
                    {  
                        for (int n=i-2; n<i+3; n++)  
                        {  
                            sum +=  warpped_images(m, n) * T(templates[index++]) ;  
                        }  
                    }  
                    sum /= T(273);  
                    if (sum > T(255))  
                        sum = T(255);  
                    warpped_images(j, i) = sum;  
                }  
            }  
        }

        Eigen::Map<Eigen::Matrix<T, 1, -1>> warpped_images_vec(warpped_images.data(), 1, 180*240);
        Eigen::Matrix<T, 1, -1> warpped_images_vec_submean = warpped_images_vec.array() - warpped_images_vec.mean();

        residual[0] =  - T(1.0 / warpped_images_vec_submean.cols()) * warpped_images_vec_submean * warpped_images_vec_submean.transpose() ;
        // residual[0] =  - T(1.0 /(180*240)) * warpped_images_vec * warpped_images_vec.transpose() ; 
        residual[0] += T(10);

        // cout << "cols " << warpped_images_vec_submean.cols() << endl;
        // cout << "mean " << warpped_images_vec.mean() << endl;
        // cout << "multi " << warpped_images_vec * warpped_images_vec.transpose() << endl;
        // cout << "residual " << residual[0] << ", ag " <<ag[0] << ", "<< ag[1] << ", " <<ag[2] << endl;

        return true;
    }


    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Matrix3Xd& coord_3d, 
        const Eigen::VectorXd& vec_delta_time, 
        const Eigen::Matrix3d& K)
        {
            return new ceres::AutoDiffCostFunction<ContrastCostFunction,1, 3>(
                new ContrastCostFunction(coord_3d, vec_delta_time, K));
        }

    // inputs     
    Eigen::Matrix3Xd coord_3d_;
    Eigen::VectorXd vec_delta_time_; 
    Eigen::Matrix3d intrisic_; 

};

// struct ContrastCostFunction
// {

//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     ContrastCostFunction(
//             const Eigen::Vector3d& point, 
//             const double delta_time, 
//             const Eigen::Matrix3d& K, 
//             cv::Mat& event_count_mat,
//             const float mean_count):
//             point_(point), delta_time_(delta_time), intrisic_(K), event_count_mat_(event_count_mat), mean_count_(mean_count)
//     {
       
//     }

//     // operator 
//     template<typename T> 
//     bool operator()(const T* ag, T* residual) const
//     {
//         // warp unditort points 
//         int count = 0; 
//         Eigen::Matrix<T, 3, 1> point_T;
//         Eigen::Matrix<T, 3, 1> delta_first_T, delta_second_T;
//         Eigen::Matrix<T, 2, 1> point_2D_T;


//         // taylor first 
//         delta_first_T(0) = -ag[2]*T(point_(1)) + ag[1]*T(point_(2));
//         delta_first_T(1) =  ag[2]*T(point_(0)) - ag[0]*T(point_(2));
//         delta_first_T(2) = -ag[1]*T(point_(0)) + ag[0]*T(point_(1));

//         // taylor second 
//         delta_second_T(0) = -ag[2]*delta_first_T(1) + ag[1]*delta_first_T(2);
//         delta_second_T(1) =  ag[2]*delta_first_T(0) - ag[0]*delta_first_T(2);
//         delta_second_T(2) = -ag[1]*delta_first_T(0) + ag[0]*delta_first_T(1);

//         point_T(0) = T(point_(0)) - delta_first_T(0)*T(delta_time_); // + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
//         point_T(1) = T(point_(1)) - delta_first_T(1)*T(delta_time_); // + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
//         point_T(2) = T(point_(2)) - delta_first_T(2)*T(delta_time_); // + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);

//         // cout << "points "<< points(0) << ", "<< points(1) << endl;
        
//         point_2D_T(0) = point_T(0)/point_T(2)*T(intrisic_(0,0)) + T(intrisic_(0,2));
//         point_2D_T(1) = point_T(1)/point_T(2)*T(intrisic_(1,1)) + T(intrisic_(1,2));

//         // cout << " points_2D_T \n " << points_2D_T.topLeftCorner(2,3)<<endl;



//         int x_int = 0, y_int = 0;
//         if constexpr(std::is_same<T, double>::value)
//         {
//             x_int = ceres::floor(point_2D_T(0));
//             y_int = ceres::floor(point_2D_T(1));
//         }
//         else
//         {
//             x_int = ceres::floor(point_2D_T(0).a);
//             y_int = ceres::floor(point_2D_T(1).a);
//         }
            
//         if(x_int<0 || x_int>237 || y_int<0 || y_int>177) 
//         {
//             residual[0] = T(0);
//             return true;
//         }
        
//         T x = point_2D_T(0), y = point_2D_T(1);
//         T dx = x - T(x_int);
//         T dy = y - T(y_int);

//         T value_leftup    = T(event_count_mat_.at<float>(y_int, x_int)    );// - mean_count_) ; 
//         T value_leftdown  = T(event_count_mat_.at<float>(y_int+1, x_int)  );// - mean_count_) ;
//         T value_rightup   = T(event_count_mat_.at<float>(y_int, x_int+1)  );// - mean_count_) ;
//         T value_rightdown = T(event_count_mat_.at<float>(y_int+1, x_int+1));// - mean_count_) ;


//         residual[0] = T(20) - ceres::pow(value_leftup, 2)    * (T(1)-dx) * (T(1)-dy)
//                             - ceres::pow(value_leftdown, 2)  * (T(1)-dx) * dy
//                             - ceres::pow(value_rightup, 2)   * dx * (T(1)-dy)
//                             - ceres::pow(value_rightdown, 2) * dx *dy ;
        
//         // cout << " residual " << residual[0] << ", ag " <<ag[0] << ", "<< ag[1] << ", " <<ag[2] << endl;
//         // cout << " residual " << residual[0] << endl;     
//         // cout << " dx " << dx << endl;  
//         // cout << " event_count_mat_ " << cv::mean(event_count_mat_)[0] << endl;     
//         // cout << " delta_time_ " << delta_time_ << endl;     
//         // cout << " dy " << dy << endl;     
//         // cout << " point_2D_T " << point_2D_T << endl;     

//         // cout <<  "value_leftup " << value_leftup << endl;
//         // cout <<  "value_leftdown " << value_leftdown << endl;
//         // cout <<  "value_rightup " << value_rightup << endl;
//         // cout <<  "value_rightdown " << value_rightdown << endl;
        
//         return true;
//     }

//     // make ceres costfunction 
//     static ceres::CostFunction* Create(
//         const Eigen::Vector3d& point, 
//         const double delta_time, 
//         const Eigen::Matrix3d& K,
//         cv::Mat& event_count_mat, 
//         const float mean_count)
//         {
//             return new ceres::AutoDiffCostFunction<ContrastCostFunction,1, 3>(
//                 new ContrastCostFunction(point, delta_time, K, event_count_mat, mean_count));
//         }

//     // inputs     
//     Eigen::Vector3d point_;
//     double delta_time_;
//     Eigen::Matrix3d intrisic_; 
//     cv::Mat event_count_mat_; 
//     float mean_count_;
// };


/**
* \brief using time as distance, using ceres as optimizer, const gradient for each optimizing.
*/
void System::EstimateMotion_CM_ceres()
{
    cout << "time "<< eventBundle.first_tstamp.toSec() <<  ", total " << eventBundle.size <<endl;

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    // double angleAxis[3] = {0.01, 0.01, 0.01}; 
       
    // init problem 
    ceres::Problem problem; 
    // add residual 

    // for(int iter_=0; iter_< 100; iter_++)
    // {
    //     Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
    //     cv::Mat event_count_mat = getWarpedEventImage(eg_angleAxis, event_warpped_Bundle, PlotOption::F32C1_EVENT_COUNT);

    //     float mean_value = cv::mean(event_count_mat)[0];
    //     cout << "mean " << mean_value << endl;
    //     for(int residual_index=0; residual_index < eventBundle.size; residual_index+=10)
    //     {

    //         ceres::CostFunction* cost_function = ContrastCostFunction::Create(
    //                                                 event_undis_Bundle.coord_3d.col(residual_index),
    //                                                 eventBundle.time_delta(residual_index), 
    //                                                 camera.eg_cameraMatrix,
    //                                                 event_count_mat,
    //                                                 mean_value);
    //         problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
    //     }

    //     ceres::Solver::Options options;
    //     // options.minimizer_progress_to_stdout = true;
    //     options.num_threads = 1;
    //     // options.logging_type = ceres::SILENT;
    //     options.linear_solver_type = ceres::SPARSE_SCHUR;
    //     // options.use_nonmonotonic_steps = false;
    //     options.max_num_iterations = 10;
    //     // options.initial_trust_region_radius = 1;
    //     ceres::Solver::Summary summary; 
    //     ceres::Solve(options, &problem, &summary);
    //     cout << summary.BriefReport() << endl;

    //     cv::waitKey(30);

    // }

        // only debug 
        // getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, est_angleAxis); 

        ceres::CostFunction* cost_function = ContrastCostFunction::Create(
                                                event_undis_Bundle.coord_3d,
                                                eventBundle.time_delta,
                                                camera.eg_cameraMatrix);
        problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = yaml_ceres_iter_thread;
        // options.update_state_every_iteration = true;
        // options.initial_trust_region_radius = 0.1;
        // options.max_trust_region_radius = 0.1;

        // options.logging_type = ceres::SILENT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.use_nonmonotonic_steps = false;
        options.max_num_iterations = yaml_ceres_iter_num;
        // options.initial_trust_region_radius = 1;
        ceres::Solver::Summary summary; 

    problem.SetParameterLowerBound(&angleAxis[0],0,-20);
    problem.SetParameterUpperBound(&angleAxis[0],0,20);
    problem.SetParameterLowerBound(&angleAxis[0],1,-20);
    problem.SetParameterUpperBound(&angleAxis[0],1,20);
    problem.SetParameterLowerBound(&angleAxis[0],2,-20);
    problem.SetParameterUpperBound(&angleAxis[0],2,20);

        ceres::Solve(options, &problem, &summary);
        cout << summary.BriefReport() << endl;

    // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;
    // if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
    // {
    //     cout <<" No convergence Event bundle time " << eventBundle.first_tstamp.toSec() <<", size " << eventBundle.size << endl;
    //     // cout << summary.BriefReport() << endl;
    // }
    

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
    // cout << "Loss: " << 0 << ", est_angleAxis " <<est_angleAxis.transpose() << endl;

    getWarpedEventImage(-est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3); 

}

