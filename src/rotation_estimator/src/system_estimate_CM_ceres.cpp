
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
        Eigen::Matrix<T, -1, -1> warpped_images;  
        warpped_images.resize(180, 240);
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


        Eigen::Map<Eigen::Matrix<T, 1, -1>> warpped_images_vec(warpped_images.data(), 1, 180*240);
        Eigen::Matrix<T, 1, -1> warpped_images_vec_submean = warpped_images_vec.array() - warpped_images_vec.mean();

        residual[0] =  - T(1.0 / warpped_images_vec_submean.cols()) * warpped_images_vec_submean * warpped_images_vec_submean.transpose() ;
        
        cout << " residual " << residual[0] << ", ag " <<ag[0] << ", "<< ag[1] << ", " <<ag[2] << endl;
        residual[0] += T(100);

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




/**
* \brief using time as distance, using ceres as optimizer, const gradient for each optimizing.
*/
void System::EstimateMotion_CM_ceres()
{
    cout << "time "<< eventBundle.first_tstamp.toSec() <<  ", total " << eventBundle.size <<endl;

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    
   
    // init problem 
    ceres::Problem problem; 
    // add residual 

    ceres::CostFunction* cost_function = ContrastCostFunction::Create(
                                            event_undis_Bundle.coord_3d,
                                            eventBundle.time_delta, 
                                            camera.eg_cameraMatrix);

    problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 14;
    // options.logging_type = ceres::SILENT;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = 100;
    // options.initial_trust_region_radius = 1;

    problem.SetParameterLowerBound(&angleAxis[0],0,-20);
    problem.SetParameterLowerBound(&angleAxis[0],1,-20);
    problem.SetParameterLowerBound(&angleAxis[0],2,-20);
    problem.SetParameterUpperBound(&angleAxis[0],0, 20);
    problem.SetParameterUpperBound(&angleAxis[0],1, 20);
    problem.SetParameterUpperBound(&angleAxis[0],2, 20);


    ceres::Solver::Summary summary; 
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;
    // if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
    // {
    //     cout <<" No convergence Event bundle time " << eventBundle.first_tstamp.toSec() <<", size " << eventBundle.size << endl;
    //     // cout << summary.BriefReport() << endl;
    // }
    

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
    cout << "Loss: " << 0 << ", est_angleAxis " <<est_angleAxis.transpose() << endl;

}

