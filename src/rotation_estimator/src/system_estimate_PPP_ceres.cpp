
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
            const Eigen::VectorXf& vec_polar,
            const Eigen::Matrix3d& K):
            coord_3d_(coord_3d), vec_delta_time_(vec_delta_time), intrisic_(K), vec_polar_(vec_polar)
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


        warped_3D_T.row(0) = coord_3d_T.row(0).array() - delta_points_T.row(0).array()*delta_time_T.transpose().array(); // + delta_second_points_T(0)*T(0.5*delta_time_*delta_time_);
        warped_3D_T.row(1) = coord_3d_T.row(1).array() - delta_points_T.row(1).array()*delta_time_T.transpose().array(); // + delta_second_points_T(1)*T(0.5*delta_time_*delta_time_);
        warped_3D_T.row(2) = coord_3d_T.row(2).array() - delta_points_T.row(2).array()*delta_time_T.transpose().array(); // + delta_second_points_T(2)*T(0.5*delta_time_*delta_time_);
        
        // cout << "coord_3d_T " << coord_3d_T.topLeftCorner(3,5) << endl;
        // cout << "delta_points_T  " << delta_points_T.topLeftCorner(3,5) << endl;
        // cout << "delta time " << delta_time_T.topLeftCorner(5,1).transpose()<< endl;
        // cout << "taylor first  " << warped_3D_T.topLeftCorner(3,5)<< endl;
                
        points_2D_T.row(0) = warped_3D_T.row(0).array()/warped_3D_T.row(2).array()*T(intrisic_(0,0)) + T(intrisic_(0,2));
        points_2D_T.row(1) = warped_3D_T.row(1).array()/warped_3D_T.row(2).array()*T(intrisic_(1,1)) + T(intrisic_(1,2));

        // cout << " points_2D_T \n " << points_2D_T.topLeftCorner(2,3)<<endl;
        // cout << "intrinct " << intrisic_ << endl;

        // // accumulate && select inlier 
        Eigen::Matrix<T, -1, 240> warpped_images_pos, warpped_images_neg;  
        warpped_images_pos.resize(180, 240);
        warpped_images_pos.setZero();
        warpped_images_neg.resize(180, 240);
        warpped_images_neg.setZero();
        // cout << "orignal mean " << warpped_images_pos.mean() << endl;

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

            if(vec_polar_(i) == 1)
            {
                warpped_images_pos(y_int, x_int)     += (T(1)-dx) * (T(1)-dy);
                warpped_images_pos(y_int+1, x_int)   += (T(1)-dx) * dy;
                warpped_images_pos(y_int, x_int+1)   += dx * (T(1)-dy);
                warpped_images_pos(y_int+1, x_int+1) += dx*dy;
            }
            else
            {
                warpped_images_neg(y_int, x_int)     += (T(1)-dx) * (T(1)-dy);
                warpped_images_neg(y_int+1, x_int)   += (T(1)-dx) * dy;
                warpped_images_neg(y_int, x_int+1)   += dx * (T(1)-dy);
                warpped_images_neg(y_int+1, x_int+1) += dx*dy;   
            }

            // if( i < 5)
            // {
            //     std::cout << "l1 " << (T(1)-dx) * (T(1)-dy) <<", l2 " << (T(1)-dx) * dy 
            //         <<", l3 " << dx * (T(1)-dy) << ", l4 " << dx*dy << endl;
            // }
        }

        // std::cout << "vec_polar_ " <<vec_polar_.topLeftCorner(0,10) << endl; 
        // std::cout << "warpped_images_neg " << warpped_images_neg(10,10) << endl;

 // gaussian_blur(warpped_images);
        {
            int templates[25] = { 1, 4, 7, 4, 1,   
                                4, 16, 26, 16, 4,   
                                7, 26, 41, 26, 7,  
                                4, 16, 26, 16, 4,   
                                1, 4, 7, 4, 1 };        
            int height = 180, width = 240;
            for (int j=2;j<height-4;j++)  
            {  
                for (int i=2;i<width-4;i++)  
                {  
                    T sum_p = T(0);  
                    T sum_n = T(0);  
                    int index = 0;  
                    for ( int m=j-2; m<j+3; m++)  
                    {  
                        for (int n=i-2; n<i+3; n++)  
                        {  
                            sum_p +=  warpped_images_pos(m, n) * T(templates[index]) ;  
                            sum_n +=  warpped_images_neg(m, n) * T(templates[index]) ;
                            index++;  
                        }  
                    }  
                    sum_p /= T(273);  
                    sum_n /= T(273);  
                    if (sum_p > T(255)) sum_p = T(255);  
                    if (sum_n > T(255)) sum_n = T(255);  
                    warpped_images_pos(j, i) = sum_p;  
                    warpped_images_neg(j, i) = sum_n;  
                }  
            }  
        }

        // std::cout << "warpped_images_neg " << warpped_images_neg(10,10) << endl;


        T loss1 = T(0), loss2 = T(0);

        T r = T(0.1), beta = T(1.59);

        T img1_sum = T(0), img2_sum = T(0);
        for(int i=0; i<180; i++)
        for(int j=0; j<240; j++)
        {
            loss1 +=  ceres::log((warpped_images_pos(i,j) + r)) +  
                ceres::log(r * (beta)) - 
                ceres::log(warpped_images_pos(i,j) + T(1.)) -
                ceres::log(r) - 
                (warpped_images_pos(i,j) + r) * ceres::log(beta + T(1));

            loss2 +=  ceres::log((warpped_images_neg(i,j) + r)) +  
                ceres::log(r * (beta)) - 
                ceres::log(warpped_images_neg(i,j) + T(1.)) -
                ceres::log(r) - 
                (warpped_images_neg(i,j) + r) * ceres::log(beta + T(1));

            img1_sum += warpped_images_pos(i,j);
            img2_sum += warpped_images_neg(i,j);

        }        

        // std::cout << "img1_sum " << img1_sum << endl;
        // std::cout << "img1_sum " << img2_sum << endl;
        // std::cout << "loss1 " << loss1 << endl;


        loss1 = loss1 / (img1_sum + T(0.01));
        loss2 = loss2 / (img2_sum + T(0.01));

        residual[0] = T(100) + loss1 + loss2;

        // cout << "loss1 " << loss1 << endl;
        // cout << "loss2 " << loss2 << endl;
        // cout << "residual " << residual[0] << endl;


        // std::cout << "residual " << residual[0] << endl;


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
        Eigen::VectorXf& vec_polar,
        const Eigen::Matrix3d& K)
        {
            return new ceres::AutoDiffCostFunction<ContrastCostFunction,1, 3>(
                new ContrastCostFunction(coord_3d, vec_delta_time,vec_polar, K));
        }

    // inputs     
    Eigen::Matrix3Xd coord_3d_;
    Eigen::VectorXd vec_delta_time_; 
    Eigen::VectorXf vec_polar_;
    Eigen::Matrix3d intrisic_; 

};


/**
* \brief using time as distance, using ceres as optimizer, const gradient for each optimizing.
*/
void System::EstimateMotion_PPP_ceres()
{
    cout << "event time "<< eventBundle.first_tstamp.toSec() <<  ", total " << eventBundle.size <<endl;

    double residuals; 
    // est_angleAxis = Eigen::Vector3d::Zero();
    double angleAxis[3] = {est_angleAxis(0), est_angleAxis(1), est_angleAxis(2)}; 
    // double angleAxis[3] = {2.1399965, 2.4666412, 2.8370314}; 
       
    // init problem 
    ceres::Problem problem; 
    // add residual 

        // only debug 
        // getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, est_angleAxis); 

        ceres::CostFunction* cost_function = ContrastCostFunction::Create(
                                                event_undis_Bundle.coord_3d,
                                                eventBundle.time_delta,
                                                eventBundle.polar,
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
        ceres::Solve(options, &problem, &summary);
        cout << summary.BriefReport() << endl;

    
    problem.SetParameterLowerBound(&angleAxis[0],0,-20);
    problem.SetParameterUpperBound(&angleAxis[0],0,20);
    problem.SetParameterLowerBound(&angleAxis[0],1,-20);
    problem.SetParameterUpperBound(&angleAxis[0],1,20);
    problem.SetParameterLowerBound(&angleAxis[0],2,-20);
    problem.SetParameterUpperBound(&angleAxis[0],2,20);


    // cout << "   iter " << iter_ << ", ceres iters " << summary.iterations.size()<< endl;
    // if(summary.BriefReport().find("NO_CONVERGENCE") != std::string::npos)
    // {
    //     cout <<" No convergence Event bundle time " << eventBundle.first_tstamp.toSec() <<", size " << eventBundle.size << endl;
    //     // cout << summary.BriefReport() << endl;
    // }
    

    est_angleAxis = Eigen::Vector3d(angleAxis[0],angleAxis[1],angleAxis[2]);
    cout << "Loss: " << 0 << ", est_angleAxis " <<est_angleAxis.transpose() << endl;

    // getWarpedEventImage(-est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3); 


}

