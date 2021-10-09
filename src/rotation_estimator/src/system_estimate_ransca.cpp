
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

using namespace std;



/**
* \brief Constructor.
* \param sampled_x, sampled_y, sampled_time from original events.
* \param warp_time given delta time.
*/
void System::getTimeResidual(int sampled_x, int sampled_y, double sampled_time, double warp_time,
    double& residual, double& grad_x, double& grad_y)
{
    // check 
    assert(sampled_x>4);
    assert(sampled_y>4);
    assert(sampled_x<235);
    assert(sampled_y<175);


    // get specific location for gaussian 8 neighbors are collected 
    Eigen::VectorXd sobel_x(9); sobel_x <<  -1, 0, 1, -2, 0, 2, -1, 0, 1; 
    Eigen::VectorXd sobel_y(9); sobel_y <<  -1, -2, -1, 0, 0, 0,1, 2, 1; 
    Eigen::VectorXd neighbors_9(9);
    Eigen::Matrix<double, 5,5> neighbors_15; 

    // cout << "sobelx " << sobel_x.transpose() << endl;
    // cout << "sobel_y " << sobel_y.transpose() << endl;
    // cout << "neighbors " << neighbors.transpose() << endl;

    cv::Mat gauss_vec = cv::getGaussianKernel(3, 1, CV_64F); 
    cv::Mat gauss_2d = gauss_vec * gauss_vec.t();
    // cout << "gaussian \n" << gauss_2d << endl;
    Eigen::Matrix3d gauss_2d_eg;
    cv::cv2eigen(gauss_2d, gauss_2d_eg ); 
    // cout << "gauss_2d_eg \n" << gauss_2d_eg << endl;

    for(int dy=-2; dy<3; dy++)
        for(int dx=-2; dx<3; dx++)
        {
            double curr_residual = 0.03;  //TODO change this parameter
            for(int i=0; i<cv_3D_surface_index_count.at<int>(sampled_y+dy,sampled_x+dx); i++)
            {
                double iter_time = eventBundle.time_delta(cv_3D_surface_index.at<int>(sampled_y+dy,sampled_x+dx,i));
                curr_residual = std::min(curr_residual, std::abs(sampled_time-warp_time-iter_time)) ;

            }       
            neighbors_15(dy+2, dx+2) = curr_residual; 
        }

    // cout << "neighbors_15 \n" << neighbors_15 << endl;

    int count = 0;  // 5x5 
    for(int dy=0; dy<3; dy++)
        for(int dx=0; dx<3; dx++)
        {
            neighbors_9(count++) = (neighbors_15.block(dy,dx,3,3).array() * gauss_2d_eg.array()).sum(); 
            // cout << "neighbors_15.block(dy,dx,3,3) \n" << neighbors_15.block(dy,dx,3,3) <<endl;
        }

    // cout << "neighbors_9 \n" << neighbors_9.transpose() << endl;
    residual = neighbors_15(2,2); // average  
    grad_x   = 0.2 * neighbors_9.transpose() * sobel_x ; // average 
    grad_y   = 0.2 * neighbors_9.transpose() * sobel_y ; // average 

    // cout << "residual " << residual << " grad_x " << grad_x << " grad_y " << grad_y <<  endl; 
}



/**
* \brief using time as distance .
* r(delta_theta) = sum(delta_t**2); like eq(8) get jacobian function 
*/
Eigen::Vector3d System::DeriveTimeErrAnalyticRansac(const Eigen::Vector3d &vel_angleAxis, 
    const std::vector<int>& vec_sampled_idx, double warp_time, double& total_residual)
{

    /* using gt  calculate time difference */ 
    cout << "using DeriveRansac " << vel_angleAxis.transpose() 
            << " time " << warp_time << endl;

    getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, vel_angleAxis, Eigen::Vector3d::Zero(), warp_time);
    event_warpped_Bundle.Projection(camera.eg_cameraMatrix);
    event_warpped_Bundle.DiscriminateInner(camera.width-4, camera.height-4);
    
    // calculate time difference
    total_residual = 0;
    std::vector<double> vec_residual, vec_Ix_interp, vec_Iy_interp;  // gradient of residual
    std::vector<int> vec_sampled_idx_valid;
    for(const int& idx : vec_sampled_idx)
    {
        // sampled data 
        int sampled_x = event_warpped_Bundle.coord.col(idx)[0];
        int sampled_y = event_warpped_Bundle.coord.col(idx)[1];
        double sampled_time = eventBundle.time_delta(idx);
        // cout << "sampled_time " << sampled_time <<", x, y" << x << "," << y << endl;

        // if(event_warpped_Bundle.isInner(idx) < 1) continue;
        if(sampled_x >= 235  ||  sampled_x < 5 || sampled_y >= 175 || sampled_y < 5 ) 
        {
            // cout << "warning DeriveTimeErrAnalyticRansac x, y" << sampled_x << "," << sampled_y << endl;
            continue; 
        }

        vec_sampled_idx_valid.push_back(idx);

        /* get warpped time residual  */
        double curr_t_residual = 0, grad_x = 0, grad_y = 0;
        getTimeResidual(sampled_x, sampled_y, sampled_time, warp_time, curr_t_residual, grad_x, grad_y);
        
        vec_residual.push_back(curr_t_residual);
        vec_Ix_interp.push_back(grad_x);
        vec_Iy_interp.push_back(grad_y);

        total_residual += curr_t_residual;
    }

    // cout << "vec_residual[i] ";
    // for(int i=0; i<5; i++ )
    // {
    //     cout << " " << vec_residual[i];
    // }
    // cout << endl;

    // cout << "vec_Ix_interp[i] " ;
    // for(int i=0; i<5; i++ )
    // {
    //     cout << " " << vec_Ix_interp[i];
    // }
    // cout << endl;
    
    // cout << "vec_Iy_interp[i] " ;
    // for(int i=0; i<5; i++ )
    // {
    //     cout << " " << vec_Iy_interp[i];
    // }
    // cout << endl;


    double grad_x_sum = std::accumulate(vec_Ix_interp.begin(), vec_Ix_interp.end(),0);
    double grad_y_sum = std::accumulate(vec_Iy_interp.begin(), vec_Iy_interp.end(),0);
    // cout << "grad_x_sum, grad_y_sum " << grad_x_sum << "," << grad_y_sum << endl; 

    int valid_size = vec_sampled_idx_valid.size();
    Eigen::Matrix3Xd eg_jacobian;
    Eigen::VectorXd Ix_interp, Iy_interp, x_z, y_z, _delta_time_valid;
    Ix_interp.resize(valid_size);
    Iy_interp.resize(valid_size);
    x_z.resize(valid_size);
    y_z.resize(valid_size);
    _delta_time_valid.resize(valid_size);
    eg_jacobian.resize(3,valid_size);

    for(int i=0; i<valid_size; i++)
    {
        int x = int(event_warpped_Bundle.coord(0,vec_sampled_idx_valid[i])), y = int(event_warpped_Bundle.coord(1,vec_sampled_idx_valid[i]));
        // conversion from float to double
        Ix_interp(i) = 2* vec_residual[i] * vec_Ix_interp[i] * camera.eg_cameraMatrix(0,0);  
        Iy_interp(i) = 2* vec_residual[i] * vec_Iy_interp[i] * camera.eg_cameraMatrix(1,1);  
        
        _delta_time_valid(i) = eventBundle.time_delta(vec_sampled_idx_valid[i]);

        x_z(i) = event_warpped_Bundle.coord_3d(0,vec_sampled_idx_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_sampled_idx_valid[i]);
        y_z(i) = event_warpped_Bundle.coord_3d(1,vec_sampled_idx_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_sampled_idx_valid[i]);
    }

    // cout << "Ix_interp " << Ix_interp.topLeftCorner(1,5) << endl;
    // cout << "Iy_interp " << Iy_interp.topLeftCorner(1,5) << endl;
    
    eg_jacobian.row(0) = -Ix_interp.array()*x_z.array()*y_z.array() 
                        - Iy_interp.array()*(1+y_z.array()*y_z.array());

    eg_jacobian.row(1) = Ix_interp.array()*(1+x_z.array()*x_z.array()) 
                        + Iy_interp.array()*x_z.array()*y_z.array();
    
    eg_jacobian.row(2) = -Ix_interp.array()*y_z.array() 
                        + Iy_interp.array()*x_z.array();    
    
    // cout << "eg_jacobian " << eg_jacobian.topLeftCorner(3,5) << endl;
    

    Eigen::Vector3d jacobian;
    if(warp_time > 0)
    {
        jacobian(0) = 10 * warp_time * eg_jacobian.row(0) * Eigen::VectorXd::Ones(valid_size);
        jacobian(1) = 10 * warp_time * eg_jacobian.row(1) * Eigen::VectorXd::Ones(valid_size);
        jacobian(2) = 10 * warp_time * eg_jacobian.row(2) * Eigen::VectorXd::Ones(valid_size);
    }
    else
    {
        jacobian(0) = 10 * eg_jacobian.row(0) * _delta_time_valid;
        jacobian(1) = 10 * eg_jacobian.row(1) * _delta_time_valid ;
        jacobian(2) = 10 * eg_jacobian.row(2) * _delta_time_valid;     
    }

    // cout << "jacobian " << jacobian.transpose() << endl;

    return jacobian;
}


