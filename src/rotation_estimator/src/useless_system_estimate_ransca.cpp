
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

using namespace std;



/**
* \brief getTimeResidual from timesurface with random warptime.
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
* \brief using time as distance with random warptime, not bottom version
* r(delta_theta) = sum(delta_t**2); like eq(8) get jacobian function 
*/
Eigen::Vector3d System::DeriveTimeErrAnalyticRansac(const Eigen::Vector3d &vel_angleAxis, 
    const std::vector<int>& vec_sampled_idx, double warp_time, double& total_residual)
{

    /* using gt  calculate time difference */ 
    cout << "using DeriveRansac " << vel_angleAxis.transpose() 
            << " time " << warp_time << endl;

    getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, vel_angleAxis, warp_time);
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




/**
* \brief using time as distance with warptime to t0, bottom version
* r(delta_theta) = sum(delta_t**2); like eq(8) get jacobian function 
*/
Eigen::Vector3d System::DeriveTimeErrAnalyticRansacBottom(const Eigen::Vector3d &vel_angleAxis, 
        const std::vector<int>& vec_sampled_idx, double& total_residual)
{
    // get t0 time surface of warpped image !!
    getWarpedEventImage(est_angleAxis, event_warpped_Bundle); // init timesurface  
    // using gradient of time residual get warpped time surface
    double timesurface_range = 0.5; // the time range of timesurface 
    cv::Mat cv_warped_timesurface = cv::Mat(180,240, CV_32FC1); 
    cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
    double default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
    cv_warped_timesurface.setTo(default_value);
    // using event_warpped_Bundle not eventBundle !!! 

    double current_residual;
    int _sample_output = 50;
    for(int i=0; i<(event_warpped_Bundle.size*timesurface_range); i++)
    {
        int sampled_x = event_warpped_Bundle.coord.col(i)[0], sampled_y = event_warpped_Bundle.coord.col(i)[1]; 

        if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
        if(visited_map.at<uchar>(sampled_y, sampled_x) > 0) continue;   // visited 
        visited_map.at<uchar>(sampled_y, sampled_x) = 1;

        // linear 
        current_residual = eventBundle.time_delta(i);
        cv_warped_timesurface.at<float>(sampled_y, sampled_x) = current_residual;   
        
        // exp version 
        // current_residual = std::exp(eventBundle.time_delta(i)*100) - 1;
        // cv_warped_timesurface.at<float>(sampled_y, sampled_x) = current_residual;   

        // if(i % _sample_output == 0)
        //     cout << "default " <<default_value << " cv_warped_timesurfac, time " << eventBundle.time_delta(i) << ", residual " << current_residual << endl;
    } 

    // blur this image 
    cv::Mat blur_image, It_dx, It_dy; 
    cv::GaussianBlur(cv_warped_timesurface, blur_image, cv::Size(5, 5), 1);
    cv::Sobel(blur_image, It_dx, CV_32FC1, 1, 0);
    cv::Sobel(blur_image, It_dy, CV_32FC1, 0, 1);

    // visualized 
    // cv::Mat normed_img, color_residual, color_blur_residual;
    // cv::normalize(cv_warped_timesurface, normed_img, 0,255, cv::NORM_MINMAX, CV_8UC3);
    // cv::applyColorMap(normed_img, color_residual, cv::COLORMAP_JET);
    
    // cv::normalize(blur_image, color_blur_residual, 0,255, cv::NORM_MINMAX, CV_8UC3);
    // cv::applyColorMap(color_blur_residual, color_blur_residual, cv::COLORMAP_JET);
    // cv::namedWindow("color_residual", cv::WINDOW_NORMAL);
    // cv::namedWindow("color_blur_residual", cv::WINDOW_NORMAL);
    // cv::imshow("color_residual", color_residual);
    // cv::imshow("color_blur_residual", color_blur_residual);
    // cv::waitKey(30);


    // warp points according to their timestamps
    getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, vel_angleAxis, -1);
    event_warpped_Bundle.Projection(camera.eg_cameraMatrix);
    event_warpped_Bundle.DiscriminateInner(camera.width, camera.height);
    
    // calculate time difference
    total_residual = 0;
    std::vector<double> vec_residual = std::vector<double>(vec_sampled_idx.size(),0);  // gradient of residual
    std::vector<double> vec_Ix_interp = std::vector<double>(vec_sampled_idx.size(),0);
    std::vector<double> vec_Iy_interp = std::vector<double>(vec_sampled_idx.size(),0);
    std::vector<int> vec_sampled_idx_valid;

    int currnet_pos = 0;
    for(const int& idx : vec_sampled_idx)
    {
        // sampled data 
        int sampled_x = event_warpped_Bundle.coord.col(idx)[0];
        int sampled_y = event_warpped_Bundle.coord.col(idx)[1];
        double sampled_time = eventBundle.time_delta(idx);
        // cout << "sampled_time " << sampled_time <<", x, y" << x << "," << y << endl;

        if(event_warpped_Bundle.isInner(idx) < 1) continue;
        if(sampled_x >= 240  ||  sampled_x < 0 || sampled_y >= 180 || sampled_y < 0 ) 
        {
            cout << "idx " << idx << " x, y" << sampled_x << "," << sampled_y << endl;
            continue;
        }
        
        vec_sampled_idx_valid.push_back(idx);

        /* get warpped time residual, version 2 */

        double curr_t_residual = blur_image.at<float>(sampled_y, sampled_x);
        vec_residual[currnet_pos] = curr_t_residual;
        vec_Ix_interp[currnet_pos] = It_dx.at<float>(sampled_y, sampled_x);
        vec_Iy_interp[currnet_pos] = It_dy.at<float>(sampled_y, sampled_x);

        currnet_pos++;
        total_residual += curr_t_residual;
    }

    double grad_x_sum = std::accumulate(vec_Ix_interp.begin(), vec_Ix_interp.end(),0);
    double grad_y_sum = std::accumulate(vec_Iy_interp.begin(), vec_Iy_interp.end(),0);
    // cout << "grad_x_sum, grad_y_sum " << grad_x_sum << "," << grad_y_sum << endl; 

    // cout << "total_residual " << total_residual << endl;

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
        Ix_interp(i) = vec_residual[i] * vec_Ix_interp[i] * camera.eg_cameraMatrix(0,0);  
        Iy_interp(i) = vec_residual[i] * vec_Iy_interp[i] * camera.eg_cameraMatrix(1,1);  
        
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
    jacobian(0) = eg_jacobian.row(0) * _delta_time_valid;
    jacobian(1) = eg_jacobian.row(1) * _delta_time_valid;
    jacobian(2) = eg_jacobian.row(2) * _delta_time_valid;

    // cout << "jacobian " << jacobian.transpose() << endl;

    return jacobian;
}


/**
* \brief using time as distance.
* \param warp_time_ratio warp time range * this ratio  
*/
void System::EstimateMotion_ransca_warp2bottom(double sample_start, double sample_end, double opti_steps)
{
    cout << "------------- sample_start " <<
        sample_start<< " sample_end " << sample_end<< ", step, " << opti_steps<< " ----" <<endl;
    
    // cout << "using angle axis " << est_angleAxis.transpose() << endl;
    // warp events and get a timesurface image with indexs 
    // cv::Mat cv_timesurface; 
    // cv_timesurface = getImageFromBundle(event_undis_Bundle, PlotOption::TIME_SURFACE);  
    // cv::normalize(cv_timesurface, hot_image_C1, 0,255, cv::NORM_MINMAX, CV_8UC1);
    // cv::cvtColor(hot_image_C1, hot_image_C3, cv::COLOR_GRAY2BGR);


    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx; 
    int samples_count = std::min(10000,int(event_undis_Bundle.size * (sample_end-sample_start)));
    getSampledVec(vec_sampled_idx, samples_count, sample_start, sample_end);


    // otimizing paramter init
        int max_iter_count = 500;
        // velocity optimize steps and smooth factor
        double mu_event = opti_steps, nu_event = 1;
        Eigen::Vector3d adam_v(0,0,0); 

        double rho_event = 0.995, nu_map = 1;
        Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
    
    // choose init velocity, test whether using last_est or {0,0,0}, 
    {
        double residual1 = 0, residual2 = 0;
        DeriveTimeErrAnalyticRansacBottom(est_angleAxis, vec_sampled_idx, residual1);
        DeriveTimeErrAnalyticRansacBottom(Eigen::Vector3d(0,0,0), vec_sampled_idx, residual2);
        if(residual1 > residual2)
        {
            cout << "at time " <<eventBundle.first_tstamp.toSec() << ", using {0,0,0} "<< endl;
            est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 
        }
    }

    double residuals; 
    int sample_output = 20;
    for(int i=0; i< max_iter_count; i++)
    {
        // compute jacobian 
        Eigen::Vector3d jacobian = DeriveTimeErrAnalyticRansacBottom(est_angleAxis, vec_sampled_idx, residuals);

        // smooth factor, RMS Prob 
            double temp_jaco = jacobian.transpose()*jacobian; 
            nu_event =  temp_jaco*(1.0 - rho_event) + rho_event * nu_event;
            angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * jacobian;

        // smooth factor, Adam 
            // adam_v = 0.9*adam_v + (1-0.9) * jacobian;
            // nu_event =  (1.0-rho_event) * jacobian.transpose()*jacobian + rho_event*nu_event;
            // angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * adam_v;

        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true); 
        est_angleAxis = SO3add(angular_velocity_compensator, est_angleAxis, true); 
        // est_angleAxis = est_angleAxis.array() + angular_velocity_compensator.array(); 


        if(false && i % sample_output == 0)
        {
            getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);
            cv::imshow("opti", curr_warpped_event_image);
            cv::waitKey(30);

            cout << "iter " << i <<", nu_event " << std::sqrt(nu_event) << endl;
            // cout << "  jacobian " << jacobian.transpose() << endl;
            cout << "  compensator norm "  << angular_velocity_compensator.norm() << ", " << angular_velocity_compensator.transpose() << endl;
            cv::Mat curr_warpped_event_image_c1 = getWarpedEventImage(est_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE);
            int est_nonzero = 0;
            cout << "  residuals " << residuals << ", var of est " << getVar(curr_warpped_event_image_c1, est_nonzero, CV_16U) <<" non_zero " <<est_nonzero <<  endl;
        }

        if(angular_velocity_compensator.norm() < 0.0005)
        {
            cout << "early break, iter " << i << " norm "<<angular_velocity_compensator.norm() << endl;
            break;
        } 

    }

    bool visual_hot_c3 = false; 
    if(visual_hot_c3)
    {
        int sample_green = 1;
        for(int i=0; i<vec_sampled_idx.size() ; i++)
        {
            // if(i<10) cout << "sampling " << sample_idx << endl;

            // viusal sample 
            if(i%sample_green != 0) continue;
            int x = int(event_undis_Bundle.coord(0, vec_sampled_idx[i]));
            int y = int(event_undis_Bundle.coord(1, vec_sampled_idx[i]));
            hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,255,0);   // green of original 
        }

        // compare with gt
        cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;   
        
        // compare with gt 
        if(using_gt)
        {
            double gt_residual = 0;
            int gt_nonzero = 0; 
            // DeriveTimeErrAnalyticLayer(gt_angleAxis, vec_sampled_idx, warp_time, gt_residual);
            // cout << "DeriveTimeErrAnalyticLayer " << gt_residual << endl;
            DeriveTimeErrAnalyticRansac(gt_angleAxis, vec_sampled_idx, -1, gt_residual);
            cout << "DeriveTimeErrAnalyticRansca " << gt_residual << endl;
            getWarpedEventImage(gt_angleAxis, event_warpped_Bundle_gt).convertTo(curr_warpped_event_image_gt, CV_32F);
            cv::Mat curr_warpped_event_image_gt_C1 = getWarpedEventImage(gt_angleAxis, event_warpped_Bundle_gt, PlotOption::U16C1_EVNET_IMAGE);
            cout << "var of gt " << getVar(curr_warpped_event_image_gt_C1, gt_nonzero, CV_16U) <<" non_zero "<< gt_nonzero <<  endl;
        
            // visualize gt blue points
            int sample_blue = 1;
            for(int i=0; i<vec_sampled_idx.size(); i++)
            {
                // viusal sample 
                if(i%sample_blue != 0) continue;

                int x = int(event_warpped_Bundle_gt.coord(0, vec_sampled_idx[i]));
                int y = int(event_warpped_Bundle_gt.coord(1, vec_sampled_idx[i]));
                if(x>239 || y>179 || x<0 ||y<0) continue;
                hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(255,0,0);   // blue of warpped 
                // cout << "all inlier blue" << endl;
            }
            cout << "gt angleAxis  " << gt_angleAxis.transpose() << endl;
            cout << "gt residuals " << gt_residual << " var of gt " << getVar(curr_warpped_event_image_gt_C1, gt_nonzero, CV_16U) <<" non_zero " << gt_nonzero <<  endl;
        }


        // plot earlier timestamps oringe 22，G：07，B：201
        for(int i=0; i<eventBundle.size/15 ; i++)
        {
            // viusal sample 
            // cout << "hello " << endl;
            if(i%sample_green != 0) continue;
            int x = int(event_undis_Bundle.coord(0, i));
            int y = int(event_undis_Bundle.coord(1, i));
            hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 165, 255);   // earlier  
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
            hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);   // red of warpped 
            // cout << "all inlier red" << endl;
        }
    }

}

