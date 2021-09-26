
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

using namespace std;


/**
* \brief given angular veloity(t1->t2), warp local event bundle become shaper
*/
cv::Mat System::getWarpedEventImage(const Eigen::Vector3d & cur_ang_vel, EventBundle& event_out,  const PlotOption& option)
{
    // cout << "get warpped event image " << endl;

    /* warp local events become sharper */
    event_out.CopySize(event_undis_Bundle);
    getWarpedEventPoints(event_undis_Bundle, event_out, cur_ang_vel); 
    event_out.Projection(camera.eg_cameraMatrix);
    event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(curr_warpped_event_image, CV_32F);

    return getImageFromBundle(event_out, option, false);

    // testing 
    // cv::Mat x_img, y_img, z_img, x5_img, y5_img, z5_img; 
    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(10,0,0)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(x_img, CV_32F);
    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(5,0,0)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(x5_img, CV_32F);


    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(0,10,0)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(y_img, CV_32F);
    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(0,5,0)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(y5_img, CV_32F);

    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(0,0,10)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(z_img, CV_32F);
    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(0,0,5)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(z5_img, CV_32F);

    // cv::imshow("x ", x_img);
    // cv::imshow("y ", y_img);
    // cv::imshow("z ", z_img);
    // cv::imshow("x 5", x5_img);
    // cv::imshow("y 5", y5_img);
    // cv::imshow("z 5", z5_img);
    // cv::waitKey(0);

    // cout << "  success get warpped event image " << endl;
}

/**
* \brief given angular veloity, warp local event bundle(t2) to the reference time(t1)
    using kim RAL21, eqation(11), since the ratation angle is ratively small
* \param cur_ang_vel angleAxis/delta_t from t1->t2, so if it is zero, not shapper operation is done in delta time. 
* \param cur_ang_pos are set (0,0,0) default, so the output is not rotated. 
* \param delta_time all events warp this time. 
*/
void System::getWarpedEventPoints(const EventBundle& eventIn, EventBundle& eventOut, 
    const Eigen::Vector3d& cur_ang_vel, const Eigen::Vector3d& cur_ang_pos, double delta_time)
{
    // the theta of rotation axis
    float ang_vel_norm = cur_ang_vel.norm(); 

    Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
    
    ang_vel_hat_mul_x.resize(3,eventIn.size);     // row, col 
    ang_vel_hat_sqr_mul_x.resize(3,eventIn.size); 
    
    // equation 11 
    ang_vel_hat_mul_x.row(0) = -cur_ang_vel(2)*eventIn.coord_3d.row(1) + cur_ang_vel(1)*eventIn.coord_3d.row(2);
    ang_vel_hat_mul_x.row(1) =  cur_ang_vel(2)*eventIn.coord_3d.row(0) - cur_ang_vel(0)*eventIn.coord_3d.row(2);
    ang_vel_hat_mul_x.row(2) = -cur_ang_vel(1)*eventIn.coord_3d.row(0) + cur_ang_vel(0)*eventIn.coord_3d.row(1);

    ang_vel_hat_sqr_mul_x.row(0) = -cur_ang_vel(2)*ang_vel_hat_mul_x.row(1) + cur_ang_vel(1)*ang_vel_hat_mul_x.row(2);
    ang_vel_hat_sqr_mul_x.row(1) =  cur_ang_vel(2)*ang_vel_hat_mul_x.row(0) - cur_ang_vel(0)*ang_vel_hat_mul_x.row(2);
    ang_vel_hat_sqr_mul_x.row(2) = -cur_ang_vel(1)*ang_vel_hat_mul_x.row(0) + cur_ang_vel(0)*ang_vel_hat_mul_x.row(1);


    eventOut.CopySize(eventIn);
    if(ang_vel_norm/3.14 * 180 < 0.1) 
    {
        cout << "  small angle vec " << ang_vel_norm/3.14 * 180 << " degree /s" << endl;
        eventOut.coord_3d = eventIn.coord_3d ;
    }
    else
    {   
        Eigen::VectorXd vec_delta_time = eventBundle.time_delta;  
        if(delta_time > 0)  // using self defined deltime. 
        {
            vec_delta_time.setConstant(delta_time);
            // cout <<"using const delta " << delta_time << endl;
        }
        // else{ cout <<"using not const delta " << delta_time << endl; }
        
        
        // second order version x_t2 = x_t1 + v_t12 * delta_t * x + second_order;
        // so x_t1 = x_t2 - v_t12 * delta_t * x 
        eventOut.coord_3d = eventIn.coord_3d
                                    + Eigen::MatrixXd( 
                                        ang_vel_hat_mul_x.array().rowwise() 
                                        * (vec_delta_time.transpose().array())
                                        + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                        * (0.5f * vec_delta_time.transpose().array().square()) );
        // cout << "delta_t " << delta_time.topRows(5).transpose() << endl;

        // first order version 
        // event_warpped_Bundle.coord_3d = event_undis_Bundle.coord_3d
        //                             + Eigen::MatrixXd( 
        //                                 ang_vel_hat_mul_x.array().rowwise() 
        //                                 * delta_time.transpose().array());

        // cout << "angle vec: " << (cur_ang_vel.array()/3.14 * 180).transpose() << " degree/s" << endl;
        // cout << "ang_vel_hat_mul_x: \n"<< ang_vel_hat_mul_x.topLeftCorner(3,5) << endl;
        // cout << "delta time: \n" << delta_time.topRows(5).transpose()<< endl;
        // cout << "ang_vel_hat_mul_x: back \n"<< ang_vel_hat_mul_x.topRightCorner(3,5) << endl;
        // cout << "event_warpped_Bundle.coord_3d: \n " << event_warpped_Bundle.coord_3d.topLeftCorner(3,5) << endl;
        
    }

    if(cur_ang_pos.norm()/3.14 * 180 > 0.1) 
    {
        cout << "  warp to global map " << cur_ang_pos.norm()/3.14 * 180 << " degree /s" << endl;
        eventOut.coord_3d = SO3(cur_ang_pos) * eventIn.coord_3d;
    }
    // cout << "sucess getWarpedEventPoints" << endl;
}


/**
* \brief calculate jacobian of motion model.
* \param vel_angleAxis estimate angular velocity. 
* \param pos_angleAxis estimate pos compensate for const vel model. 
*/
Eigen::Vector3d System::DeriveErrAnalytic(const Eigen::Vector3d &vel_angleAxis, const Eigen::Vector3d &pos_angleAxis)
{
    getWarpedEventImage(vel_angleAxis, event_warpped_Bundle, 
        PlotOption::U16C1_EVNET_IMAGE).convertTo(curr_warpped_event_image, CV_32F);

    cv::Mat truncated_image; 
    int threshold = 10; 
    cv::threshold(curr_warpped_event_image, truncated_image, threshold, 255, cv::THRESH_TRUNC); 

    cv::Mat blur_image, Ix, Iy; 
    cv::GaussianBlur(truncated_image, blur_image, cv::Size(5, 5), 1);
    cv::Sobel(blur_image, Ix, CV_32FC1, 1, 0);
    cv::Sobel(blur_image, Iy, CV_32FC1, 0, 1);

    // cout << "using ag " << vel_angleAxis.transpose() << endl;
    // cout << "warped_event_image " << cv::norm(curr_warpped_event_image) << 
    //     ", truncated_image " << cv::norm(truncated_image) << 
    //     ", blur_image " << cv::norm(blur_image) << 
    //     ", Ix" << cv::norm(Ix) << endl;

    Eigen::VectorXd Ix_interp, Iy_interp, x_z, y_z, _delta_time_valid;  // the first row of euq(8)
    Eigen::Matrix3Xd eg_jacobian;
    
    std::vector<int> vec_valid;
    for(int i=0; i<event_warpped_Bundle.isInner.size(); i++)
    {
        if(event_warpped_Bundle.isInner[i] > 0)
            vec_valid.push_back(i);
    } 
    int valid_size = event_warpped_Bundle.isInner.sum();

    Ix_interp.resize(valid_size);
    Iy_interp.resize(valid_size);
    x_z.resize(valid_size);
    y_z.resize(valid_size);
    _delta_time_valid.resize(valid_size);
    eg_jacobian.resize(3,valid_size);

    Eigen::VectorXd warped_image_delta_t;
    warped_image_delta_t.resize(valid_size);

    for(int i=0; i<vec_valid.size(); i++)
    {
        int x = int(event_warpped_Bundle.coord(0,vec_valid[i])), y = int(event_warpped_Bundle.coord(1,vec_valid[i]));
        _delta_time_valid(i) = eventBundle.time_delta(vec_valid[i]);
        // conversion from float to double
        Ix_interp(i) = curr_warpped_event_image.at<float>(y,x) * Ix.at<float>(y,x);  
        Iy_interp(i) = curr_warpped_event_image.at<float>(y,x) * Iy.at<float>(y,x);
        
        // Ix_interp(i) = Ix.at<float>(y,x);  
        // Iy_interp(i) = Iy.at<float>(y,x);
        // warped_image_delta_t(i) = eventBundle.time_delta(vec_valid[i]) 
        //                         * curr_warpped_event_image.at<float>(y, x);

        x_z(i) = event_warpped_Bundle.coord_3d(0,vec_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_valid[i]);
        y_z(i) = event_warpped_Bundle.coord_3d(1,vec_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_valid[i]);
        
    }

    Ix_interp *= camera.eg_cameraMatrix(0,0);
    Iy_interp *= camera.eg_cameraMatrix(1,1);
    
    eg_jacobian.row(0) = -Ix_interp.array()*x_z.array()*y_z.array() 
                        - Iy_interp.array()*(1+y_z.array()*y_z.array());

    eg_jacobian.row(1) = Ix_interp.array()*(1+x_z.array()*x_z.array()) 
                        + Iy_interp.array()*x_z.array()*y_z.array();
    
    eg_jacobian.row(2) = -Ix_interp.array()*y_z.array() 
                        + Iy_interp.array()*x_z.array();


    // cout << "Ix_interp " << Ix_interp.topRows(5).transpose() << endl;
    // cout << "warped_image_delta_t " << warped_image_delta_t.topRows(5).transpose() << endl;
    // cout << "eg_jacobian " << eg_jacobian.topLeftCorner(1,5).transpose()  <<endl;
    // cout << "_delta_time " << _delta_time_valid.topRows(5).transpose()  <<endl;
    // cout << "curr_warpped_event_image :\n" << curr_warpped_event_image(cv::Range(100, 110), cv::Range(100, 110)) << endl;
    
    
    Eigen::Vector3d jacobian;

    jacobian(0) = eg_jacobian.row(0) * _delta_time_valid;
    jacobian(1) = eg_jacobian.row(1) * _delta_time_valid;
    jacobian(2) = eg_jacobian.row(2) * _delta_time_valid;

    // jacobian(0) = eg_jacobian.row(0) * warped_image_delta_t;
    // jacobian(1) = eg_jacobian.row(1) * warped_image_delta_t;
    // jacobian(2) = eg_jacobian.row(2) * warped_image_delta_t;

    // cout << "gradient " << jacobian.transpose() << endl;
    return jacobian;
}



/**
* \brief Constructor.
* \param sampled_x, sampled_y, sampled_time from original events.
* \param warp_time given delta time.
*/
void System::getTimeResidual(int sampled_x, int sampled_y, double sampled_time, double warp_time,
    double& residual, double& grad_x, double& grad_y)
{
    // check 
    assert(sampled_x>-5);
    assert(sampled_y>-5);
    assert(sampled_x<235);
    assert(sampled_y<175);


    // get specific location for gaussian 8 neighbors are collected 
    Eigen::VectorXd sobel_x(9); sobel_x <<  -1, 0, 1, -2, 0, 2, -1, 0, 1; 
    Eigen::VectorXd sobel_y(9); sobel_y <<  1, 2, 1, 0, 0, 0, -1, -2, -1; 
    Eigen::VectorXd neighbors_9(9);
    Eigen::Matrix<double, 5,5> neighbors_15; 

    // cout << "sobelx " << sobel_x.transpose() << endl;
    // cout << "sobel_y " << sobel_y.transpose() << endl;
    // cout << "neighbors " << neighbors.transpose() << endl;

    cv::Mat gauss_vec = cv::getGaussianKernel(3, 0.5, CV_64F); 
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

        if(event_warpped_Bundle.isInner(idx) < 1) continue;
        if(sampled_x >= 240  ||  sampled_x < 0 || sampled_y >= 180 || sampled_y < 0 ) 
            cout << "x, y" << sampled_x << "," << sampled_y << endl;

        vec_sampled_idx_valid.push_back(idx);

        /* get warpped time residual  */
        double curr_t_residual = 0, grad_x = 0, grad_y = 0;
        getTimeResidual(sampled_x, sampled_y, sampled_time, warp_time, curr_t_residual, grad_x, grad_y);
        
        vec_residual.push_back(curr_t_residual);
        vec_Ix_interp.push_back(grad_x);
        vec_Iy_interp.push_back(grad_y);

        total_residual += curr_t_residual;
    }

    double grad_x_sum = std::accumulate(vec_Ix_interp.begin(), vec_Ix_interp.end(),0);
    double grad_y_sum = std::accumulate(vec_Iy_interp.begin(), vec_Iy_interp.end(),0);
    // cout << "grad_x_sum, grad_y_sum " << grad_x_sum << "," << grad_y_sum << endl; 

    int valid_size = vec_sampled_idx_valid.size();
    Eigen::Matrix3Xd eg_jacobian;
    Eigen::VectorXd Ix_interp, Iy_interp, x_z, y_z;
    Ix_interp.resize(valid_size);
    Iy_interp.resize(valid_size);
    x_z.resize(valid_size);
    y_z.resize(valid_size);
    eg_jacobian.resize(3,valid_size);

    for(int i=0; i<valid_size; i++)
    {
        int x = int(event_warpped_Bundle.coord(0,vec_sampled_idx_valid[i])), y = int(event_warpped_Bundle.coord(1,vec_sampled_idx_valid[i]));
        // conversion from float to double
        Ix_interp(i) = 2* vec_residual[i] * vec_Ix_interp[i] * camera.eg_cameraMatrix(0,0) * warp_time ;  
        Iy_interp(i) = 2* vec_residual[i] * vec_Iy_interp[i] * camera.eg_cameraMatrix(1,1) * warp_time;  
        
        x_z(i) = event_warpped_Bundle.coord_3d(0,vec_sampled_idx_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_sampled_idx_valid[i]);
        y_z(i) = event_warpped_Bundle.coord_3d(1,vec_sampled_idx_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_sampled_idx_valid[i]);
    }

    
    eg_jacobian.row(0) = -Ix_interp.array()*x_z.array()*y_z.array() 
                        - Iy_interp.array()*(1+y_z.array()*y_z.array());

    eg_jacobian.row(1) = Ix_interp.array()*(1+x_z.array()*x_z.array()) 
                        + Iy_interp.array()*x_z.array()*y_z.array();
    
    eg_jacobian.row(2) = -Ix_interp.array()*y_z.array() 
                        + Iy_interp.array()*x_z.array();    
    
    
    Eigen::Vector3d jacobian;
    jacobian(0) = eg_jacobian.row(0) * Eigen::VectorXd::Ones(valid_size);
    jacobian(1) = eg_jacobian.row(1) * Eigen::VectorXd::Ones(valid_size);
    jacobian(2) = eg_jacobian.row(2) * Eigen::VectorXd::Ones(valid_size);

    return jacobian;
}


/**
* \brief using time as distance .
* r(delta_theta) = sum(delta_t**2); like eq(8) get jacobian function 
*/
Eigen::Vector3d System::DeriveTimeErrAnalyticLayer(const Eigen::Vector3d &vel_angleAxis, 
    const std::vector<int>& vec_sampled_idx, double warp_time, double& total_residual)
{

    // using gradient of time residual get warpped time surface
    cv::Mat cv_warped_timesurface = cv::Mat(180,240, CV_32FC1);
    double sampled_time = eventBundle.time_delta(eventBundle.time_delta.rows()-10);
    for(int sampled_x=0; sampled_x<240; sampled_x++)
        for(int sampled_y=0; sampled_y<180; sampled_y++)
    {
        double current_residual = 0.03;
        for(int i=0; i<cv_3D_surface_index_count.at<int>(sampled_y,sampled_x); i++)
        {
            double iter_time = eventBundle.time_delta(cv_3D_surface_index.at<int>(sampled_y,sampled_x,i));
            current_residual = std::min(current_residual, std::abs(sampled_time-warp_time-iter_time)) ;
            // cout << "  iter_time " << iter_time << " current_residual "<<current_residual << endl; 
        }
        // 1000 to increase float precision 
        cv_warped_timesurface.at<float>(sampled_y, sampled_x) = current_residual*1000;        
    } 

    cv::Mat blur_image, It_dx, It_dy; 
    cv::GaussianBlur(cv_warped_timesurface, blur_image, cv::Size(5, 5), 1);
    cv::Sobel(blur_image, It_dx, CV_32FC1, 1, 0);
    cv::Sobel(blur_image, It_dy, CV_32FC1, 0, 1);

    // for display 
    // cv::Mat cv_warped_timesurface_display, blur_image_display, It_dx_display, It_dy_display; 
    // cv::normalize(cv_warped_timesurface, cv_warped_timesurface_display, 0,255, cv::NORM_MINMAX,CV_8UC1);
    // cv::normalize(blur_image, blur_image_display, 0,255, cv::NORM_MINMAX,CV_8UC1);
    // cv::normalize(It_dx, It_dx_display, 0,255, cv::NORM_MINMAX,CV_8UC1);
    // cv::normalize(It_dy, It_dy_display, 0,255, cv::NORM_MINMAX,CV_8UC1);

    // cv::imshow("cv_warped_timesurface", cv_warped_timesurface_display);
    // cv::imshow("blur_image", blur_image_display);
    // cv::imshow("It_dx", It_dx_display);
    // cv::imshow("It_dy", It_dy_display);


    /* using gt  calculate time difference */ 
    /* using {0,0,0}  calculate time difference */ 
    // getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, {0,0,0}, Eigen::Vector3d::Zero(), warp_time);
    // getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, gt_angleAxis, Eigen::Vector3d::Zero(), warp_time);
    getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, vel_angleAxis, Eigen::Vector3d::Zero(), warp_time);
    event_warpped_Bundle.Projection(camera.eg_cameraMatrix);
    event_warpped_Bundle.DiscriminateInner(camera.width, camera.height);
    
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

        if(event_warpped_Bundle.isInner(idx) < 1) continue;
        if(sampled_x >= 240  ||  sampled_x < 0 || sampled_y >= 180 || sampled_y < 0 ) 
            cout << "x, y" << sampled_x << "," << sampled_y << endl;

        vec_sampled_idx_valid.push_back(idx);

        /* get warpped time residual, version 2 */

        double curr_t_residual = blur_image.at<float>(sampled_y, sampled_x)/1000;
        vec_residual.push_back(curr_t_residual);
        vec_Ix_interp.push_back(It_dx.at<float>(sampled_y, sampled_x)/1000);
        vec_Iy_interp.push_back(It_dy.at<float>(sampled_y, sampled_x)/1000);

        total_residual += curr_t_residual;
    }

    double grad_x_sum = std::accumulate(vec_Ix_interp.begin(), vec_Ix_interp.end(),0);
    double grad_y_sum = std::accumulate(vec_Iy_interp.begin(), vec_Iy_interp.end(),0);
    // cout << "grad_x_sum, grad_y_sum " << grad_x_sum << "," << grad_y_sum << endl; 

    // cout << "total_residual " << total_residual << endl;

    int valid_size = vec_sampled_idx_valid.size();
    Eigen::Matrix3Xd eg_jacobian;
    Eigen::VectorXd Ix_interp, Iy_interp, x_z, y_z;
    Ix_interp.resize(valid_size);
    Iy_interp.resize(valid_size);
    x_z.resize(valid_size);
    y_z.resize(valid_size);
    eg_jacobian.resize(3,valid_size);

    for(int i=0; i<valid_size; i++)
    {
        int x = int(event_warpped_Bundle.coord(0,vec_sampled_idx_valid[i])), y = int(event_warpped_Bundle.coord(1,vec_sampled_idx_valid[i]));
        // conversion from float to double
        Ix_interp(i) = 2* vec_residual[i] * vec_Ix_interp[i] * camera.eg_cameraMatrix(0,0) * warp_time ;  
        Iy_interp(i) = 2* vec_residual[i] * vec_Iy_interp[i] * camera.eg_cameraMatrix(1,1) * warp_time;  
        
        x_z(i) = event_warpped_Bundle.coord_3d(0,vec_sampled_idx_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_sampled_idx_valid[i]);
        y_z(i) = event_warpped_Bundle.coord_3d(1,vec_sampled_idx_valid[i]) / event_warpped_Bundle.coord_3d(2,vec_sampled_idx_valid[i]);

    }

    
    eg_jacobian.row(0) = -Ix_interp.array()*x_z.array()*y_z.array() 
                        - Iy_interp.array()*(1+y_z.array()*y_z.array());

    eg_jacobian.row(1) = Ix_interp.array()*(1+x_z.array()*x_z.array()) 
                        + Iy_interp.array()*x_z.array()*y_z.array();
    
    eg_jacobian.row(2) = -Ix_interp.array()*y_z.array() 
                        + Iy_interp.array()*x_z.array();    
    
    
    Eigen::Vector3d jacobian;
    jacobian(0) = eg_jacobian.row(0) * Eigen::VectorXd::Ones(valid_size);
    jacobian(1) = eg_jacobian.row(1) * Eigen::VectorXd::Ones(valid_size);
    jacobian(2) = eg_jacobian.row(2) * Eigen::VectorXd::Ones(valid_size);

    return jacobian;
}

/**
* \brief using time as distance.
* \param sample_ratio the later(time) part of events, 
* \param warp_time_ratio warp time range * this ratio  
*/
void System::EstimateMotion_ransca_once(double sample_ratio, double warp_time_ratio, double opti_steps)
{
    cout << "------------- sample_ratio " <<
        sample_ratio<< " warp_time_ratio " << warp_time_ratio<< ", step, " << opti_steps<< " ----" <<endl;
    // warp events and get a timesurface image with indexs 
    cv::Mat cv_timesurface; 
    cv_timesurface = getImageFromBundle(event_undis_Bundle, PlotOption::TIME_SURFACE);  
    // cv::Mat hot_image for visualization ;
    cv::normalize(cv_timesurface, hot_image_C1, 0,255, cv::NORM_MINMAX, CV_8UC1);
    // cv::applyColorMap(hot_image_C1, hot_image_C3, cv::COLORMAP_JET);
    // cv::cvtColor(hot_image, hot_image, cv::COLOR_BGR2GRAY);
    cv::cvtColor(hot_image_C1, hot_image_C3, cv::COLOR_GRAY2BGR);

    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx;
    cv::RNG rng(int(ros::Time::now().nsec));
    int samples_count = std::min(1000,int(event_undis_Bundle.coord.cols()/2));
    int sample_green = 2;
    for(int i=0; i< samples_count; i++)
    {
        int sample_idx = rng.uniform(int(event_undis_Bundle.coord.cols()*sample_ratio), event_undis_Bundle.coord.cols());
        vec_sampled_idx.push_back(sample_idx);
        // if(i<10) cout << "sampling " << sample_idx << endl;

        // viusal sample 
        if(i%sample_green != 0) continue;
        int x = int(event_undis_Bundle.coord(0, sample_idx));
        int y = int(event_undis_Bundle.coord(1, sample_idx));
        hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,255,0);   // green of original 

    }
    
    // warp time range  
    std::vector<double>  vec_sampled_time;
    for(const int& i: vec_sampled_idx)
        vec_sampled_time.push_back(eventBundle.time_delta(i));

    double delta_time_range = *(std::min_element(vec_sampled_time.begin(), vec_sampled_time.end()));
    double warp_time = delta_time_range * warp_time_ratio; 
    cout <<"sample count " << samples_count<< ", delta_time_range " << delta_time_range << "， warp_time " << warp_time <<endl;

    // otimizing paramter init
        int max_iter_count = 10;
        // velocity optimize steps and smooth factor
        double mu_event = opti_steps, nu_event = 0.9;
        Eigen::Vector3d adam_v(0,0,0); 

        double rho_event = 0.9, nu_map = 1;
        Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
        // est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 

    // compare with gt 
    double gt_residual = 0;
    int gt_nonzero = 0; 
    // DeriveTimeErrAnalyticLayer(gt_angleAxis, vec_sampled_idx, warp_time, gt_residual);
    // cout << "DeriveTimeErrAnalyticLayer " << gt_residual << endl;
    DeriveTimeErrAnalyticRansac(gt_angleAxis, vec_sampled_idx, warp_time, gt_residual);
    cout << "DeriveTimeErrAnalyticRansca " << gt_residual << endl;
    getWarpedEventImage(gt_angleAxis, event_warpped_Bundle_gt).convertTo(curr_warpped_event_image_gt, CV_32F);
    cv::Mat curr_warpped_event_image_gt_C1 = getWarpedEventImage(gt_angleAxis, event_warpped_Bundle_gt, PlotOption::U16C1_EVNET_IMAGE);
    cout << "var of gt " << getVar(curr_warpped_event_image_gt_C1, gt_nonzero) <<" non_zero "<< gt_nonzero <<  endl;
    
    // visualize gt blue points
    int sample_blue = 2;
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

    double residuals = 0, pre_residual = 10; 
    for(int i=0; i< max_iter_count; i++)
    // while(abs(residuals - pre_residual) > 0.1)
    {
        pre_residual = residuals;
        // compute jacobian 
        Eigen::Vector3d jacobian = DeriveTimeErrAnalyticLayer(est_angleAxis, vec_sampled_idx, warp_time, residuals);
        // Eigen::Vector3d jacobian = DeriveTimeErrAnalyticRansac(est_angleAxis, vec_sampled_idx, warp_time, residuals);

        // smooth factor, RMS Prob 
            double temp_jaco = jacobian.transpose()*jacobian; 
            nu_event =  temp_jaco*(1.0 - rho_event) + rho_event * nu_event;
            angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * jacobian;

        // smooth factor, Adam 
            // adam_v = 0.8*adam_v + (1-0.8) * jacobian;
            // nu_event =  (1.0-rho_event) * jacobian.transpose()*jacobian + rho_event*nu_event;
            // angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * adam_v;

        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true); 
        // est_angleAxis = SO3add(angular_velocity_compensator, est_angleAxis, true); 
        est_angleAxis = est_angleAxis.array() + angular_velocity_compensator.array(); 

        getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32F);
        cv::imshow("opti", curr_warpped_event_image);
        cv::waitKey(100);

        cout << "iter " << i <<", nu_event " << std::sqrt(nu_event) << endl;
        // cout << "  jacobian " << jacobian.transpose() << endl;
        cout << "  compensator   " << angular_velocity_compensator.transpose() << endl;
        cv::Mat curr_warpped_event_image_c1 = getWarpedEventImage(est_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE);
        int est_nonzero = 0;
        cout << "  residuals " << residuals << ", var of est " << getVar(curr_warpped_event_image_c1, est_nonzero) <<" non_zero " <<est_nonzero <<  endl;

    }

    // visualize est
    int sample_red = 2;
    for(int i=0; i<vec_sampled_idx.size(); i++)
    {
         // viusal sample 
        if(i%sample_red != 0) continue;
        int x = int(event_warpped_Bundle.coord(0, vec_sampled_idx[i])); 
        int y = int(event_warpped_Bundle.coord(1, vec_sampled_idx[i]));
        hot_image_C3.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);   // red of warpped 
        if(x>239 || y>179 || x<0 ||y<0) continue;
        // cout << "all inlier red" << endl;
    }

    // compare with gt
    cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;
    cout << "gt angleAxis        " << gt_angleAxis.transpose() << endl;
    cout << "gt residuals " << gt_residual << " var of gt " << getVar(curr_warpped_event_image_gt_C1, gt_nonzero) <<" non_zero " << gt_nonzero <<  endl;
    
    cout <<"count "<<eventBundle.coord.cols() << " sample count " << samples_count<< ", delta_time_range " << delta_time_range << "， warp_time " << warp_time <<endl;

}

/**
* \brief using derived jacobian and updated step size to evaluate current velocity and position.
*/
void System::EstimateMotion_kim()
{
    // paramters 
    int max_iter_count = 65;
        // velocity optimize steps and smooth factor
        double mu_event = 0.01, nu_event = 1; 
        double rho_event = 0.995, nu_map = 1;
        // position 
        double mu_map = 0.05;


    Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
    est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 

    for(int i=0; i< max_iter_count; i++)
    {
        // compute jacobian 
        Eigen::Vector3d jacobian = DeriveErrAnalytic(est_angleAxis, angular_position_compensator);
        // smooth factor
        double temp_jaco = jacobian.transpose()*jacobian; 
        nu_event =  temp_jaco*(1.0 - rho_event) + rho_event * nu_event;
        // update velocity TODO minus gradient ?? 
        angular_velocity_compensator =  mu_event / std::sqrt(nu_event) * jacobian;
        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true); 
        est_angleAxis = SO3add(est_angleAxis, angular_velocity_compensator, true); 

        getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32F);
        cv::imshow("opti", curr_warpped_event_image);
        cv::waitKey(100);

        cout << "iter " << i <<", scale " << mu_event / std::sqrt(nu_event) << endl;
        // cout << "  jacobian " << jacobian.transpose() << endl;
        cout << "  compensator   " << angular_velocity_compensator.transpose() << endl;
        int non_zero = 0;
        cout << "  var of est " << getVar(curr_warpped_event_image, non_zero) <<" non_zero " << non_zero <<  endl;
    }

    cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;
    cout << "gt angleAxis        " << gt_angleAxis.transpose() << endl;

} 




/**
* \brief given event_warpped_Bundle and rotation matrix, 
* \param vec_Bundle_Maps,  
* \param curr_map_image, output 
*/
void System::getMapImage()
{
    cout << "mapping global" << endl;
    
    /* warp current event to t0 using gt */
    Eigen::Matrix3d R_b2f = get_global_rotation_b2f(0,vec_gt_poseData.size()-1);
    Eigen::AngleAxisd angAxis_b2f(R_b2f);

    /* warp current event to t0 using estimated data */


    event_Map_Bundle.CopySize(event_warpped_Bundle);
    getWarpedEventPoints(event_warpped_Bundle,event_Map_Bundle,Eigen::Vector3d(0,0,0),angAxis_b2f.axis()*angAxis_b2f.angle());
    event_Map_Bundle.Projection(camera.eg_MapMatrix);
    event_Map_Bundle.DiscriminateInner(camera.width_map, camera.height_map);
    event_Map_Bundle.angular_position = angAxis_b2f.axis() * angAxis_b2f.angle(); 
    vec_Bundle_Maps.push_back(event_Map_Bundle);

    // cout << "test " << vec_Bundle_Maps[0].coord.topLeftCorner(2,5) << endl;



    /* get map from all events to t0 */
    cv::Mat temp_img; 
    curr_map_image.setTo(0);

    int start_idx = (vec_Bundle_Maps.size()-3) > 0 ? vec_Bundle_Maps.size()-3 : 0; 
    for(size_t i=start_idx; i<vec_Bundle_Maps.size(); i++)
    {
        // get 2d image 
        getImageFromBundle(vec_Bundle_Maps[i], PlotOption::U16C1_EVNET_IMAGE, true).convertTo(temp_img, CV_32F);
        // cout << "temp_img.size(), " << temp_img.size() << "type " << temp_img.type() << endl;
        
        curr_map_image += temp_img;

    }

    curr_map_image.convertTo(curr_map_image, CV_32F);

    // cout << "mask type " << mask.type() << "size " <<mask.size() <<  ", mask 5,5 : \n" << mask(cv::Range(1,6),cv::Range(1,6)) << endl;
    // cout << "curr_map_image type " << curr_map_image.type()<< ", curr_map_image size" << curr_map_image.size() << endl;
    // curr_map_image.setTo(255, mask);

    cout << "  get mapping sucess " << endl;
}
