
#include "system.hpp"
#include <sophus/so3.hpp>

using namespace std;


/**
* \brief given angular veloity(t1->t2), warp local event bundle become shaper
*/
void System::getWarpedEventImage(const Eigen::Vector3d & cur_ang_vel, const PlotOption& option)
{
    // cout << "get warpped event image " << endl;

    /* warp local events become sharper */
    event_warpped_Bundle.CopySize(event_undis_Bundle);
    getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, cur_ang_vel); 
    event_warpped_Bundle.Projection(camera.eg_cameraMatrix);
    event_warpped_Bundle.DiscriminateInner(camera.width, camera.height);

    getImageFromBundle(event_warpped_Bundle, option, false).convertTo(curr_warpped_event_image, CV_32F);

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
            cout <<"using const delta " << delta_time << endl;
        }
        
        
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
}


/**
* \brief calculate jacobian of motion model.
* \param vel_angleAxis estimate angular velocity. 
* \param pos_angleAxis estimate pos compensate for const vel model. 
*/
Eigen::Vector3d System::DeriveErrAnalytic(const Eigen::Vector3d &vel_angleAxis, const Eigen::Vector3d &pos_angleAxis)
{
    getWarpedEventImage(vel_angleAxis, PlotOption::U16C1_EVNET_IMAGE);

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
double System::getTimeResidual(int sampled_x, int sampled_y, double sampled_time, double warp_time)
{
    // check 
    assert(sampled_x>-1);
    assert(sampled_y>-1);
    assert(sampled_x<240);
    assert(sampled_y<180);

    // get specific location 
    double current_residual = 1e4;
    for(int i=0; i<cv_3D_surface_index_count.at<int>(sampled_y,sampled_x); i++)
    {
        double iter_time = eventBundle.time_delta(cv_3D_surface_index.at<int>(sampled_y,sampled_x,i));
        current_residual = std::min(current_residual, sampled_time-warp_time-iter_time) ;
        int iter_x = event_undis_Bundle.coord(0,cv_3D_surface_index.at<int>(sampled_y,sampled_x,i));
        int iter_y = event_undis_Bundle.coord(1,cv_3D_surface_index.at<int>(sampled_y,sampled_x,i));
        // cout << "x, y " << iter_x <<"," <<iter_y  <<", iter_time " << iter_time << endl; 
    }

    if(cv_3D_surface_index_count.at<int>(sampled_y, sampled_x) == 0) 
    {
        // current_residual = sampled_time-warp_time;
        current_residual = 0.1;
    }


    // cout << "sample_time " << sampled_time << " warp_time " << warp_time 
    //     << " current_residual " << current_residual <<  endl; 

    return std::abs(current_residual);
}


/**
* \brief using time as distance .
* r(delta_theta) = sum(delta_t**2); like eq(8) get jacobian function 
*/
Eigen::Vector3d System::DeriveTimeErrAnalytic(const Eigen::Vector3d &vel_angleAxis, 
    const std::vector<int>& vec_sampled_idx, double warp_time, double& total_residual)
{

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

        // get warpped time residual  
        double curr_t_residual = getTimeResidual(sampled_x, sampled_y,sampled_time, warp_time);

        // TODO get gradient of warpped time residual 
        double curr_t_residual_x_left  = getTimeResidual(sampled_x-1, sampled_y,sampled_time, warp_time);
        double curr_t_residual_x_right = getTimeResidual(sampled_x+1, sampled_y,sampled_time, warp_time);
        double curr_t_residual_y_up    = getTimeResidual(sampled_x, sampled_y-1,sampled_time, warp_time);  // TODO check 
        double curr_t_residual_y_down  = getTimeResidual(sampled_x, sampled_y+1,sampled_time, warp_time);

        vec_residual.push_back(curr_t_residual);
        vec_Ix_interp.push_back((curr_t_residual_x_right-curr_t_residual_x_left)/2);
        vec_Iy_interp.push_back((curr_t_residual_y_up-curr_t_residual_y_down)/2);

        total_residual += curr_t_residual;
    }
    // cout << "total_residual " << total_residual << endl;

    // Eigen::Map<VectorXd> eg_residual(&vec_residual[0], vec_residual.size());
    // Eigen::Map<VectorXd> Ix_interp(&vec_Ix_interp[0], vec_Ix_interp.size());
    // Eigen::Map<VectorXd> Iy_interp(&vec_Iy_interp[0], vec_Iy_interp.size());

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
*/
void System::EstimateMotion_ransca()
{
    // warp events and get a timesurface image with indexs 
    cv::Mat timesurface; 
    timesurface = getImageFromBundle(event_undis_Bundle, PlotOption::TIME_SURFACE);  

    // cv::Mat hot_image for visualization ;
    cv::normalize(timesurface, hot_image, 0,255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(hot_image, hot_image, cv::COLORMAP_JET);

    cv::cvtColor(hot_image, hot_image, cv::COLOR_BGR2GRAY);
    cv::cvtColor(hot_image, hot_image, cv::COLOR_GRAY2BGR);
    // cv::imshow("timesurface", hot_image);
    // cv::waitKey(100);


    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx;
    cv::RNG rng(0XFFFF);
    int samples_count = std::min(1000,int(event_undis_Bundle.coord.cols()/2));
    for(int i=0; i< samples_count; i++)
    {
        int sample_idx = rng.uniform(int(event_undis_Bundle.coord.cols()/2), event_undis_Bundle.coord.cols());
        vec_sampled_idx.push_back(sample_idx);

        // viusal sample 
        int x = int(event_undis_Bundle.coord(0, sample_idx));
        int y = int(event_undis_Bundle.coord(1, sample_idx));
        hot_image.at<cv::Vec3b>(y,x) += cv::Vec3b(0,255,0);   // green of original 

        // hot_image.at<cv::Vec3b>(sampled_y,sampled_x) += cv::Vec3b(0,0,255);   // red of warpped 
    }
    
    // warp time range  
    std::vector<double>  vec_sampled_time;
    for(const int& i: vec_sampled_idx)
        vec_sampled_time.push_back(eventBundle.time_delta(i));
    double delta_time_range = *(std::min_element(vec_sampled_time.begin(), vec_sampled_time.end()));
    double warp_time = delta_time_range / 1.0; 
    cout << "delta_time_range " << delta_time_range << "， warp_time " << warp_time <<endl;

    // otimizing 
        int max_iter_count = 100;
        // velocity optimize steps and smooth factor
        double mu_event = 0.05, nu_event = 1; 
        double rho_event = 0.995, nu_map = 1;


    Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
    est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 

    double residuals = 0, pre_residual = 10; 
    for(int i=0; i< max_iter_count; i++)
    // while(abs(residuals - pre_residual) > 0.1)
    {
        pre_residual = residuals;
        // compute jacobian 
        Eigen::Vector3d jacobian = DeriveTimeErrAnalytic(est_angleAxis, vec_sampled_idx, warp_time, residuals);
        // smooth factor
        double temp_jaco = jacobian.transpose()*jacobian; 
        nu_event =  temp_jaco*(1.0 - rho_event) + rho_event * nu_event;
        // update velocity TODO minus gradient ?? 
        angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * jacobian;

        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true); 
        // cout << "compensator   " << angular_velocity_compensator.transpose() << endl;
        // cout << "est_angleAxis " << est_angleAxis.transpose() << endl;
        // est_angleAxis = SO3add(angular_velocity_compensator, est_angleAxis, true); 
        est_angleAxis = est_angleAxis.array() + angular_velocity_compensator.array(); 
        // cout << "est_angleAxis " << est_angleAxis.transpose() << endl;


        // cout << "nu_event " << nu_event <<"," << "rho_event " <<rho_event << endl;
        cout << "iter " << i <<", scale " << mu_event / std::sqrt(nu_event) << endl;
        cout << "  jacobian " << jacobian.transpose() << endl;
        cout << "  residuals " << residuals << endl; 

        getWarpedEventImage(est_angleAxis);
        cv::imshow("opti", curr_warpped_event_image);
        cv::waitKey(100);

    }

    // visualize 
    for(int i=0; i<vec_sampled_idx.size(); i++)
    {
         // viusal sample 
        int x = int(event_warpped_Bundle.coord(0, vec_sampled_idx[i]));
        int y = int(event_warpped_Bundle.coord(1, vec_sampled_idx[i]));
        hot_image.at<cv::Vec3b>(y,x) += cv::Vec3b(0,0,255);   // red of warpped 
    }

    // compare with gt
    cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;
    cout << "gt angleAxis        " << gt_angleAxis.transpose() << endl;
    DeriveTimeErrAnalytic(gt_angleAxis, vec_sampled_idx, warp_time, residuals);
    cout << "gt residuals " << residuals << endl; 

    // DeriveTimeErrAnalytic(gt_angleAxis*2, vec_sampled_idx, warp_time, residuals);
    // cout << "gt2 residuals " << residuals << endl; 
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

        // cout << "nu_event " << nu_event <<"," << "rho_event " <<rho_event << endl;
        cout << "scale " << mu_event / std::sqrt(nu_event) << endl;
        cout << "jacobian " << jacobian.transpose() << endl;

        getWarpedEventImage(est_angleAxis);
        cv::imshow("opti", curr_warpped_event_image);
        cv::waitKey(100);

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
