
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
    // cout << "eventUndistorted.coord.cols() " << event_undis_Bundle.coord.cols() << endl;
    /* warp local events become sharper */
    event_out.CopySize(event_undis_Bundle);
    getWarpedEventPoints(event_undis_Bundle, event_out, cur_ang_vel); 
    event_out.Projection(camera.eg_cameraMatrix);
    event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(curr_warpped_event_image, CV_32F);

    return getImageFromBundle(event_out, option, false);

    // testing 

    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(0,10,0)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(y_img, CV_32F);
    // getWarpedEventPoints(event_undis_Bundle, event_out, Eigen::Vector3d(0,5,0)); 
    // event_out.Projection(camera.eg_cameraMatrix);
    // event_out.DiscriminateInner(camera.width, camera.height);
    // getImageFromBundle(event_out, option, false).convertTo(y5_img, CV_32F);

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
* \param cur_ang_vel angleAxis/delta_t from t2>t1, if add minus, it becomes t1->t2. 
* \param cur_ang_pos from t2->t0. default set (0,0,0) default, so the output is not rotated. 
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
    if(ang_vel_norm < 1e-8) 
    {
        // cout << "  small angle vec " << ang_vel_norm/3.14 * 180 << " degree /s" << endl;
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
        
        
        // second order version;
        // so x_t1 = x_t2*R{t2->t1}. 
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


Eigen::Vector3d System::GetGlobalTimeResidual()
{
    ;
}


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
    getWarpedEventPoints(event_undis_Bundle, event_warpped_Bundle, vel_angleAxis, Eigen::Vector3d::Zero(), -1);
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
        {
            cout << "idx " << idx << " x, y" << sampled_x << "," << sampled_y << endl;
            continue;
        }

        vec_sampled_idx_valid.push_back(idx);

        /* get warpped time residual, version 2 */

        double curr_t_residual = blur_image.at<float>(sampled_y, sampled_x);
        vec_residual.push_back(curr_t_residual);
        vec_Ix_interp.push_back(It_dx.at<float>(sampled_y, sampled_x));
        vec_Iy_interp.push_back(It_dy.at<float>(sampled_y, sampled_x));

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
* \brief given start and end ratio (0~1), and samples_count, return noise free sampled events. 
* \param vec_sampled_idx 
* \param samples_count 
*/
void System::getSampledVec(vector<int>& vec_sampled_idx, int samples_count, double sample_start, double sample_end)
{
    cv::RNG rng(int(ros::Time::now().nsec));
    // get samples, filtered out noise  
    for(int i=0; i< samples_count;)
    {
        int sample_idx = rng.uniform(int(event_undis_Bundle.size*sample_start), int(event_undis_Bundle.size*sample_end));

        // check valid 8 neighbor hood existed 
        int sampled_x = event_undis_Bundle.coord.col(sample_idx)[0];
        int sampled_y = event_undis_Bundle.coord.col(sample_idx)[1];
        if(sampled_x >= 239  ||  sampled_x < 1 || sampled_y >= 179 || sampled_y < 1 ) 
        {
            // cout << "x, y" << sampled_x << "," << sampled_y << endl;
            continue;
        }

        int count = 0;
        for(int j=-1; j<2; j++)
            for(int k=-1; k<2; k++)
            {
                count += (  curr_undis_event_image.at<cv::Vec3f>(sampled_y+j,sampled_x+k)[0] + 
                            curr_undis_event_image.at<cv::Vec3f>(sampled_y+j,sampled_x+k)[1] +
                            curr_undis_event_image.at<cv::Vec3f>(sampled_y+j,sampled_x+k)[2] ) > 0;
            }

        // valid 
        if(count > 4)
        {
            vec_sampled_idx.push_back(sample_idx);
            i++;
        }
    }
    
}

/**
* \brief using time as distance.
* \param warp_time_ratio warp time range * this ratio  
*/
void System::EstimateMotion_ransca_warp2bottom(double sample_start, double sample_end, double opti_steps)
{
    cout << "------------- sample_start " <<
        sample_start<< " sample_end " << sample_end<< ", step, " << opti_steps<< " ----" <<endl;
    
    cout << "using angle axis " << est_angleAxis.transpose() << endl;
    // warp events and get a timesurface image with indexs 
    cv::Mat cv_timesurface; 
    cv_timesurface = getImageFromBundle(event_undis_Bundle, PlotOption::TIME_SURFACE);  
    cv::normalize(cv_timesurface, hot_image_C1, 0,255, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(hot_image_C1, hot_image_C3, cv::COLOR_GRAY2BGR);


    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx; 
    int samples_count = std::min(10000,int(event_undis_Bundle.size * (sample_end-sample_start)));
    getSampledVec(vec_sampled_idx, samples_count, sample_start, sample_end);


    // otimizing paramter init
        int max_iter_count = 300;
        // velocity optimize steps and smooth factor
        double mu_event = opti_steps, nu_event = 1;
        Eigen::Vector3d adam_v(0,0,0); 

        double rho_event = 0.995, nu_map = 1;
        Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
    
    // test last est and {0,0,0}, choose init velocity
    {
        double residual1 = 0, residual2 = 0;
        DeriveTimeErrAnalyticRansacBottom(est_angleAxis, vec_sampled_idx, residual1);
        DeriveTimeErrAnalyticRansacBottom(Eigen::Vector3d(0,0,0), vec_sampled_idx, residual2);
        if(residual1 > residual2)
        {
            cout << "at time " <<(eventBundle.first_tstamp - eventBundle.last_tstamp).toSec() << 
            ", using {0,0,0"<< endl;
            est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 

        }
    }

    double residuals = 0, pre_residual = 10; 
    int sample_output = 2;
    for(int i=0; i< max_iter_count; i++)
    {
        pre_residual = residuals;
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

        if(angular_velocity_compensator.norm() < 0.001) break;

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
