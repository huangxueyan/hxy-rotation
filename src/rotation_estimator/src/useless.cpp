// TODO global residual map 
void get_global_residual_map()
{
    cv::Mat residual_map(40,40,CV_32F); 
    cv::Mat contrast_map(40,40,CV_32F); 
    cv::Mat _temp_Mat; 
    cout << "counting redisual map " << endl;
    for(double z=-2; z<2; z+=0.5)
    {
        for(double x=-2; x<2; x+=0.5)
        for(double y=-2; y<2; y+=0.5)
        {
            Eigen::Vector3d vec(x,y,z);
            DeriveTimeErrAnalyticRansac(vec, vec_sampled_idx, warp_time, gt_residual);
            getWarpedEventImage(vec, event_warpped_Bundle_gt).convertTo(_temp_Mat, CV_32F);
            cout << "DeriveTimeErrAnalyticRansca " << gt_residual << endl;
            cout << "var of gt " << getVar(_temp_Mat) << endl;
            residual_map.at<float>(int(y*10+20),int(x*10+20)) = gt_residual;
            // contrast_map.at<float>(int(y*10+20),int(x*10+20)) = getVar(_temp_Mat); 
        }   
        cv::Mat color_const, color_residual ;
        cv::normalize(residual_map, color_residual, 0, 255, cv::NORM_MINMAX, CV_8U);
        // cv::applyColorMap(contrast_map, color_const, cv::COLORMAP_JET);
        cv::applyColorMap(color_residual, color_residual, cv::COLORMAP_JET);
        // cv::imshow(" contrast_map ", color_const);
        cv::imshow(" residual_map ", color_residual);
        cv::waitKey(0);
    }
}



/**
* \brief using time as distance, random warptime.
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
    cv::normalize(cv_timesurface, hot_image_C1, 0,255, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(hot_image_C1, hot_image_C3, cv::COLOR_GRAY2BGR);

    // select 100 random points, and warp delta_t < min(t_point_delta_t). 
    // accumulate all time difference before and after warpped points. 
    std::vector<int> vec_sampled_idx;
    cv::RNG rng(int(ros::Time::now().nsec));
    int samples_count = std::min(1000,int(event_undis_Bundle.coord.cols()/2));

    // get samples 
    for(int i=0; i< samples_count;)
    {
        int sample_idx = rng.uniform(int(event_undis_Bundle.coord.cols()*sample_ratio), event_undis_Bundle.coord.cols());

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

        // cout << "x, y " << count << curr_undis_event_image.at<cv::Vec3f>(sampled_y,sampled_x) << endl;
        // valid 
        if(count > 4)
        {
            vec_sampled_idx.push_back(sample_idx);
            i++;
        }
    }
    
    // get warp time range  
    std::vector<double>  vec_sampled_time;
    for(const int& i: vec_sampled_idx)
        vec_sampled_time.push_back(eventBundle.time_delta(i));

    double delta_time_range = *(std::min_element(vec_sampled_time.begin(), vec_sampled_time.end()));
    double warp_time = delta_time_range * warp_time_ratio; 
    cout <<"sample count " << samples_count << ", delta_time_range " << delta_time_range << "， warp_time " << warp_time <<endl;

    // otimizing paramter init
        int max_iter_count = 30;
        // velocity optimize steps and smooth factor
        double mu_event = opti_steps, nu_event = 1;
        Eigen::Vector3d adam_v(0,0,0); 

        double rho_event = 0.9, nu_map = 1;
        Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
        // est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 

    double residuals = 0, pre_residual = 10; 
    for(int i=0; i< max_iter_count; i++)
    // while(abs(residuals - pre_residual) > 0.1)
    {
        pre_residual = residuals;
        // compute jacobian 
        // Eigen::Vector3d jacobian = DeriveTimeErrAnalyticLayer(est_angleAxis, vec_sampled_idx, warp_time, residuals);
        Eigen::Vector3d jacobian = DeriveTimeErrAnalyticRansac(est_angleAxis, vec_sampled_idx, warp_time, residuals);

        // smooth factor, RMS Prob 
            // double temp_jaco = jacobian.transpose()*jacobian; 
            // nu_event =  temp_jaco*(1.0 - rho_event) + rho_event * nu_event;
            // angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * jacobian;

        // smooth factor, Adam 
            adam_v = 0.9*adam_v + (1-0.9) * jacobian;
            nu_event =  (1.0-rho_event) * jacobian.transpose()*jacobian + rho_event*nu_event;
            angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * adam_v;

        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true); 
        est_angleAxis = SO3add(angular_velocity_compensator, est_angleAxis, true); 
        // est_angleAxis = est_angleAxis.array() + angular_velocity_compensator.array(); 

        getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);
        cv::imshow("opti", curr_warpped_event_image);
        cv::waitKey(10);

        cout << "iter " << i <<", nu_event " << std::sqrt(nu_event) << endl;
        // cout << "  jacobian " << jacobian.transpose() << endl;
        cout << "  compensator   " << angular_velocity_compensator.transpose() << endl;
        cv::Mat curr_warpped_event_image_c1 = getWarpedEventImage(est_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE);
        int est_nonzero = 0;
        cout << "  residuals " << residuals << ", var of est " << getVar(curr_warpped_event_image_c1, est_nonzero, CV_16U) <<" non_zero " <<est_nonzero <<  endl;
    }

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

    // compare with gt
    cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;   
    cout <<"total count "<<eventBundle.coord.cols() << " sample " << samples_count<< ", time_range " << delta_time_range << "， warp_time " << warp_time <<endl;
    
    // compare with gt 
    if(using_gt)
    {
        double gt_residual = 0;
        int gt_nonzero = 0; 
        // DeriveTimeErrAnalyticLayer(gt_angleAxis, vec_sampled_idx, warp_time, gt_residual);
        // cout << "DeriveTimeErrAnalyticLayer " << gt_residual << endl;
        DeriveTimeErrAnalyticRansac(gt_angleAxis, vec_sampled_idx, warp_time, gt_residual);
        cout << "DeriveTimeRansca gt residual " << gt_residual << endl;
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
        cout << "est angleAxis  " << est_angleAxis.transpose() << endl;
        cout << "gt residuals " << gt_residual << " var of gt " << getVar(curr_warpped_event_image_gt_C1, gt_nonzero, CV_16U) <<" non_zero " << gt_nonzero <<  endl;
    }
}





/**
* \brief update eventBundle.
*/
void System::updateEventBundle()
{
    double max_interval = 0.03;

    // save in queue eventData and event bundle (local events)
    if(vec_last_event.empty())
    {
        std::vector<dvs_msgs::Event>& curr_vec_event = que_vec_eventData.front();
        que_vec_eventData.pop(); 
        assert((curr_vec_event.back().ts-curr_vec_event.front().ts).toSec() > max_interval);
        vec_last_event = curr_vec_event;
        vec_last_event_idx = 2; 
    }


    ros::Time begin_t = vec_last_event[vec_last_event_idx].ts; 

    if((vec_last_event.back().ts-begin_t).toSec() > max_interval)  // not need to used new event que element. 
    {
        // bool flag = false;
        if((vec_last_event.back().ts-begin_t).toSec() > 10)
        {
            for(int i=0; i< 10; i++)
            {
                cout << "begin_t i" << i << "," << vec_last_event[i].ts.sec << "." <<  vec_last_event[i].ts.nsec << endl;
            }
            cout << "vec_last_event_idx " << vec_last_event_idx 
                << "back " << vec_last_event.back().ts.sec << "." <<  vec_last_event.back().ts.nsec << 
                    " size" << vec_last_event.size() << endl;
        }


        for(int i=vec_last_event_idx+1; i<vec_last_event.size(); i++)
        {
            if((vec_last_event[i].ts-begin_t).toSec() > max_interval)
            {
                // flag = true;
                std::vector<dvs_msgs::Event> input_event(&vec_last_event[vec_last_event_idx], &vec_last_event[i]); 
                eventBundle.Append(input_event);       // in event bundle 
                vec_last_event_idx = i;
                cout << "current pack is enough " << (vec_last_event[i].ts-begin_t).toSec() <<", count " << input_event.size() << endl;
                break; 
            }
        }
    }
    else 
    {
        // recovery old events in vec_last_event
        // extract new events 
        std::vector<dvs_msgs::Event> curr_vec_event = que_vec_eventData.front();
        que_vec_eventData.pop();    // guarantee the earlier events are processed                          

        std::vector<dvs_msgs::Event> input_event(&vec_last_event[vec_last_event_idx], &vec_last_event[vec_last_event.size()]);
        
        for(int i=0; i<curr_vec_event.size(); i++)
        {
            if((curr_vec_event[i].ts-begin_t).toSec() > max_interval)
            {
                vec_last_event_idx = i;
                break; 
            }
        }
        
        input_event.insert(input_event.end(), curr_vec_event.begin(), curr_vec_event.begin()+vec_last_event_idx);
        eventBundle.Append(input_event);       // in event bundle 
        vec_last_event = curr_vec_event;

        cout << "current pack not enough " << (curr_vec_event[vec_last_event_idx].ts-begin_t).toSec() <<", count " << input_event.size() << endl;
    }
    
    cout << "input_event " << eventBundle.size << endl;
    cout << "-------end of update -------" << endl;

}



/**
* \brief using time as distance .
* r(delta_theta) = sum(delta_t**2); like eq(8) get jacobian function 
*/
Eigen::Vector3d System::DeriveTimeErrAnalyticLayer(const Eigen::Vector3d &vel_angleAxis, 
    const std::vector<int>& vec_sampled_idx, double warp_time, double& total_residual)
{

    cout << "using DeriveTimeErrAnalyticLayer " << vel_angleAxis.transpose() 
            << " time " << warp_time << endl;

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

    cout << "Ix_interp " << Ix_interp.topLeftCorner(1,5) << endl;
    cout << "Iy_interp " << Iy_interp.topLeftCorner(1,5) << endl;

    
    eg_jacobian.row(0) = -Ix_interp.array()*x_z.array()*y_z.array() 
                        - Iy_interp.array()*(1+y_z.array()*y_z.array());

    eg_jacobian.row(1) = Ix_interp.array()*(1+x_z.array()*x_z.array()) 
                        + Iy_interp.array()*x_z.array()*y_z.array();
    
    eg_jacobian.row(2) = -Ix_interp.array()*y_z.array() 
                        + Iy_interp.array()*x_z.array();    
    
    cout << "eg_jacobian " << eg_jacobian.topLeftCorner(3,5) << endl;
    
    
    Eigen::Vector3d jacobian;
    jacobian(0) = eg_jacobian.row(0) * Eigen::VectorXd::Ones(valid_size);
    jacobian(1) = eg_jacobian.row(1) * Eigen::VectorXd::Ones(valid_size);
    jacobian(2) = eg_jacobian.row(2) * Eigen::VectorXd::Ones(valid_size);

    cout << "jacobian " << jacobian.transpose() << endl;

    return jacobian;
}
