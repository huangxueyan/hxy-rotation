// TODO global residual map 
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
