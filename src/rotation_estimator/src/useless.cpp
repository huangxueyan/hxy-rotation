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
