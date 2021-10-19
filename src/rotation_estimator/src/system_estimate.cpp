
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
* \param delta_time all events warp this time, if delta_time<0, warp to t0. 
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


