
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

using namespace std;


/**
* \brief given angular veloity(t1->t2), warp local event bundle become shaper
*/
cv::Mat System::getWarpedEventImage(const Eigen::Vector2d & cur_trans_vel, EventBundle& event_out,  const PlotOption& option, bool ref_t1)
{
    // cout << "get warpped event image " << endl;
    // cout << "eventUndistorted.coord.cols() " << event_undis_Bundle.coord.cols() << endl;
    /* warp local events become sharper */
    ros::Time t1, t2, t3;

    event_out.CopySize(event_undis_Bundle);
    // t1 = ros::Time::now(); 

    getWarpedEventPoints(event_undis_Bundle, event_out, cur_trans_vel, ref_t1); 
    // t2= ros::Time::now(); 
    
    event_out.Projection(camera.eg_cameraMatrix);

    event_out.DiscriminateInner(camera.width, camera.height);
    // t3 = ros::Time::now(); 


    // cout << "   getWarpedEventPoints time " << (t2-t1).toSec() << endl;
    // cout << "   DiscriminateInner time " << (t3-t2).toSec() << endl;
    return getImageFromBundle(event_out, option);
}

/**
* \brief given angular veloity, warp local event bundle(t2) to the reference time(t1)
    using kim RAL21, eqation(11), since the ratation angle is ratively small
* \param cur_ang_vel angleAxis/delta_t from t1->t2, if multiply minus time, it becomes t2->t1. (AngleAxis inverse only add minus) 
* \param cur_ang_pos from t2->t0. default set (0,0,0) default, so the output is not rotated. 
* \param delta_time all events warp this time, if delta_time<0, warp to t1. 
*/
void System::getWarpedEventPoints(const EventBundle& eventIn, EventBundle& eventOut, 
    const Eigen::Vector2d& cur_trans_vel,  bool ref_t1)
{
    // cout << "projecting " << endl;
    // the theta of rotation axis

        // cout << "using whole warp "  <<endl;
        Eigen::VectorXd vec_delta_time = eventBundle.time_delta;  // positive 
        if(ref_t1) vec_delta_time = eventBundle.time_delta.array() - eventBundle.time_delta(eventBundle.size-1);   // negative 


        eventOut.coord_3d.row(0) = eventIn.coord_3d.row(0).array() + vec_delta_time.transpose().array() * cur_trans_vel(0);
        eventOut.coord_3d.row(1) = eventIn.coord_3d.row(1).array() + vec_delta_time.transpose().array() * cur_trans_vel(1);
        eventOut.coord_3d.row(2) = eventIn.coord_3d.row(2);
                

        // cout << "usingg est" << cur_ang_vel.transpose() << endl;
        // cout << "original  " << eventIn.coord_3d.topLeftCorner(3,5)<< endl;
        // cout << "ang_vel_hat_mul_x " << ang_vel_hat_mul_x.topLeftCorner(3,5)<< endl;
        // cout << "delta time " << vec_delta_time.topRows(5).transpose() << endl;
        // cout << "final \n" << eventOut.coord_3d.topLeftCorner(3,5) <<  endl;

        // rodrigues version wiki
            // Eigen::Matrix<double,3,1> axis = cur_ang_vel.normalized();
            // Eigen::VectorXd angle_vec = vec_delta_time * ang_vel_norm ;

            // Eigen::VectorXd cos_angle_vec = angle_vec.array().cos();
            // Eigen::VectorXd sin_angle_vec = angle_vec.array().sin();

            // Eigen::Matrix3Xd first = eventIn.coord_3d.array().rowwise() * cos_angle_vec.transpose().array(); 
            // Eigen::Matrix3Xd second = (-eventIn.coord_3d.array().colwise().cross(axis)).array().rowwise() * sin_angle_vec.transpose().array();
            // Eigen::VectorXd third1 = axis.transpose() * eventIn.coord_3d;
            // Eigen::VectorXd third2 = third1.array() * (1-cos_angle_vec.array()).array();;
            // Eigen::Matrix3Xd third = axis * third2.transpose();
            // eventOut.coord_3d = first + second + third; 

        // cout << "last \n " << eventOut.coord_3d.bottomRightCorner(3,5) <<  endl;

    

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
                // count += (  curr_undis_event_image.at<float>(sampled_y+j,sampled_x+k) != default_value)
            }

        // valid denoised
        if(count > yaml_denoise_num)  // TODO 
        {
            vec_sampled_idx.push_back(sample_idx);
            i++;
        }
    }
    
}
