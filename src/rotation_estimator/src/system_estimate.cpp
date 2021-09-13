
#include "system.hpp"
#include <sophus/so3.hpp>

using namespace std;


/**
* \brief given angular veloity, warp local event bundle to the reference time 
*/
void System::getWarpedEventImage(const Eigen::Vector3f & cur_ang_vel, const PlotOption& option)
{
    cout << "get warpped event image " << endl;

    getWarpedEventPoints(cur_ang_vel); 
    event_warpped_Bundle.Projection(camera.eg_cameraMatrix);

    // event_warpped_Bundle.DiscriminateInner(camera.width-1, camera.height-1);

    getImageFromBundle(event_warpped_Bundle, curr_warpped_event_image); 

}

/**
* \brief given angular veloity, warp local event bundle to the reference time
    using kim RAL21, eqation(11), since the ratation angle is ratively small 
        w_hat << 0, -w(2), w(1),
        w(2), 0, -w(0),
        -w(1), w(0), 0;
*/
void System::getWarpedEventPoints(const Eigen::Vector3f& cur_ang_vel, const Eigen::Vector3f& cur_ang_pos)
{
    // the theta of rotation axis
    float ang_vel_norm = cur_ang_vel.norm(); 

    Eigen::Matrix3Xf ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
    
    ang_vel_hat_mul_x.conservativeResize(3,event_undis_Bundle.size);     // row, col 
    ang_vel_hat_sqr_mul_x.conservativeResize(3,event_undis_Bundle.size); 
    
    // equation 11 
    ang_vel_hat_mul_x.row(0) = -cur_ang_vel(2)*event_undis_Bundle.coord_3d.row(1) + cur_ang_vel(1)*event_undis_Bundle.coord_3d.row(2);
    ang_vel_hat_mul_x.row(1) =  cur_ang_vel(2)*event_undis_Bundle.coord_3d.row(0) - cur_ang_vel(0)*event_undis_Bundle.coord_3d.row(2);
    ang_vel_hat_mul_x.row(2) = -cur_ang_vel(1)*event_undis_Bundle.coord_3d.row(0) + cur_ang_vel(0)*event_undis_Bundle.coord_3d.row(1);

    ang_vel_hat_sqr_mul_x.row(0) = -cur_ang_vel(2)*ang_vel_hat_mul_x.row(1) + cur_ang_vel(1)*ang_vel_hat_mul_x.row(2);
    ang_vel_hat_sqr_mul_x.row(1) =  cur_ang_vel(2)*ang_vel_hat_mul_x.row(0) - cur_ang_vel(0)*ang_vel_hat_mul_x.row(2);
    ang_vel_hat_sqr_mul_x.row(2) = -cur_ang_vel(1)*ang_vel_hat_mul_x.row(0) + cur_ang_vel(0)*ang_vel_hat_mul_x.row(1);


    event_warpped_Bundle.CopySize(event_undis_Bundle);
    if(ang_vel_norm/3.14 * 180 < 0.1) 
    {
        cout << "  small angle vec " << ang_vel_norm/3.14 * 180 << " degree /s" << endl;
        event_warpped_Bundle.coord_3d = event_undis_Bundle.coord_3d;
    }
    else
    {   
        auto delta_time = eventBundle.time_delta;  
        
        // second order version 
        // event_warpped_Bundle.coord_3d = event_undis_Bundle.coord_3d
        //                             + Eigen::MatrixXf( 
        //                                 ang_vel_hat_mul_x.array().rowwise() 
        //                                 * delta_time.transpose().array()
        //                                 + ang_vel_hat_sqr_mul_x.array().rowwise() 
        //                                 * (0.5f * delta_time.transpose().array().square()) );
        
        // first order version 
        event_warpped_Bundle.coord_3d = event_undis_Bundle.coord_3d ;
                                    // + Eigen::MatrixXf( 
                                    //     ang_vel_hat_mul_x.array().rowwise() 
                                    //     * delta_time.transpose().array());
                                        
        // cout << "angle vec: " << (cur_ang_vel.array()/3.14 * 180).transpose() << " degree/s" << endl;
        // cout << "ang_vel_hat_mul_x vec: \n"<< ang_vel_hat_mul_x.topLeftCorner(3,5) << endl;
        // cout << "delta time: \n" << delta_time.topRows(10).transpose()<< endl;
        // cout << "angular * delta time: \n" << (ang_vel_hat_mul_x.array().rowwise() 
        //                                 * delta_time.transpose().array()).topLeftCorner(3,5)<< endl; 
        // cout << "delta time: \n" << delta_time.topRows(5).transpose() << endl;
        // cout << "ang_vel_hat_mul_x : \n" << ang_vel_hat_mul_x.topLeftCorner(3,5) << endl;; 
    }

    // cout << "event_undis_Bundle \n" << event_undis_Bundle.coord_3d.topLeftCorner(3,5) << endl;
    // cout << "event_warpped_Bundle \n" << event_warpped_Bundle.coord_3d.topLeftCorner(3,5) << endl;
    

    // cout << "angle velocity: " <<  (cur_ang_vel.array() / 3.14 * 180).transpose() << " degrees / second." << endl;
    // TODO calculate true rotation matrix results 

}
