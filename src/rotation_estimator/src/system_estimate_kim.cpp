
#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>

using namespace std;



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
    int threshold = 5; 
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
        // Ix_interp(i) = curr_warpped_event_image.at<float>(y,x) * Ix.at<float>(y,x);  
        // Iy_interp(i) = curr_warpped_event_image.at<float>(y,x) * Iy.at<float>(y,x);
        
        Ix_interp(i) = Ix.at<float>(y,x);  
        Iy_interp(i) = Iy.at<float>(y,x);
        warped_image_delta_t(i) = eventBundle.time_delta(vec_valid[i]) 
                                * curr_warpped_event_image.at<float>(y, x);

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
    // cout << "eg_jacobian " << eg_jacobian.topLeftCorner(1,5) <<endl;
    // cout << "_delta_time " << _delta_time_valid.topRows(5).transpose()  <<endl;
    // cout << "curr_warpped_event_image :\n" << curr_warpped_event_image(cv::Range(100, 110), cv::Range(100, 110)) << endl;
    
    
    Eigen::Vector3d jacobian;

    // jacobian(0) = eg_jacobian.row(0) * _delta_time_valid;
    // jacobian(1) = eg_jacobian.row(1) * _delta_time_valid;
    // jacobian(2) = eg_jacobian.row(2) * _delta_time_valid;

    jacobian(0) = eg_jacobian.row(0) * warped_image_delta_t;
    jacobian(1) = eg_jacobian.row(1) * warped_image_delta_t;
    jacobian(2) = eg_jacobian.row(2) * warped_image_delta_t;

    // cout << "gradient " << jacobian.transpose() << endl;
    return jacobian;
}



/**
* \brief using derived jacobian and updated step size to evaluate current velocity and position.
*/
void System::EstimateMotion_kim()
{
    // paramters 
    int max_iter_count = 200;
        // velocity optimize steps and smooth factor
        double mu_event = 0.03, nu_event = 1; 
        double rho_event = 0.995;
        // position 
        double mu_map = 0.05, nu_map = 1;


    Eigen::Vector3d angular_velocity_compensator(0,0,0), angular_position_compensator(0,0,0);
    est_angleAxis = Eigen::Vector3d(0,0,0); // set to 0. 

    int output_sample = 10;
    for(int i=0; i< max_iter_count; i++)
    {
        // compute jacobian 
        Eigen::Vector3d jacobian = DeriveErrAnalytic(est_angleAxis, angular_position_compensator);
        // smooth factor
        double temp_jaco = jacobian.transpose()*jacobian; 
        nu_event =  temp_jaco*(1.0 - rho_event) + rho_event * nu_event;
        // update velocity from t2->t1.  
        angular_velocity_compensator = mu_event / std::sqrt(nu_event + 1e-8) * jacobian;
        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true);
        est_angleAxis = SO3add(angular_velocity_compensator, est_angleAxis , true); 
        

        if(angular_velocity_compensator.norm() < 0.001) break; // early break

        // visualize
        if(false && i % output_sample == 0)
        {
            getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32F);
            cv::imshow("opti", curr_warpped_event_image);
            cv::waitKey(10);

            cout << "iter " << i <<", scale " << mu_event / std::sqrt(nu_event) << endl;
            // cout << "  jacobian " << jacobian.transpose() << endl;
            cout << "  compensator   " << angular_velocity_compensator.transpose() << endl;
            int non_zero = 0;
            cout << "  var of est " << getVar(curr_warpped_event_image, non_zero, CV_32F) <<" non_zero " << non_zero <<  endl;
            
        }
    }

    // getWarpedEventImage(est_angleAxis, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32F);
    // // cv::normalize(curr_warpped_event_image, curr_warpped_event_image, 0, 255, cv::NORM_MINMAX, CV_8UC3);
    // cv::imshow("opti", curr_warpped_event_image/3);
    // cv::waitKey(10);

    // cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;
    // cout << "gt angleAxis   " << gt_angleAxis.transpose() << endl;

} 

