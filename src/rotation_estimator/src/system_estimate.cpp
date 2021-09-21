
#include "system.hpp"
#include <sophus/so3.hpp>

using namespace std;


/**
* \brief given angular veloity, warp local event bundle become shaper
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
* \brief given angular veloity, warp local event bundle to the reference time
    using kim RAL21, eqation(11), since the ratation angle is ratively small
* \param cur_ang_vel angleAxis/delta_t, so if it is zero, not shapper operation is done in delta time. 
* \param cur_ang_pos are set (0,0,0) default, so the output is not rotated. 
*/
void System::getWarpedEventPoints(const EventBundle& eventIn, EventBundle& eventOut, const Eigen::Vector3d& cur_ang_vel, const Eigen::Vector3d& cur_ang_pos)
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
        Eigen::VectorXd delta_time = eventBundle.time_delta;  
        
        // second order version 
        eventOut.coord_3d = eventIn.coord_3d
                                    + Eigen::MatrixXd( 
                                        ang_vel_hat_mul_x.array().rowwise() 
                                        * (-delta_time.transpose().array())
                                        + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                        * (0.5f * delta_time.transpose().array().square()) );
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
* \brief CM method self diff version .
*/
class CMCostAnalytic: public ceres::SizedCostFunction<1, 3> {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    CMCostAnalytic(const Eigen::Matrix3Xd& coord_3d, const Eigen::VectorXd& delta_time, const Eigen::Matrix3d& K)
        :_coord_3d(coord_3d), _delta_time(delta_time), _intrisic(K){
            cout << "CM loss init :" << endl;
            counter = 0;}

    virtual ~CMCostAnalytic() {} 
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const 
    {

        Eigen::Matrix<double, 3, -1> ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
        ang_vel_hat_mul_x.resize(3,_coord_3d.cols());
        ang_vel_hat_sqr_mul_x.resize(3,_coord_3d.cols());
        
        const double ag_x = parameters[0][0], ag_y = parameters[0][1], ag_z = parameters[0][2] ;

        // equation 11 
        ang_vel_hat_mul_x.row(0) = -ag_z*_coord_3d.row(1) + ag_y*_coord_3d.row(2);
        ang_vel_hat_mul_x.row(1) =  ag_z*_coord_3d.row(0) - ag_x*_coord_3d.row(2);
        ang_vel_hat_mul_x.row(2) = -ag_y*_coord_3d.row(0) + ag_x*_coord_3d.row(1);

        ang_vel_hat_sqr_mul_x.row(0) = -ag_z*ang_vel_hat_mul_x.row(1)+ ag_y*ang_vel_hat_mul_x.row(2);
        ang_vel_hat_sqr_mul_x.row(1) =  ag_z*ang_vel_hat_mul_x.row(0)- ag_x*ang_vel_hat_mul_x.row(2);
        ang_vel_hat_sqr_mul_x.row(2) = -ag_y*ang_vel_hat_mul_x.row(0)+ ag_x*ang_vel_hat_mul_x.row(1);

        Eigen::Matrix<double,3,-1> new_coord_3d = _coord_3d
                                        + Eigen::Matrix<double,3,-1>
                                        (   ang_vel_hat_mul_x.array().rowwise() 
                                            * _delta_time.transpose().array()
                                            + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                            * (0.5f * _delta_time.transpose().array().square()) );
        // project and store in a image 
        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.resize(2,_coord_3d.cols());
        new_coord_2d.row(0) = new_coord_3d.row(0).array() / new_coord_3d.row(2).array() * _intrisic(0,0) + _intrisic(0,2);
        new_coord_2d.row(1) = new_coord_3d.row(1).array() / new_coord_3d.row(2).array() * _intrisic(1,1) + _intrisic(1,2);
        
        // TODO gaussian version. 
        cv::Mat image = cv::Mat(180, 240, CV_32FC1);
        image = cv::Scalar(0);

        std::vector<size_t> vec_valid;
        for(size_t i=0; i<_coord_3d.cols(); i++)
        {
            // TODO add gaussian 
            int x = int(new_coord_2d(0,i)), y = int(new_coord_2d(1,i));

            if(x >= 240 || x < 0 || y >= 180 || y < 0)
            {
                // cout <<" overflow x, y" << x <<"," << y << endl;
                continue;
            }

            vec_valid.push_back(i);
            image.at<float>(y,x) += 1;  // TODO should be gaussian
        }

        // calculate residual 
        // residuals[0] = -cv::norm(image);
        // cv::Mat mean, var_mat; 
        // cv::meanStdDev(image, mean, var_mat);

        float event_norm =  1.0/255;
        image *= event_norm;  // normalized. 
        residuals[0] = 5 - cv::norm(image)*cv::norm(image);


        // save imaeg 
        // cv::Mat image_write;
        // cv::normalize(image, image_write, 0,1, cv::NORM_MINMAX, CV_8U);
        // cv::threshold(image_write, image_write, 0.3, 255, cv::THRESH_BINARY);
        // cv::imwrite("/home/hxt/Desktop/hxy-rotation/data/optimize/"+std::to_string(ros::Time::now().toSec())+".png", image_write);
        // cv::imshow("optimizeing ", image);
        // cv::waitKey(100);
        cout << "loss " << residuals[0]  <<", angle "<< ag_x << "," << ag_y << "," << ag_z << endl; 

        // calculate gradient 
        if(!jacobians) return true;
        if(!jacobians[0]) return true;

        cv::Mat blur_image, Ix, Iy; 
        cv::GaussianBlur(image, blur_image, cv::Size(5, 5), 1);
        cv::Sobel(blur_image, Ix, CV_32FC1, 1, 0);
        cv::Sobel(blur_image, Iy, CV_32FC1, 0, 1);

        Eigen::VectorXd Ix_interp, Iy_interp, x_z, y_z, _delta_time_valid;  // the first row of euq(8)
        Eigen::Matrix3Xd eg_jacobian;
        Ix_interp.resize(vec_valid.size());
        Iy_interp.resize(vec_valid.size());
        x_z.resize(vec_valid.size());
        y_z.resize(vec_valid.size());
        _delta_time_valid.resize(vec_valid.size());
        eg_jacobian.resize(3,vec_valid.size());

        for(int i=0; i<vec_valid.size(); i++)
        {
            int x = int(new_coord_2d(0,vec_valid[i])), y = int(new_coord_2d(1,vec_valid[i]));
            _delta_time_valid(i) = _delta_time(vec_valid[i]);
            // conversion from float to double
            Ix_interp(i) = image.at<float>(y,x) * Ix.at<float>(y,x);  
            Iy_interp(i) = image.at<float>(y,x) * Iy.at<float>(y,x);

            x_z(i) = new_coord_3d(0,vec_valid[i]) / new_coord_3d(2,vec_valid[i]);
            y_z(i) = new_coord_3d(1,vec_valid[i]) / new_coord_3d(2,vec_valid[i]);
        }

        Ix_interp *= _intrisic(0,0);
        Iy_interp *= _intrisic(1,1);
        
        eg_jacobian.row(0) = -Ix_interp.array()*x_z.array()*y_z.array() 
                            - Iy_interp.array()*(1+y_z.array()*y_z.array());

        eg_jacobian.row(1) = Ix_interp.array()*(1+x_z.array()*x_z.array()) 
                            + Iy_interp.array()*x_z.array()*y_z.array();
        
        eg_jacobian.row(2) = -Ix_interp.array()*y_z.array() 
                            + Iy_interp.array()*x_z.array();

        // cout << "eg_jacobian " << eg_jacobian.cols() << ","<<eg_jacobian.rows() <<endl;
        // cout << "_delta_time " << _delta_time_valid.cols() << ","<<_delta_time_valid.rows() <<endl;
        
        jacobians[0][0] = eg_jacobian.row(0) * _delta_time_valid;
        jacobians[0][1] = eg_jacobian.row(1) * _delta_time_valid;
        jacobians[0][2] = eg_jacobian.row(2) * _delta_time_valid;
        
        return true;
    }


private:
    // inter midia 
    
    // inputs     
    Eigen::Matrix3d _intrisic; 
    Eigen::Matrix3Xd _coord_3d; 
    Eigen::VectorXd _delta_time;
    int counter; 

};


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
    cout << "warped_event_image " << cv::norm(curr_warpped_event_image) << 
        ", truncated_image " << cv::norm(truncated_image) << 
        ", blur_image " << cv::norm(blur_image) << 
        ", Ix" << cv::norm(Ix) << endl;

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


    cout << "Ix_interp " << Ix_interp.topRows(5).transpose() << endl;
    cout << "warped_image_delta_t " << warped_image_delta_t.topRows(5).transpose() << endl;
    cout << "eg_jacobian " << eg_jacobian.topLeftCorner(1,5).transpose()  <<endl;
    cout << "_delta_time " << _delta_time_valid.topRows(5).transpose()  <<endl;
    cout << "curr_warpped_event_image :\n" << curr_warpped_event_image(cv::Range(100, 110), cv::Range(100, 110)) << endl;
    
    
    Eigen::Vector3d jacobian;

    jacobian(0) = eg_jacobian.row(0) * _delta_time_valid;
    jacobian(1) = eg_jacobian.row(1) * _delta_time_valid;
    jacobian(2) = eg_jacobian.row(2) * _delta_time_valid;

    // jacobian(0) = eg_jacobian.row(0) * warped_image_delta_t;
    // jacobian(1) = eg_jacobian.row(1) * warped_image_delta_t;
    // jacobian(2) = eg_jacobian.row(2) * warped_image_delta_t;

    cout << "gradient " << jacobian.transpose() << endl;
    return jacobian;
}
void System::EstimateMotion()
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
        angular_velocity_compensator = - mu_event / std::sqrt(nu_event) * jacobian;
        // est_angleAxis = SO3add(angular_velocity_compensator,est_angleAxis , true); 
        est_angleAxis = SO3add(est_angleAxis, angular_velocity_compensator, true); 

        // cout << "nu_event " << nu_event <<"," << "rho_event " <<rho_event << endl;


        getWarpedEventImage(est_angleAxis);
        cv::imshow("opti", curr_warpped_event_image);
        cv::waitKey(100);

    }

    cout << "estimated angleAxis " <<  est_angleAxis.transpose() << endl;
    cout << "gt angleAxis        " << gt_angleAxis.transpose() << endl;

} 


/**
* \brief used for ceres to implement CM methods, automatie version. .
*/
struct CMCostFunction
{
    CMCostFunction(const Eigen::Matrix3Xd& coord_3d, const Eigen::VectorXd& delta_time, const Eigen::Matrix3d& K)
        :_coord_3d(coord_3d), _delta_time(delta_time), _intrisic(K)
    {
        cout << "CM loss init :" << endl;
        // cout << "  _coord_3d \n" << _coord_3d.rows() << ", " << _coord_3d.cols() << endl;
        // cout << "  delta_time \n" << delta_time.rows() << ", " << delta_time.cols() << endl;
        // cout << "  _delta_time \n" << _delta_time.rows() << ", " << _delta_time.cols() << endl;
        // cout << "  _intrisic \n" << _intrisic << endl;

    }

    // operator 
    template<typename T> 
    bool operator()(const T* ag, T* residual) const
    {
        Eigen::Matrix<T, 3, -1> ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x, coord_3d_T;
        ang_vel_hat_mul_x.resize(3,_coord_3d.cols());
        ang_vel_hat_sqr_mul_x.resize(3,_coord_3d.cols());
        coord_3d_T.resize(3,_coord_3d.cols());
        
        coord_3d_T = _coord_3d.template cast<T>();
        Eigen::Matrix<T, -1, 1> delta_time_T = _delta_time.template cast<T>();
        // equation 11 
        ang_vel_hat_mul_x.row(0) = -ag[2]*coord_3d_T.row(1) + ag[1]*coord_3d_T.row(2);
        ang_vel_hat_mul_x.row(1) =  ag[2]*coord_3d_T.row(0) - ag[0]*coord_3d_T.row(2);
        ang_vel_hat_mul_x.row(2) = -ag[1]*coord_3d_T.row(0) + ag[0]*coord_3d_T.row(1);

        ang_vel_hat_sqr_mul_x.row(0) = -ag[2]*ang_vel_hat_mul_x.row(1) + ag[1]*ang_vel_hat_mul_x.row(2);
        ang_vel_hat_sqr_mul_x.row(1) =  ag[2]*ang_vel_hat_mul_x.row(0) - ag[0]*ang_vel_hat_mul_x.row(2);
        ang_vel_hat_sqr_mul_x.row(2) = -ag[1]*ang_vel_hat_mul_x.row(0) + ag[0]*ang_vel_hat_mul_x.row(1);


        // cout << "coord_3d_T \n" << coord_3d_T.rows() << ", " << coord_3d_T.cols() << endl;
        // cout << "ang_vel_hat_mul_x \n" << ang_vel_hat_mul_x.rows() << ", " << ang_vel_hat_mul_x.cols() << endl;
        // cout << "ang_vel_hat_sqr_mul_x \n" << ang_vel_hat_sqr_mul_x.rows() << ", " << ang_vel_hat_sqr_mul_x.cols() << endl;
        // cout << "_delta_time \n" << _delta_time.rows() << ", " << _delta_time.cols() << endl;

        Eigen::Matrix<T,3,-1> new_coord_3d = coord_3d_T
                                        + Eigen::Matrix<T,3,-1>
                                        (   ang_vel_hat_mul_x.array().rowwise() 
                                            * delta_time_T.transpose().array()
                                            + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                            * (T(0.5) * delta_time_T.transpose().array().square()) );
        // project and store in a image 
        new_coord_3d.row(0) = new_coord_3d.row(0).array() / new_coord_3d.row(2).array() * T(_intrisic(0,0)) + T(_intrisic(0,2));
        new_coord_3d.row(1) = new_coord_3d.row(1).array() / new_coord_3d.row(2).array() * T(_intrisic(1,1)) + T(_intrisic(1,2));
        

        // TODO gaussian version. 
        Eigen::Matrix<T,-1,-1> image; 
        image.resize(180,240);  // set to 0

        for(int i=0; i<_coord_3d.cols(); i++)
        {
            // x, y index 
            int x, y;
            if constexpr (std::is_same<T, double>::value)
            {
                x = int(new_coord_3d(0,i));
                y = int(new_coord_3d(1,i));
            }
            else
            {
                x = int(new_coord_3d(0,i).a);
                y = int(new_coord_3d(1,i).a);
            }

            if(x >= 240 || x < 0 || y >= 180 || y < 0)
            {
                // cout <<" overflow x, y" << x <<"," << y << endl;
                continue;
            }
            image(y,x) += T(1);  // TODO should be gaussian
        }

        // calculate mean 
        int counterNonZero = 1;
        T mean = T(0);
        for (int i = 0; i < 180; i++)
        {
            for (int j = 0; j < 240; j++)
            {
                if (image(i, j) > T(0.0))
                {
                    mean += image(i, j);
                    counterNonZero++;
                }
            }
        }
        mean /= T(counterNonZero);

        // calculate std 
        T std = T(0);
        for (int i = 0; i < 180; i++)
        {
            for (int j = 0; j < 240; j++)
            {
                if (image(i, j) > T(0.0))
                {
                    std += (image(i, j) - mean)*(image(i, j) - mean);
                }
            }
        }
        std /= T(counterNonZero);

        // residual[0] = T(1e5) - std;  
        residual[0] = new_coord_3d.norm();  

        return true;
    }

    // TODO FIXME reference to CVPR2019 for gaussian smoother. 
    
    // make ceres costfunction 
    static ceres::CostFunction* Create(
        const Eigen::Matrix3Xd& coord_3d, const Eigen::VectorXd& delta_time, const Eigen::Matrix3d& K)
        {
            return new ceres::AutoDiffCostFunction<CMCostFunction,1, 3>(
                new CMCostFunction(coord_3d, delta_time, K));
        }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // inputs     
    Eigen::Matrix3d _intrisic; 
    Eigen::Matrix3Xd _coord_3d; 
    Eigen::VectorXd _delta_time;
    // Eigen::Matrix3Xd ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
};


/**
* \brief using CM with current event bundles.
*/
void System::localCM()
{

    // set problem and cost_function 
    ceres::Problem problem; 
    
    // auto diff version 
    // ceres::CostFunction* cost_function = CMCostFunction::Create(
    //         event_undis_Bundle.coord_3d, eventBundle.time_delta, camera.eg_cameraMatrix);

    // analytic version 
    ceres::CostFunction* cost_function = new CMCostAnalytic( event_undis_Bundle.coord_3d, eventBundle.time_delta, camera.eg_cameraMatrix);

    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); 

    // add residual 
    // Eigen::Vector3d agnleAxis(0,0,0);
    double agnleAxis[3] = {0,0,0};
    problem.AddResidualBlock(cost_function, nullptr, agnleAxis);

    // set opti paramters 
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; 
    options.minimizer_progress_to_stdout = true;
    options.use_nonmonotonic_steps = false;

    // output 
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    cout << summary.BriefReport() << endl;

    // test 
    double residual[1] = {0};
    double** paramter = new double*[1];
    paramter[0] = new double[3] {gt_angleAxis(0), gt_angleAxis(1), gt_angleAxis(2)}; 
    (*cost_function).Evaluate(paramter, residual, nullptr);

    cout << "estimated angle axis " << agnleAxis[0]<<","<<agnleAxis[1]<<","<<agnleAxis[2] << endl; 
    cout << "gt loss " << residual[0] << ", gt angle axis " << gt_angleAxis.transpose() << endl;


    est_angleAxis(0) = agnleAxis[0];
    est_angleAxis(1) = agnleAxis[1];
    est_angleAxis(2) = agnleAxis[2];

    return ;

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
