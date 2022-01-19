#include "system.hpp"



struct NumericDiffCostFunctor
{

    NumericDiffCostFunctor(const Eigen::Matrix3Xd& coord_3d, const Eigen::VectorXd& delta_time, const Eigen::Matrix3d& K)
    :_coord_3d(coord_3d), _delta_time(delta_time), _intrisic(K)
    {
        cout << "CM loss init :" << endl;
    }

    bool operator()(const double* const parameters, double* residuals) const 
    {

        Eigen::Matrix<double, 3, -1> ang_vel_hat_mul_x, ang_vel_hat_sqr_mul_x;
        ang_vel_hat_mul_x.resize(3,_coord_3d.cols());
        ang_vel_hat_sqr_mul_x.resize(3,_coord_3d.cols());
        
        const double ag_x = parameters[0], ag_y = parameters[1], ag_z = parameters[2];

        cout << "size " << _coord_3d.cols() << ", ag_x "<< ag_x << ", " << ag_y << ", "<<ag_z << endl;

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
                                            * (-_delta_time.transpose().array())
                                            + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                            * (0.5f * _delta_time.transpose().array().square()) );
        // project and store in a image 
        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.resize(2,_coord_3d.cols());
        new_coord_2d.row(0) = new_coord_3d.row(0).array() / new_coord_3d.row(2).array() * _intrisic(0,0) + _intrisic(0,2);
        new_coord_2d.row(1) = new_coord_3d.row(1).array() / new_coord_3d.row(2).array() * _intrisic(1,1) + _intrisic(1,2);
        
        cout << "new coord " << new_coord_2d.topLeftCorner(2,5) << endl;
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

        // float event_norm =  1.0/255;
        // image *= event_norm;  // normalized. 
        residuals[0] = 50000 - cv::norm(image)*cv::norm(image);
    }


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // inputs     
    Eigen::Matrix3d _intrisic; 
    Eigen::Matrix3Xd _coord_3d; 
    Eigen::VectorXd _delta_time;
};



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

        cout << "size " << _coord_3d.cols() << ", ag_x "<< ag_x << ", " << ag_y << ", "<<ag_z << endl;

        // equation 11 
        ang_vel_hat_mul_x.row(0) = -ag_z*_coord_3d.row(1) + ag_y*_coord_3d.row(2);
        ang_vel_hat_mul_x.row(1) =  ag_z*_coord_3d.row(0) - ag_x*_coord_3d.row(2);
        ang_vel_hat_mul_x.row(2) = -ag_y*_coord_3d.row(0) + ag_x*_coord_3d.row(1);

        ang_vel_hat_sqr_mul_x.row(0) = -ag_z*ang_vel_hat_mul_x.row(1)+ ag_y*ang_vel_hat_mul_x.row(2);
        ang_vel_hat_sqr_mul_x.row(1) =  ag_z*ang_vel_hat_mul_x.row(0)- ag_x*ang_vel_hat_mul_x.row(2);
        ang_vel_hat_sqr_mul_x.row(2) = -ag_y*ang_vel_hat_mul_x.row(0)+ ag_x*ang_vel_hat_mul_x.row(1);

        Eigen::Matrix<double,3,-1> new_coord_3d = _coord_3d + 
                                           (ang_vel_hat_mul_x.array().rowwise() 
                                            * (-_delta_time.transpose().array())).matrix();
                                            // + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                            // * (0.5f * _delta_time.transpose().array().square()) ;
        // project and store in a image 
        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.resize(2,_coord_3d.cols());
        new_coord_2d.row(0) = new_coord_3d.row(0).array() / new_coord_3d.row(2).array() * _intrisic(0,0) + _intrisic(0,2);
        new_coord_2d.row(1) = new_coord_3d.row(1).array() / new_coord_3d.row(2).array() * _intrisic(1,1) + _intrisic(1,2);
        
        cout << "new coord " << new_coord_2d.topLeftCorner(2,5) << endl;
        // TODO gaussian version. 
        cv::Mat image = cv::Mat(180, 240, CV_32FC1);
        image = cv::Scalar(0);

        std::vector<size_t> vec_valid;
        for(size_t i=0; i<_coord_3d.cols(); i++)
        {
            // TODO add gaussian 
            int x = int(new_coord_2d(0,i)), y = int(new_coord_2d(1,i));

            if(x >= 239 || x < 1 || y >= 179 || y < 1)
            {
                // cout <<" overflow x, y" << x <<"," << y << endl;
                continue;
            }

            vec_valid.push_back(i);
            image.at<float>(y,  x)   += (1 - (new_coord_2d(0,i) - x)) * (1-(new_coord_2d(1,i) - y));  
            image.at<float>(y+1,x)   += (1 - (new_coord_2d(0,i) - x)) * (new_coord_2d(1,i) - y); 
            image.at<float>(y,  x+1) += (new_coord_2d(0,i) - x) * (1-(new_coord_2d(1,i) - y)); 
            image.at<float>(y+1,x+1) += (new_coord_2d(0,i) - x) * (new_coord_2d(1,i) - y);  
        }

        // calculate residual 
        // residuals[0] = -cv::norm(image);
        // cv::Mat mean, var_mat; 
        // cv::meanStdDev(image, mean, var_mat);

        // float event_norm =  1.0/255;
        // image *= event_norm;  // normalized. 
        cv::Mat blur_image;
        cv::GaussianBlur(image, blur_image, cv::Size(5, 5), 1);

        residuals[0] = cv::norm(image)*cv::norm(image);
        // residuals[0] = 50000 - cv::norm(image)*cv::norm(image);
        cout << "loss " << residuals[0]  <<", angle "<< ag_x << "," << ag_y << "," << ag_z << endl; 

        // save imaeg 
        cv::Mat image_write;
        cv::normalize(image, image_write, 0,255, cv::NORM_MINMAX, CV_8U);
        // cv::threshold(image_write, image_write, 0.3, 255, cv::THRESH_BINARY);
        // std::string time = std::to_string(ros::Time::now().toSec());
        // cv::imwrite("/home/hxt/Desktop/hxy-rotation/data/optimize/"+time+".png", image_write);
        cv::imshow("optimizeing ", image_write);
        cv::waitKey(100);


        // calculate gradient 
        if (jacobians == NULL || jacobians[0] == NULL)
            return true;

        // cv::Mat truncated_image, blur_image, Ix, Iy; 
        // int threshold = 10; 
        // cv::threshold(image, truncated_image, threshold, 255, cv::THRESH_TRUNC); 
        // cv::GaussianBlur(truncated_image, blur_image, cv::Size(5, 5), 1);
        cv::Mat Ix, Iy; 
        cv::Sobel(blur_image, Ix, CV_32FC1, 1, 0);
        cv::Sobel(blur_image, Iy, CV_32FC1, 0, 1);

        // cout << "warped_event_image " << cv::norm(image) << 
        //     ", truncated_event " << cv::norm(truncated_image) << 
        //     ", blur_image " << cv::norm(blur_image) << 
        //     ", Ix" << cv::norm(Ix) << endl;

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
        

        cout << "jacobian " << jacobians[0][0]<<"," <<jacobians[0][1] <<","<< jacobians[0][2] << endl; 

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
    ceres::CostFunction* cost_function = new CMCostAnalytic(event_undis_Bundle.coord_3d, eventBundle.time_delta, camera.eg_cameraMatrix);

    // ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,1, 3>   
    //                     (new NumericDiffCostFunctor(event_undis_Bundle.coord_3d, eventBundle.time_delta, camera.eg_cameraMatrix)) ;

    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); 

    // add residual 
    // Eigen::Vector3d agnleAxis(0,0,0);
    double temp[3]= {0,0,0};
    double* agnleAxis = &temp[0];
    problem.AddResidualBlock(cost_function, nullptr, agnleAxis);

    // set opti paramters 
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; 
    options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = false;

    // output 
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    cout << summary.FullReport() << endl;

    // test 
    // double residual[1] = {0};
    // double** paramter = new double*[1];
    // paramter[0] = new double[3] {gt_angleAxis(0), gt_angleAxis(1), gt_angleAxis(2)}; 
    // (*cost_function).Evaluate(paramter, residual, nullptr);

    // cout << "estimated angle axis " << agnleAxis[0]<<","<<agnleAxis[1]<<","<<agnleAxis[2] << endl; 
    // cout << "gt loss " << residual[0] << ", gt angle axis " << gt_angleAxis.transpose() << endl;


    est_angleAxis(0) = agnleAxis[0];
    est_angleAxis(1) = agnleAxis[1];
    est_angleAxis(2) = agnleAxis[2];

    return ;

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
                                            * (-delta_time_T.transpose().array())
                                            + ang_vel_hat_sqr_mul_x.array().rowwise() 
                                            * (T(0.5) * delta_time_T.transpose().array().square()) );
        // project and store in a image 
        new_coord_3d.row(0) = new_coord_3d.row(0).array() / new_coord_3d.row(2).array() * T(_intrisic(0,0)) + T(_intrisic(0,2));
        new_coord_3d.row(1) = new_coord_3d.row(1).array() / new_coord_3d.row(2).array() * T(_intrisic(1,1)) + T(_intrisic(1,2));
        

        // TODO gaussian version. 
        Eigen::Matrix<T,-1,-1> image; 
        image.setZero(180, 240);  // set to 0

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
        mean = mean / T(counterNonZero);

        cout << "mean " << mean << endl;

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
        std = std / T(counterNonZero);

        // T event_norm =  T(1.0/255);
        // image *= event_norm;  // normalized. 

        residual[0] = T(5) - std;
        cout << "std" << residual[0] << endl;
        // residual[0] = T(1e5) - std;  
        // residual[0] = new_coord_3d.norm();  

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



