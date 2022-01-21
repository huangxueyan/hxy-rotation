#include "system.hpp"
#include "numerics.hpp"
#include <sophus/so3.hpp>
#include <algorithm>
#include <ceres/cubic_interpolation.h>
using namespace std;


// CMax version 
void System::EstimateRunTime_CM()
{
    cout << "evaluating run time using CMax" << endl;

    int total_iter_num = yaml_iter_num;

    int right_range = 90, left_range = -20;
    cv::Mat loss_img(right_range-left_range, right_range-left_range, CV_32F, cv::Scalar(0));
    
    double best_var = 0;

    for(int _y = left_range ; _y < right_range; _y++)
    for(int _x = left_range ; _x < right_range; _x++)
    {
        // get timesurface earlier 
        
        double angleAxis[2] = {0.01 * _x, 0.01 * _y}; 
        Eigen::Vector2d eg_trans_vel = Eigen::Vector2d(angleAxis[0],angleAxis[1]);

        Eigen::Matrix2Xd new_coord; new_coord.setZero(2, eventBundle.size); 

        new_coord.row(0) = event_undis_Bundle.coord_3d.row(0).array() + eventBundle.time_delta.transpose().array() * angleAxis[0];
        new_coord.row(1) = event_undis_Bundle.coord_3d.row(1).array() + eventBundle.time_delta.transpose().array() * angleAxis[1];

        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.setZero(2, eventBundle.size);
        new_coord_2d.row(0) = new_coord.row(0).array() * camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
        new_coord_2d.row(1) = new_coord.row(1).array() * camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);

        Eigen::Matrix<double,180,-1> iwe; iwe.setZero(180,240);

        for(int i=0; i<eventBundle.size; i++)
        {
            int x_int = new_coord_2d(0, i), y_int = new_coord_2d(1, i); 
            double dx = new_coord_2d(0, i) - x_int, dy = new_coord_2d(1, i) - y_int;
            
            if(x_int<1 || x_int>=239 || y_int<1 || y_int>=179) continue;
            
            iwe(y_int, x_int)     += (1.0-dx) * (1.0-dy);
            iwe(y_int+1, x_int)   += (1.0-dx) * dy;
            iwe(y_int, x_int+1)   += dx * (1.0-dy);
            iwe(y_int+1, x_int+1) += dx * dy;
        }

        // cout << "iwe max before mean" << iwe.maxCoeff() << endl;


        // convert eigen2cv 
        cv::Mat cv_iwe;
        cv::eigen2cv(iwe, cv_iwe);

        // gaussian blur 
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::Mat cv_iwe_blur;
        cv::GaussianBlur(cv_iwe, cv_iwe_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        // convert cv2eigen
        cv::cv2eigen(cv_iwe_blur, iwe); 

        double mean = iwe.mean(); 
        iwe.array() -= mean; 

        // double var = iwe.array().pow(2).sum() / iwe.size();
        double var = iwe.array().pow(2).sum();
        // cout << "iwe_pos.sum()" << iwe_pos.sum() << " map_pos_sum " << map_pos_sum << endl;
        if(_x % 3 == 0 && _y % 3 == 0)
        {
            cout << "using " << eg_trans_vel.transpose() <<endl;

            cout << "iter_ " << _x <<"," <<_y << " var " << var << endl;
            // visualize 
            getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
            cv::imshow("cmax warp ", cv_iwe_blur);
            visualize(); 
            cout << "----------------------" << endl;
        }

        if(var > best_var && std::abs(_y)<3)
        {
            best_var = var;
            getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);
            cv::normalize(curr_warpped_event_image, curr_warpped_event_image, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
            cv::threshold(curr_warpped_event_image, curr_warpped_event_image, 0.1, 255, cv::THRESH_BINARY);
            cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/iwe_CM_" 
                +std::to_string(best_var) + +"_("+std::to_string(_x) + "," + std::to_string(_y) + ").png", curr_warpped_event_image);
        }

        loss_img.at<float>(_y-left_range, _x - left_range) = var; 
    }

    cv::namedWindow("loss", cv::WINDOW_NORMAL); 
        cv::Mat image_color;
        cv::normalize(loss_img, image_color, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
        cv::applyColorMap(image_color, image_color, cv::COLORMAP_JET);
    cv::imshow("loss", image_color);
    cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/loss_CM.png", image_color);
    cv::waitKey(0);

}

// STPPP version 
void System::EstimateRunTime_PPP()
{
    cout << "evaluating run time using PPP" << endl;

    int total_iter_num = yaml_iter_num;

    int right_range = 90, left_range = -20;
    cv::Mat loss_img(right_range-left_range, right_range-left_range, CV_32F, cv::Scalar(0));
    
    double best_var = -100;

    for(int _x = left_range ; _x < right_range; _x++)
    for(int _y = left_range ; _y < right_range; _y++)
    {
        // get timesurface earlier 
        double angleAxis[2] = {0.01 * _x, 0.01 * _y}; 
        Eigen::Vector2d eg_trans_vel(angleAxis[0],angleAxis[1]);
        
        double residual = 0;


        Eigen::Matrix2Xd new_coord; new_coord.setZero(2, eventBundle.size);
        new_coord.row(0) = event_undis_Bundle.coord_3d.row(0).array() + eventBundle.time_delta.transpose().array() * angleAxis[0];
        new_coord.row(1) = event_undis_Bundle.coord_3d.row(1).array() + eventBundle.time_delta.transpose().array() * angleAxis[1];

        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.setZero(2, eventBundle.size);
        new_coord_2d.row(0) = new_coord.row(0).array() * camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
        new_coord_2d.row(1) = new_coord.row(1).array() * camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);
        
        Eigen::Matrix<double,180,-1> iwe_pos; iwe_pos.setZero(180,240);
        Eigen::Matrix<double,180,-1> iwe_neg; iwe_neg.setZero(180,240);

        for(int i=0; i<eventBundle.size; i++)
        {
            int x_int = new_coord_2d(0, i), y_int = new_coord_2d(1, i); 
            double dx = new_coord_2d(0, i) - x_int, dy = new_coord_2d(1, i) - y_int;
            
            if(x_int<0 || x_int>=240 || y_int<0 || y_int>=180) continue;
            
            if(eventBundle.polar(i) == 1)
            {
                iwe_pos(y_int, x_int)     += (1.0-dx) * (1.0-dy);
                iwe_pos(y_int+1, x_int)   += (1.0-dx) * dy;
                iwe_pos(y_int, x_int+1)   += dx * (1.0-dy);
                iwe_pos(y_int+1, x_int+1) += dx * dy;
            }
            else
            {
                iwe_neg(y_int, x_int)     += (1.0-dx) * (1.0-dy);
                iwe_neg(y_int+1, x_int)   += (1.0-dx) * dy;
                iwe_neg(y_int, x_int+1)   += dx * (1.0-dy);
                iwe_neg(y_int+1, x_int+1) += dx * dy;               
            }
        }

        // convert eigen2cv 
        cv::Mat cv_iwe_pos, cv_iwe_neg;
        cv::eigen2cv(iwe_pos, cv_iwe_pos);
        cv::eigen2cv(iwe_neg, cv_iwe_neg);

        // cout << "cv_iwe_neg " << cv_iwe_pos.size() << ",iwe " << iwe_pos.rows() << ", " <<  iwe_pos.cols() << endl;

        // gaussian blur 
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::Mat cv_iwe_pos_blur, cv_iwe_neg_blur;
        cv::GaussianBlur(cv_iwe_pos, cv_iwe_pos_blur, cv::Size(gaussian_size, gaussian_size), sigma);
        cv::GaussianBlur(cv_iwe_neg, cv_iwe_neg_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        // convert cv2eigen
        cv::cv2eigen(cv_iwe_pos_blur, iwe_pos); 
        cv::cv2eigen(cv_iwe_neg_blur, iwe_neg); 

        // cout << "cv_iwe_neg_blur " << cv_iwe_neg_blur.size() << ",iwe " << iwe_pos.rows() << ", " <<  iwe_pos.cols() << endl;

        double map_pos_sum = 0, map_neg_sum = 0;
        double r = 0.1, beta = 1.59;
        for(int i=0; i<180; i++)
        for(int j=0; j<240; j++)
        {
            map_neg_sum += lgamma(iwe_neg(i, j) + r) + 
                            r * log(beta) -
                            lgamma(iwe_neg(i, j) + 1.0) - 
                            lgamma(r) - 
                            (iwe_neg(i, j) + r) * log(beta + 1);

            map_pos_sum += lgamma(iwe_pos(i, j) + r) + 
                            r * log(beta) -
                            lgamma(iwe_pos(i, j) + 1.0) - 
                            lgamma(r) - 
                            (iwe_pos(i, j) + r) * log(beta + 1);
        }

        double loss1 = map_pos_sum / iwe_pos.sum(); 
        double loss2 = map_neg_sum / iwe_neg.sum(); 

        if(_x % 3 == 0 && _y % 3 == 0)
        {
            cout << "using " << eg_trans_vel.transpose() <<endl;
            cout << "iter_ " << _x <<"," <<_y << " loss " << loss1 + loss2 << endl;
            // visualize 
            
            getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
            cv::imshow("cv_iwe_neg_blur ", cv_iwe_neg_blur);
            cv::imshow("cv_iwe_pos_blur ", cv_iwe_pos_blur);
            visualize(); 
        }


        if(loss1 + loss2 > best_var && std::abs(_y)<3)
        {
            best_var = loss1 + loss2;
            getWarpedEventImage(eg_trans_vel, event_warpped_Bundle).convertTo(curr_warpped_event_image, CV_32FC3);  // get latest warpped events 
            cv::normalize(curr_warpped_event_image, curr_warpped_event_image, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
            cv::threshold(curr_warpped_event_image, curr_warpped_event_image, 0.1, 255, cv::THRESH_BINARY);
            cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/iwe_PPP_" 
                +std::to_string(std::abs(best_var)) + +"_("+std::to_string(_x) + "," + std::to_string(_y) + ").png", curr_warpped_event_image);
        }

        loss_img.at<float>(_y-left_range, _x - left_range) = loss1 + loss2; 
    }

    cv::namedWindow("loss", cv::WINDOW_NORMAL); 
    cv::Mat image_color;
    cv::normalize(loss_img, image_color, 255, 0, cv::NORM_MINMAX, CV_8UC1);  
    cv::applyColorMap(image_color, image_color, cv::COLORMAP_JET);
    cv::imshow("loss", image_color);
    cv::imwrite("/home/hxy/Desktop/hxy-rotation/data/translation_estimation/loss_PPP.png", image_color);
    cv::waitKey(0);

}
