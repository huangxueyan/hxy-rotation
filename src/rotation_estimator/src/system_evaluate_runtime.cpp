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

    double angleAxis[3] = {0.01, 0.01, 0.01}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        
        double residual = 0;

        Eigen::Matrix3Xd points_raw = event_undis_Bundle.coord_3d;

        Eigen::Matrix<double, 3, -1> delta_rot; delta_rot.resize(3, eventBundle.size);

        delta_rot.row(0) = -eg_angleAxis(2)*points_raw.row(1) + eg_angleAxis(1)*points_raw.row(2);
        delta_rot.row(1) =  eg_angleAxis(2)*points_raw.row(0) - eg_angleAxis(0)*points_raw.row(2);
        delta_rot.row(2) = -eg_angleAxis(1)*points_raw.row(0) + eg_angleAxis(0)*points_raw.row(1);

        Eigen::Matrix3Xd new_coord_3d = points_raw.array() + delta_rot.array().rowwise() * eventBundle.time_delta.transpose().array();

        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.resize(2, eventBundle.size);
        new_coord_2d.row(0) = new_coord_3d.row(0).array()/new_coord_3d.row(2).array()*camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
        new_coord_2d.row(1) = new_coord_3d.row(1).array()/new_coord_3d.row(2).array()*camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);
        

        Eigen::Matrix<double,180,-1> iwe; iwe.resize(180,240);

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
        iwe = (iwe.array() - mean).matrix(); 

        // cout << "mean " << mean << endl;
        // cout << "iwe max after mean" << iwe.maxCoeff() << endl;

        Eigen::Map<Eigen::Matrix<double, 1, -1>> iwe_vec(iwe.data(), 1, 180*240);
        double var = iwe_vec * iwe_vec.transpose(); 

        // cout << "iwe_pos.sum()" << iwe_pos.sum() << " map_pos_sum " << map_pos_sum << endl;
        // cout << "iter " << iter_ << " var " << var / (180*240) << endl;
    }

}

// STPPP version 
void System::EstimateRunTime_PPP()
{
    cout << "evaluating run time using PPP" << endl;

    int total_iter_num = yaml_iter_num;

    double angleAxis[3] = {0.01, 0.01, 0.01}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
        
        double residual = 0;

        Eigen::Matrix3Xd points_raw = event_undis_Bundle.coord_3d;

        Eigen::Matrix<double, 3, -1> delta_rot; delta_rot.resize(3, eventBundle.size);

        delta_rot.row(0) = -eg_angleAxis(2)*points_raw.row(1) + eg_angleAxis(1)*points_raw.row(2);
        delta_rot.row(1) =  eg_angleAxis(2)*points_raw.row(0) - eg_angleAxis(0)*points_raw.row(2);
        delta_rot.row(2) = -eg_angleAxis(1)*points_raw.row(0) + eg_angleAxis(0)*points_raw.row(1);

        Eigen::Matrix3Xd new_coord_3d = points_raw.array() + delta_rot.array().rowwise() * eventBundle.time_delta.transpose().array();

        Eigen::Matrix<double,2,-1> new_coord_2d; new_coord_2d.resize(2, eventBundle.size);
        new_coord_2d.row(0) = new_coord_3d.row(0).array()/new_coord_3d.row(2).array()*camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
        new_coord_2d.row(1) = new_coord_3d.row(1).array()/new_coord_3d.row(2).array()*camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);
        
        Eigen::Matrix<double,180,-1> iwe_pos; iwe_pos.resize(180,240);
        Eigen::Matrix<double,180,-1> iwe_neg; iwe_neg.resize(180,240);

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

        // cout << "iwe_pos.sum()" << iwe_pos.sum() << " map_pos_sum " << map_pos_sum << endl;
        // cout << "iter " << iter_ << " loss1 + loss2 " << loss1 + loss2 << endl;
    }
}

// single warp version 
void System::EstimateRunTime_Single()
{
    cout << "evaluating run time using Single" << endl;

    double ts_start = yaml_ts_start, ts_end = yaml_ts_end;
    int sample_num = yaml_sample_count, total_iter_num = yaml_iter_num;

    double angleAxis[3] = {0.01, 0.01, 0.01}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
         
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 

        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);

        // cout << "default value is " << early_default_value << endl;


        // get warpped events ref t0 and its timesurface 
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle);  
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 
            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 

        // add gaussian on cv_earlier_timesurface
        cv::Mat cv_earlier_timesurface_blur;
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        vector<float> line_grid_early; line_grid_early.assign((float*)cv_earlier_timesurface_blur.data, (float*)cv_earlier_timesurface_blur.data + 180*240);

        ceres::Grid2D<float,1> grid_early(line_grid_early.data(), 0, 180, 0, 240);
        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_early);


        double residual = 0;

        for(int opti_loop = 0; opti_loop < yaml_ceres_iter_num; opti_loop++)
        {

            int samples_count = std::min(sample_num, int(eventBundle.size)); 
            std::vector<int> vec_sampled_idx; 
            for(int _k=1; _k<=samples_count; _k++)
            {
                vec_sampled_idx.push_back(eventBundle.size - _k);
            }
            // getSampledVec(vec_sampled_idx, samples_count, 0, 1);

            Eigen::Matrix3Xd points_raw = event_undis_Bundle.coord_3d(Eigen::all, vec_sampled_idx);
            Eigen::VectorXd delta_time_early =eventBundle.time_delta(vec_sampled_idx);

            Eigen::Matrix<double, 3, -1> delta_rot; delta_rot.resize(3, samples_count);

            delta_rot.row(0) = -eg_angleAxis(2)*points_raw.row(1) + eg_angleAxis(1)*points_raw.row(2);
            delta_rot.row(1) =  eg_angleAxis(2)*points_raw.row(0) - eg_angleAxis(0)*points_raw.row(2);
            delta_rot.row(2) = -eg_angleAxis(1)*points_raw.row(0) + eg_angleAxis(0)*points_raw.row(1);

            Eigen::Matrix3Xd new_coord_3d_early = points_raw.array() + delta_rot.array().rowwise() * delta_time_early.transpose().array();

            Eigen::Matrix<double,2,-1> new_coord_2d_early; new_coord_2d_early.resize(2, samples_count);
            new_coord_2d_early.row(0) = new_coord_3d_early.row(0).array()/new_coord_3d_early.row(2).array()*camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
            new_coord_2d_early.row(1) = new_coord_3d_early.row(1).array()/new_coord_3d_early.row(2).array()*camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);
            

            for(int i=0; i<samples_count; i++)
            {
                double early_loss = 0;
                interpolator_early_ptr->Evaluate(new_coord_2d_early(i,1), new_coord_2d_early(i,0), &early_loss);
                residual += early_loss; 

                if(new_coord_2d_early(i,1) < 0 || new_coord_2d_early(i,1) > 180 || 
                   new_coord_2d_early(i,0) < 0 || new_coord_2d_early(i,0) > 240  )
                   {
                    //    cout << "points " << new_coord_2d_early(i, 0) << ", " << new_coord_2d_early(i, 1) << ", value " << early_loss << endl;
                   }
            }
        }

        // cout << "iter " << iter_ << " residual " << residual << endl;
    }

}

// double warp
void System::EstimateRunTime_Double()
{
    cout << "evaluating run time using Double" << endl;

    double ts_start = yaml_ts_start, ts_end = yaml_ts_end;
    int sample_num = yaml_sample_count, total_iter_num = yaml_iter_num;

    double angleAxis[3] = {0.01, 0.01, 0.01}; 
    
    for(int iter_= 1; iter_<= total_iter_num; iter_++)
    {
        // get timesurface earlier 
        Eigen::Vector3d eg_angleAxis(angleAxis[0],angleAxis[1],angleAxis[2]);
         
        double timesurface_range = ts_start + iter_/float(total_iter_num) * (ts_end-ts_start);  
        cv::Mat cv_earlier_timesurface = cv::Mat(180,240, CV_32FC1); 
        cv::Mat cv_later_timesurface = cv::Mat(180,240, CV_32FC1); 


        // cv::Mat visited_map = cv::Mat(180,240, CV_8U); visited_map.setTo(0);
        float early_default_value = eventBundle.time_delta(int(eventBundle.size*timesurface_range));
        cv_earlier_timesurface.setTo(early_default_value * yaml_default_value_factor);
        float later_default_value = eventBundle.time_delta(eventBundle.size-1) - eventBundle.time_delta(int(eventBundle.size*(1-timesurface_range)));
        cv_later_timesurface.setTo(later_default_value * yaml_default_value_factor);

        // get warpped events ref t0 and its timesurface 
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle);  
        for(int i= event_warpped_Bundle.size*timesurface_range; i >=0; i--)
        {
            
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 
            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
                cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(i);  
        } 

        // get warpped events ref t1 and its timesurface 
        getWarpedEventImage(eg_angleAxis, event_warpped_Bundle, PlotOption::U16C1_EVNET_IMAGE, true);
        for(int i= event_warpped_Bundle.size*(1-timesurface_range); i< event_warpped_Bundle.size; i++)
        {
            int sampled_x = std::round(event_warpped_Bundle.coord.col(i)[0]), sampled_y = std::round(event_warpped_Bundle.coord.col(i)[1]); 
            if(event_warpped_Bundle.isInner[i] < 1) continue;               // outlier 
                cv_later_timesurface.at<float>(sampled_y, sampled_x) = eventBundle.time_delta(event_warpped_Bundle.size-1) - eventBundle.time_delta(i);  
        } 

        // add gaussian on cv_earlier_timesurface
        cv::Mat cv_earlier_timesurface_blur, cv_later_timesurface_blur;
        int gaussian_size = yaml_gaussian_size;
        float sigma = yaml_gaussian_size_sigma;
        cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);
        cv::GaussianBlur(cv_later_timesurface, cv_later_timesurface_blur, cv::Size(gaussian_size, gaussian_size), sigma);

        vector<float> line_grid_early; line_grid_early.assign((float*)cv_earlier_timesurface_blur.data, (float*)cv_earlier_timesurface_blur.data + 180*240);
        vector<float> line_grid_later; line_grid_later.assign((float*)cv_later_timesurface_blur.data, (float*)cv_later_timesurface_blur.data + 180*240);

        ceres::Grid2D<float,1> grid_early(line_grid_early.data(), 0, 180, 0, 240);
        ceres::Grid2D<float,1> grid_later(line_grid_later.data(), 0, 180, 0, 240);
        auto* interpolator_early_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_early);
        auto* interpolator_later_ptr = new ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>(grid_later);  


        double residual = 0;

        for(int opti_loop = 0; opti_loop < yaml_ceres_iter_num; opti_loop++)
        {
            int samples_count = std::min(sample_num, int(eventBundle.size)); 
            std::vector<int> vec_sampled_idx; 
            for(int _k=1; _k<=samples_count; _k++)
            {
                vec_sampled_idx.push_back(eventBundle.size - _k);
            }
            // getSampledVec(vec_sampled_idx, samples_count, 0, 1);

            Eigen::Matrix3Xd points_raw = event_undis_Bundle.coord_3d(Eigen::all, vec_sampled_idx);
            Eigen::VectorXd delta_time_early =eventBundle.time_delta(vec_sampled_idx);
            Eigen::VectorXd delta_time_later =eventBundle.time_delta(vec_sampled_idx).array() - eventBundle.time_delta(event_warpped_Bundle.size-1);

            Eigen::Matrix<double, 3, -1> delta_rot; delta_rot.resize(3, samples_count);

            delta_rot.row(0) = -eg_angleAxis(2)*points_raw.row(1) + eg_angleAxis(1)*points_raw.row(2);
            delta_rot.row(1) =  eg_angleAxis(2)*points_raw.row(0) - eg_angleAxis(0)*points_raw.row(2);
            delta_rot.row(2) = -eg_angleAxis(1)*points_raw.row(0) + eg_angleAxis(0)*points_raw.row(1);

            Eigen::Matrix3Xd new_coord_3d_early = points_raw.array() + delta_rot.array().rowwise() * delta_time_early.transpose().array();
            Eigen::Matrix3Xd new_coord_3d_later = points_raw.array() + delta_rot.array().rowwise() * delta_time_later.transpose().array();

            Eigen::Matrix<double,2,-1> new_coord_2d_early; new_coord_2d_early.resize(2, samples_count);
            Eigen::Matrix<double,2,-1> new_coord_2d_later; new_coord_2d_later.resize(2, samples_count);
            new_coord_2d_early.row(0) = new_coord_3d_early.row(0).array()/new_coord_3d_early.row(2).array()*camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
            new_coord_2d_early.row(1) = new_coord_3d_early.row(1).array()/new_coord_3d_early.row(2).array()*camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);
            new_coord_2d_later.row(0) = new_coord_3d_later.row(0).array()/new_coord_3d_later.row(2).array()*camera.eg_cameraMatrix(0,0) + camera.eg_cameraMatrix(0,2);
            new_coord_2d_later.row(1) = new_coord_3d_later.row(1).array()/new_coord_3d_later.row(2).array()*camera.eg_cameraMatrix(1,1) + camera.eg_cameraMatrix(1,2);


            for(int i=0; i<samples_count; i++)
            {
                double early_loss, later_loss = 0;
                interpolator_early_ptr->Evaluate(new_coord_2d_early(i,1), new_coord_2d_early(i, 0), &early_loss);
                interpolator_later_ptr->Evaluate(new_coord_2d_later(i,1), new_coord_2d_later(i, 0), &later_loss);
                residual += early_loss + later_loss; 
            }
        }

        // cout << "iter " << iter_ << " residual " << residual << endl;
    }

}