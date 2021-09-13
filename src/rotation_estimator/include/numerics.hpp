#pragma once 

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <math.h>

#include <sophus/so3.hpp>

/**
* \brief from 3 theta to 3x3 matrix asymmetrix matrix.
* \param x rpy rotation angle in rad format.
*/
Eigen::Matrix3f hat(const Eigen::Vector3f &x); 

Eigen::Matrix3f unhat(const Eigen::Matrix3f &x); 


/**
* \brief from 3 theta to rotation matrix.
* \param rpy rotation angle in rad format. .
*/
Eigen::Matrix3f SO3(const Eigen::Vector3f &x); 


/**
* \brief from 3 theta to rotation matrix.
* \param rpy rotation angle in rad format. .
*/
Eigen::Matrix3f SO3add(const Eigen::Vector3f &x, const Eigen::Vector3f &y); 




struct MyEulerAngles {
    double roll, pitch, yaw;
};


/**
* \brief return rpy(xyz) theta.
*/
Eigen::Vector3d toEulerAngles(Eigen::Quaterniond q);