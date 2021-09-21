
#include "numerics.hpp"



/**
* \brief convert x[1x3] to asym matrix [3x3].
* \param[in] x .
*/
Eigen::Matrix3d hat(const Eigen::Vector3d &x)
{
    Eigen::Matrix3d x_hat; 
    x_hat << 0, -x(2), x(1), 
        x(2), 0, -x(0),
        x(1), x(0), 0;
    return x_hat; 
}


/**
* \brief from asy matrix [3x3] to vector [1x3].
*/
Eigen::Vector3d unhat(const Eigen::Matrix3d &x_hat)
{
    Eigen::Vector3d x;
    x << x_hat(2,1), x_hat(0,2), x(1,0);
    return x;
}

/**
* \brief from 3 AngleAxis theta to rotation matrix.
* \param rpy rotation angle in rad format. .
*/
Eigen::Matrix3d SO3(const Eigen::Vector3d &x)
{
    return Eigen::MatrixX3d(hat(x).exp());
}

/**
* \brief Constructor.
* \param x_hat. 
*/
Eigen::Vector3d InvSO3(const Eigen::Matrix3d &R)
{
    return unhat(R.log());
}   

/**
* \brief from 3 theta to rotation matrix.
* \param rpy rotation angle in rad format. .
*/
Eigen::Vector3d SO3add(const Eigen::Vector3d &x1, const Eigen::Vector3d &x2, bool circle)
{
    if (circle && (x1 + x2).norm() > M_PI)
    {
        return x1 + x2;
    }
    else
    {
        return InvSO3(SO3(x1) * SO3(x2));
    }
}


/**
* \brief from quaternion to euler anglers， return rpy(xyz) theta.
*/
Eigen::Vector3d toEulerAngles(Eigen::Quaterniond q){
    MyEulerAngles angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    Eigen::Vector3d v3d(angles.roll, angles.pitch, angles.yaw);
    return v3d;
}


Eigen::Quaterniond ToQuaternion(double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    double w = cr * cp * cy + sr * sp * sy;
    double x = sr * cp * cy - cr * sp * sy;
    double y = cr * sp * cy + sr * cp * sy;
    double z = cr * cp * sy - sr * sp * cy;

    
    Eigen::Quaterniond q(w,x,y,z);
    // std::cout <<"Quaterniond: "<< q.coeffs().transpose() << ", norm: "<<  q.coeffs().norm() << std::endl;
    return q;
}