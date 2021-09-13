#include "database.hpp"


using namespace std; 


CameraPara::CameraPara()
{
    width = 240; 
    height = 180; 

    fx = 230.2097;
    fy = 231.1228;
    cx = 121.6862;
    cy = 86.8208;
    rd1 = -0.4136;
    rd2 = 0.2042;

    cameraMatrix = (cv::Mat_<float>(3,3) << 
                    fx, 0, cx , 0, fy, cy, 0, 0, 1); 
    distCoeffs = (cv::Mat_<double>(1,4) << rd1, rd2, 0 ,0 ); 

    cout << "camera param: \n" << endl;
    cout << cameraMatrix << endl; 

    cv::cv2eigen(cameraMatrix, eg_cameraMatrix);
    // cout << eg_cameraMatrix << endl; 

    // the map展示的虚拟相机k内参可以自定义大小。 
    int map_scale = 2; // visualize 
    eg_MapMatrix = eg_cameraMatrix * 1;
    
    eg_MapMatrix(0,2) = map_scale * width / 2;
    eg_MapMatrix(1,2) = map_scale * height / 2; 
    
}

void EventBundle::Clear()
{
    size = 0; 
    // eigen 
    coord.conservativeResize(2,size);
    coord_3d.conservativeResize(3,size);
    time_delta.conservativeResize(size);
    time_delta_rev.conservativeResize(size);
    
    angular_position = Eigen::Vector3f::Zero();
    angular_velocity = Eigen::Vector3f::Zero();
    
    // vector 
    // time_stamps.clear();
    x.clear(); 
    y.clear();
    isInner.clear();
    
}

void EventBundle::DiscriminateInner(int widht, int height)
{
    isInner.resize(size); 

    if(x.size() != isInner.size()) cout << "inner wrong size" << endl;

    for(uint32_t i; i < size; ++i)
    {
        if(x[i]<0 || x[i]>=widht || y[i]<0 || x[i]>=height) isInner[i] = false;
        else isInner[i] = true;
    }
}

void EventBundle::Append(EventData& eventData)
{   
    if(size == 0) 
    {
        first_tstamp = eventData.event.front().ts;
        abs_tstamp = eventData.time_stamp;
    }
    last_tstamp = eventData.event.back().ts;
    
    size_t old_size = size; 
    size += eventData.event.size(); 

    coord.conservativeResize(2,size);
    coord_3d.conservativeResize(3,size);

    time_delta.conservativeResize(size);
    time_delta_rev.conservativeResize(size);  // not used 
    
    int counter = 0; 
    for(const auto& i: eventData.event)
    {
        // TODO sampler 

        x.push_back(i.x);
        y.push_back(i.y);
        polar.push_back(i.polarity==0);

        coord(0, old_size+counter) = i.x;
        coord(1, old_size+counter) = i.y;
        coord_3d(2,old_size+counter) = 1;
        time_delta(old_size+counter) = (i.ts - first_tstamp).toSec();
        counter++;
    }

    // deep copy
    // coord_3d.topRightCorner(2,size-old_size) = coord.topRightCorner(2,size-old_size); 

    // cout << "coord example" << endl;
    // for(int i=0; i< 5; i++)
    //     cout << eventData.event[i].x << "," << eventData.event[i].y << endl;
    // cout << coord.topLeftCorner(2,5)  << endl; 
    // cout << coord_3d.topLeftCorner(3,5) << endl;
}


/**
* \brief project camera to unisophere 
* \param K, camera proj matrix. from coor -> coor_3d
*/
void EventBundle::InverseProjection(Eigen::Matrix3f& K)
{
    // eigen array pixel-wise operates 

    cout << "inverse project \n " << endl;
    // cout << "coord size " << coord.size()  <<  "," << coord_3d.size() << endl;
    
    if(coord_3d.cols() != size)
    {
        coord_3d.conservativeResize(3, size);
        cout << "resizing coord_3d" << endl;
    }   
    // cout << coord_3d.topLeftCorner(3,5) << endl;

    coord_3d.row(0) = (coord.row(0).array()-K(0,2)) / K(0,0);
    coord_3d.row(1) = (coord.row(1).array()-K(0,2)) / K(1,1);
    
    coord_3d.row(2) = Eigen::MatrixXf::Ones(1, size);

}


/**
* \brief project camera to unisophere 
* \param K, camera proj matrix. from coor_3d -> coor
*/
void EventBundle::Projection(Eigen::Matrix3f& K)
{
    // eigen array pixel-wise operates 
    if(coord.cols() != size)
    {
        coord.conservativeResize(3, size);
        cout << "resizing coord 2d" << endl;
    }   

    coord.row(0) = coord_3d.row(0).array() / coord_3d.row(2).array() * K(0,0) + K(0,2);
    coord.row(1) = coord_3d.row(1).array() / coord_3d.row(2).array() * K(1,1) + K(1,2);
    coord = coord.array().round(); // pixel wise 
}


/**
* \brief copy the size paramter of another eventbundle.
*/
void EventBundle::CopySize(const EventBundle& ref)
{
    time_delta = ref.time_delta; 

    size = ref.size; 

    coord.conservativeResize(2,size);
    coord_3d.conservativeResize(3,size);
    

}