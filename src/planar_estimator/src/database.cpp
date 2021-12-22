#include "database.hpp"


using namespace std; 


CameraPara::CameraPara()
{
    width = 240; 
    height = 180; 

    // fx = 230.2097;
    // fy = 231.1228;
    // cx = 121.6862;
    // cy = 86.8208;
    // rd1 = -0.4136;
    // rd2 = 0.2042;

    fx = 199.092366542;
    fy = 198.82882047;
    cx = 132.192071378;
    cy = 110.712660011;
    k1 = -0.368436311798;
    k2 = 0.150947243557;
    p1 = -0.000296130534385;
    p2 = -0.000759431726241;
    k3 = 0.0;

     


    cameraMatrix = (cv::Mat_<float>(3,3) << 
                    fx, 0, cx , 0, fy, cy, 0, 0, 1); 
    distCoeffs = (cv::Mat_<double>(1,5) << k1, k2, p1, p2, k3); 

    // cout << "camera param:" << endl;
    // cout << cameraMatrix << endl; 

    cv::cv2eigen(cameraMatrix, eg_cameraMatrix);
    // cout << eg_cameraMatrix << endl; 

    // the map展示的虚拟相机k内参可以自定义大小。 
    height_map = 2 * height; 
    width_map = 2 * width;
    eg_MapMatrix = eg_cameraMatrix; // deep copy

    eg_MapMatrix(0,2) = width_map/2;
    eg_MapMatrix(1,2) = height_map/2;
    
    cout << "camera matrix:: \n" << cameraMatrix << endl;
    cout << "camera matrix eigen: \n" << eg_cameraMatrix << endl;
    cout << "map matrix eigen: \n" << eg_MapMatrix << endl;

}

/**
* \brief make a reference, TODO it a more efficient way?.
*/
EventBundle::EventBundle()
{
    size = 0;
    coord.resize(2, size);
    coord_3d.resize(3, size);
}


/**
* \brief make a reference, TODO it a more efficient way?.
*/
EventBundle::EventBundle(const EventBundle& eb)
{
    // deep copy 
    coord = eb.coord;
    coord_3d = eb.coord_3d;
    isInner = eb.isInner;


    angular_position = eb.angular_position;
    angular_velocity = eb.angular_velocity;

    time_delta = eb.time_delta; 

    size = eb.size; 

    // x = eb.x;
    // y = eb.y; 
    polar = eb.polar;

}


/**
* \brief reset all.
*/
void EventBundle::Clear()
{
    // cout << "event bundle clearing, size " <<  size << endl;

    size = 0; 

    // eigen reconstruct
    coord.resize(2,size);
    coord_3d.resize(3,size);
    isInner.resize(size); 

    time_delta.resize(size);
    time_delta_rev.resize(size);
    
    angular_position = Eigen::Vector3d::Zero();
    angular_velocity = Eigen::Vector3d::Zero();
    
    // vector 
    // time_stamps.clear();
    // x.clear(); 
    // y.clear();
    // isInner.clear();
    
    // cout << "  event bundle clearing sucess" << endl;
}


/**
* \brief strong DiscriminateInner, for the convinient of gradient .
* \param width  given boundary, may camera boundary or map view boundary
* \param height 
*/
void EventBundle::DiscriminateInner(int width, int height)
{
    // cout << "DiscriminateInner "<< size << endl;
    isInner.resize(size); 
    // if(x.size() != isInner.size()) cout << "inner wrong size" << endl;

    for(uint32_t i = 0; i < size; ++i)
    {
        if(coord(0,i)<3 || coord(0,i)>=(width-3) || coord(1,i)<3 || coord(1,i)>=(height-3)) 
            isInner[i] = 0;
        else isInner[i] = 1;
    }
}

EventBundle::~EventBundle(){

}


/**
* \brief append eventData to event bundle, resize bundle.
*/
void EventBundle::Append(std::vector<dvs_msgs::Event>& vec_eventData)
{   
    if(size == 0) 
    {
        // cout << "first appending events, "<< vec_eventData.size() << endl;
        first_tstamp = vec_eventData.front().ts;
        // abs_tstamp = eventData.time_stamp;
    }
    else
    {
        cout << "appending events" << endl;
    }
    
    last_tstamp = vec_eventData.back().ts;
    
    size_t old_size = size; 
    size += vec_eventData.size(); 

    coord.conservativeResize(2,size);
    coord_3d.conservativeResize(3,size);
    polar.conservativeResize(size);

    time_delta.conservativeResize(size);
    time_delta_rev.conservativeResize(size);  // not used 
    
    int counter = 0; 
    for(const auto& i: vec_eventData)
    {
        // TODO sampler 

        // x.push_back(i.x);
        // y.push_back(i.y);
        polar(old_size+counter) = i.polarity==0;

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
void EventBundle::InverseProjection(Eigen::Matrix3d& K)
{
    // eigen array pixel-wise operates 

    // cout << "inverse project \n " << endl;
    // cout << "coord size " << coord.size()  <<  "," << coord_3d.size() << endl;
    
    if(coord_3d.cols() != size)
    {
        coord_3d.resize(3, size);
        cout << "resizing coord_3d" << endl;
    }   

    coord_3d.row(0) = (coord.row(0).array()-K(0,2)) / K(0,0);
    coord_3d.row(1) = (coord.row(1).array()-K(1,2)) / K(1,1);
    
    coord_3d.row(2) = Eigen::MatrixXd::Ones(1, size);
    
    // cout << coord_3d.topLeftCorner(3,5) << endl;
    // coord_3d.colwise().normalize();

    // cout << coord_3d.topLeftCorner(3,5) << endl;
    
    // cout << coord_3d.bottomRightCorner(3,5) << endl;

}


/**
* \brief project camera to unisophere 
* \param K, camera proj matrix. from coor_3d -> coor
*/
void EventBundle::Projection(Eigen::Matrix3d& K)
{
    // eigen array pixel-wise operates 
    if(coord.cols() != size)
    {
        coord.resize(3, size);
        cout << "---------wronging----------" << endl;
        cout << "resizing coord 2d" << endl;
    }   

    coord.row(0) = coord_3d.row(0).array() / coord_3d.row(2).array() * K(0,0) + K(0,2);
    coord.row(1) = coord_3d.row(1).array() / coord_3d.row(2).array() * K(1,1) + K(1,2);

    // TODO add gaussian 
    // coord = coord.array().round(); // pixel wise 


    // cout << "  projecting sucess " << endl;
}


/**
* \brief copy the size paramter of another eventbundle.
*/
void EventBundle::CopySize(const EventBundle& ref)
{
    // time_delta = ref.time_delta; 

    size = ref.size; 
    
    coord.resize(2,size);
    coord_3d.resize(3,size);
    isInner.resize(size); 
    

}