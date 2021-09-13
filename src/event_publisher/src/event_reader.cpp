
#include "event_reader.hpp"



Event_reader::Event_reader(std::string yaml,
        ros::Publisher* event_array_pub , 
        ros::Publisher* event_image_pub )  // default 只需在定义时写一遍即可
{

    // read yaml setting
    cv::FileStorage fSetting(yaml,cv::FileStorage::READ);
    if(!fSetting.isOpened())
    {
        throw std::runtime_error(string("could not open file") + yaml);
    }
    
    height = fSetting["height"];
    width = fSetting["width"];

    event_bundle_size = 5;
    eventData.reserve(size_t(1e6));
    eventData_counter = 0;
    
    event_array_pub_ = event_array_pub;
    event_image_pub_ = event_image_pub;
}

Event_reader::~Event_reader() {}


void Event_reader::read(const string& dir)
{
    string filename = dir + "/events.txt"; 

    std::ifstream openFile(filename, std::ios::in);
    if(!openFile.is_open())
    {
        throw std::runtime_error("counld not read file");
    }

    string line, token;
    vector<string> vToken;

    double t_stamp; 
    uint16_t x_pos, y_pos; 
    uint8_t polar; 

    dvs_msgs::Event msg; 
    while(getline(openFile,line))
    {
        stringstream ss(line);

        ss >> t_stamp >> msg.x >> msg.y >> polar;
        msg.ts = ros::Time(t_stamp);
        msg.polarity = polar;
        // cout << msg.ts << " "<< msg.x << " "<<msg.y << " "<<msg.polarity << endl;

        eventData.push_back(msg);
    }

    cout << "read events : \n" << msg.ts << " "<< msg.x << " "<<msg.y << " "<<msg.polarity << endl;


    openFile.close();

}


/**
* \brief publish events to.
*/
void Event_reader::publish()
{

    dvs_msgs::EventArrayPtr msg = dvs_msgs::EventArrayPtr(new dvs_msgs::EventArray()); 
    msg->height = height;
    msg->width = width;

    int count = 0;
    while(count++ < event_bundle_size)
    {
        msg->header.seq = eventData_counter;
        msg->header.stamp = eventData[eventData_counter].ts;
        msg->events.push_back(eventData[eventData_counter++]);

        if(eventData_counter == eventData.size())
        {
            cout << "publish at the end of file" << endl;
            break;
        }
    }

    
    // publish event image 
    render();
    event_image.encoding = "mono8";
    event_image.image = cv::imread("/home/hxt/Pictures/test.jpg",cv::IMREAD_GRAYSCALE);
    event_image_pub_->publish(event_image.toImageMsg());

    // publish events 
    cout << "sending " << event_bundle_size << "events" << endl;
    event_array_pub_->publish(msg);
    
}



/**
* \brief render event into event images.
*/
void Event_reader::render()
{




}
