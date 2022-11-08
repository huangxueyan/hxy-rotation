import rospy 
import numpy as np
import rosbag 

from dvs_msgs.msg import Event, EventArray 
from sensor_msgs.msg import CameraInfo

folder = "/home/hxy/Documents/rosbag/Mono-unzip/poster_rotation/"
# folder = "/home/hxy/Documents/rosbag/DVSMOTION20/Camera_Motion_Data_Mat/classroom_sequence/"
filename = "events.txt"

event_data = np.loadtxt(folder + filename, max_rows = int(3e4))

DELTA_t = 0.01
START_TIME, END_TIME = event_data[0,0], event_data[-1,0]
TOTAL = event_data.shape[0]
with rosbag.Bag(folder + 'output_deltat{:.2f}.bag'.format(DELTA_t), 'w') as bag:

    idx = 0
    for bag_i in range(1, int((END_TIME - START_TIME) / DELTA_t) + 1):
        timestamp = rospy.Time.from_sec(event_data[idx, 0])
        evs = EventArray()
        evs.header.stamp = timestamp
        # evs.width = 364
        # evs.height = 260
        evs.width = 240
        evs.height = 180
        evs.events = []
        t0 = event_data[idx, 0]
        while idx < TOTAL and event_data[idx, 0] - t0 < DELTA_t:
            ev = Event()
            ev.x = np.uint(event_data[idx, 1])
            ev.y = np.uint(event_data[idx, 2])
            ev.ts = rospy.Time.from_sec(event_data[idx, 0])
            ev.polarity = np.bool(event_data[idx, 3])
            evs.events.append(ev)
            idx += 1

        bag.write("/dvs/events", evs, timestamp)

        # add camera info 
        # cam = CameraInfo()
        # cam.header.stamp = timestamp
        # cam.height = 260
        # cam.width = 364
        # cam.distortion_model = "plumb_bob"
        # cam.D = [-0.361699983842314, 0.158697458565572, 0, 0, 0.0]
        # cam.K = [289.7740, 0.0, 185.97, 0.0, 288.8098, 132.5850, 0, 0, 1]
        # bag.write("/dvs/camera_info", cam, timestamp)

