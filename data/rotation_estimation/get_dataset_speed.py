from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

folder = "/home/hxy/Documents/rosbag/Mono-unzip/poster_rotation/"
# folder = "/home/hxy/Documents/rosbag/DVSMOTION20/Camera_Motion_Data_Mat/classroom_sequence/"
filename = "events.txt"

COUNT = 10000
events = pd.read_csv(folder + filename, sep=" ", header=None, skiprows=0, nrows=100000)
events.columns = ["ts", "x", "y", "p"]
events_set = events.to_numpy()
N_ROW = events_set.shape[0]

speed_list, time_list = [], []
for i in range(int(N_ROW / COUNT)):

    delta_t = events_set[(i+1)*COUNT - 1, 0] - events_set[i*COUNT, 0] 
    speed_list.append(COUNT / delta_t / 1e6)

duration = events_set[-1][0] - events_set[0][0]


print("total {:0.2f} Mevs, duration {:0.2f} s, avg {:0.2f} Mev/s".format(N_ROW/1e6, duration, N_ROW / duration / 1e6))
print("max speed {:0.2f} M/s, min sppeed {:0.2f} M/s".format(max(speed_list), min(speed_list)))

pass