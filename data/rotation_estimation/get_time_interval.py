from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

folder = "/home/hxy/Desktop/ECCV22-all/hxy-rotation/data/rotation_estimation/shapes_rotation/"
# folder = "/home/hxy/Desktop/ECCV22-all/hxy-rotation/data/rotation_estimation/classroom_sequence/"
est_filename = "221103_baset0.01_resetzero_dynasize5k_double_warp_30000_outlier50_batchtime40ms_batchlength50_timerange(0.8-0.8)_iter1_ceres10_gaussan5_sigma1.0_denoise4.txt"
# gt_filename = "221104_baset0.01_final5_resetzero_dynasize5k_double_warp_1000_outlier50_batchtime20ms_batchlength150_timerange(0.8-0.8)_iter1_ceres10_gaussan5_sigma1.0_denoise4.txt"
est_data = np.loadtxt(folder + est_filename)

t_list, size_list = [], []
for vel_info in est_data:
    t_list.append((vel_info[1] + vel_info[2]) / 2)
    size_list.append(vel_info[6])


plt.figure()
plt.bar(t_list, size_list)
plt.show()


# get event count
folder = "/home/hxy/Documents/rosbag/Mono-unzip/poster_rotation/"
filename = "events.txt"
events = pd.read_csv(folder + filename, sep=" ", header=None, skiprows=0, nrows=None)
events.columns = ["ts", "x", "y", "p"]
events_set = events.to_numpy()
N_ROW = events_set.shape[0]

size_list, time_list = [], []
idx = 0
for t in range(60):
    pre_idx = idx
    while idx < N_ROW and  events_set[idx, 0] < t + 1:
        idx += 1
    time_list.append(t)
    size_list.append(idx - pre_idx)

plt.figure()
plt.bar(time_list, size_list)

plt.show()

pass