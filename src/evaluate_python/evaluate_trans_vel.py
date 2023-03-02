
#%% Average Relative Pose Errors(ARPE) in rad 
from turtle import color
from matplotlib import markers
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import interp 
from scipy.interpolate import interp1d
import math 

import numpy as np 
import scipy.linalg as lg 


#%% using read_gtvelocity.py in Zhu-RAL folder to get gt angular and translation velocity   

# rpg rotation
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/slider_far/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/poster_rotation/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/dynamic_rotation/groundtruth.txt"
gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Zhu-RAL/indoor_flying1/gt_trans_rot_velocity.txt")

# read estimated data 
est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/indoor_flying1/inner_norm0.01_ransac_size30k_double_warp_1500_timerange(0.5-0.5)_iter3_ceres3_gaussan5_sigma1.0_denoise2.txt"
est_filename2 = "/home/hxy/Desktop/EventEMin/output/6dof/indoor_flying1/NoWhiten_ApproTsallis_size30k_estimates.txt"

est_data = np.loadtxt(est_filename)
est_data[:, 0] = (est_data[:,1] + est_data[:,2]) / 2

est_data2 = np.loadtxt(est_filename2)
est_data2[:, 0] = (est_data2[:,1] + est_data2[:,2]) / 2


# # translation
plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.title("estimated translation velocity (m/s)")
plt.plot(gt_vel[::4, 0], gt_vel[::4, 4], '--', color='#808080', linewidth=2, alpha=0.9)   # x
plt.plot(est_data[:, 0], est_data[:, 6],       color='#7a95c4', linewidth=2, alpha=0.9)   # x
plt.plot(est_data2[:, 0], est_data2[:, 6],     color='coral', linewidth=2, alpha=0.8)   # x
plt.xlim([5,35])
plt.xticks([10,20,30],[])
plt.yticks([0.4,0,-0.4])
# plt.legend(["gt_x", "est_x"])
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(gt_vel[::4, 0], gt_vel[::4, 5], '--', color='#808080', linewidth=2, alpha=0.9)   # x
plt.plot(est_data[:, 0], est_data[:, 7],       color='#7a95c4', linewidth=2, alpha=0.9)   # x
plt.plot(est_data2[:, 0], est_data2[:, 7],     color='coral', linewidth=2, alpha=0.8)   # x
# plt.legend(["gt_y", "est_y"])
plt.yticks([0.3, 0,-0.3])
plt.xticks([10,20,30,],[])
plt.xlim([5,35])
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(gt_vel[::4, 0], gt_vel[::4, 6], '--', color='#808080', linewidth=2, alpha=0.9)   # x
plt.plot(est_data[:, 0], est_data[:, 8],       color='#7a95c4', linewidth=2, alpha=0.9)   # x
plt.plot(est_data2[:, 0], est_data2[:, 8],     color='coral', linewidth=2, alpha=0.8)   # x
plt.grid()
# plt.legend(["gt_z", "_z"])
plt.xticks([10,20,30,])
plt.yticks([0.5,0,-0.5])
plt.xlim([5,35])

plt.xlabel("time(s)")
plt.ylabel("velocity (m/s)")

plt.show()

exit()

#%% utiles 
# # Average Relative Rotation Errors(ARRE) in degree
import numpy as np 
import scipy.linalg as lg 

def angAxis2mat(v):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
def mat2angAxis(m):
    return np.array([m[2,1],m[0,2],-m[1,0]])



r_est = np.array([0.4894532, 0.4894532, 0.1249779])
r_gt = np.array([1,2,3])


R_est = lg.expm(angAxis2mat(r_est))
R_gt = lg.expm(angAxis2mat(r_gt))

RRE = lg.norm(mat2angAxis(lg.logm(R_est.transpose()*R_gt)))


#%% utiles 
#%% ARPE calculation 

t_est = np.array([1,2,3])
t_gt = np.array([1,2,3])
norm_t1_t2 = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
prod_t1_t2 = np.matmul(t_est, np.transpose(t_gt)) 

RPE = np.math.acos(prod_t1_t2 / norm_t1_t2)

