
#%% Average Relative Pose Errors(ARPE) in rad 
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import interp 
from scipy.interpolate import interp1d
import math 
from scipy.spatial.transform import Rotation 
from scipy.spatial.transform import Slerp

#%% using read_gtvelocity.py in Zhu-RAL folder to get gt angular and translation velocity   

# rpg rotation
gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/boxes_rotation/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/poster_rotation/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/dynamic_rotation/groundtruth.txt"

# ZHU 
# gt_filename = "/home/hxy/Documents/rosbag/Zhu-RAL/indoor_flying1/indoor_flying1_left_pose_tum.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Zhu-RAL/outdoor_driving1/outdoor_driving1_left_pose_tum.txt"

# EMIN
    # boxes   ARRE 0.008248  
    # poster  ARRE 0.011082 
    # dynamic ARRE 0.009667
    # 6DOF indoor1   ARRE 0.005864 
    # 6DOF outdoor1  ARRE 0.009482 
# est_filename = "/home/hxy/Desktop/EventEMin/output/rotation/boxes/whiten_ApproPoten_size30k_estimates_rms7.71.txt"     
# est_filename = "/home/hxy/Desktop/EventEMin/output/rotation/poster/whiten_ApproPoten_size30k_estimates_rms12.189.txt"  
# est_filename = "/home/hxy/Desktop/EventEMin/output/rotation/dynamic/whiten_ApproPoten_size30k_estimates_rms6.40.txt" 
# est_filename = "/home/hxy/Desktop/EventEMin/output/6dof/indoor_flying1/whiten_ApproPoten_size20k_estimates.txt"         
# est_filename = "/home/hxy/Desktop/EventEMin/output/6dof/outdoor_driving1/whiten_ApproPoten_size30k_estimates.txt"       

# HXY 
    # boxes   ARRE 0.006838
    # poster  ARRE 0.009440
    # dynamic ARRE 0.008131
    # 6DOF indoor1   ARRE 0.005796
    # 6DOF outdoor1  ARRE 0.009482 
est_filename = "/home/hxy/Desktop/hxy-rotation/data/rotation_estimation/boxes_rotation/ransac_size30k_double_warp_30000_timerange(0.2-0.8)_iter30_ceres10_gaussan5_sigma1.0_denoise4_defaultval1.0_rms6.63.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/rotation_estimation/poster_rotation/ransac_size30k_double_warp_2000_timerange(0.2-0.8)_iter50_ceres10_gaussian5_sigma1_denoise6_rms10.28.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/rotation_estimation/dynamic_rotation/ransac_size30k_double_warp_6000_timerange(0.2-0.8)_iter30_ceres10_gaussan5_sigma1.0_denoise4_defaultval1.0_rms5.22.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/indoor_fly1/poster_ransac_size30k_double_warp_5000_timerange(0.8-0.8)_iter20_ceres10_gaussan5_sigma1.0_denoise4_defaultval1.0.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/outdoor_driving1/poster_ransac_size30k_double_warp_10000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4_defaultval1.0.txt"

# STPPP 
    # boxes   ARRE 0.006840
    # poster  ARRE 0.009459
    # dynamic ARRE 0.008229
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/dataset/boxes_rotation/est_stppp_size_30k_rms6.73.txt"     
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/dataset/poster_rotation/est_stppp_size_30k_rms10.37.txt"   
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/dataset/dynamic_rotation/est_stppp_size_30k_rms5.18.txt"   


# read estimated data 
est_data = np.loadtxt(est_filename)
est_data[:, 0] = (est_data[:,1] + est_data[:,2]) / 2
if 'PPP' in est_filename:
    est_data = est_data[:,(0, 1,2, 4,5,6)]

# visualize the velocity 
# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Mono-unzip/dynamic_rotation/gt_velocity.txt")  # abs pos from cam to world
# plt.figure(figsize=(15, 9))
# plt.subplot(3, 1, 1)
# plt.title("relative rotation velocity ")
# plt.plot(gt_vel[:,0], gt_vel[:,1], linewidth=1)   # x 
# plt.plot(est_data[:,0], est_data[:,3], linewidth=1)   # x 
# plt.legend(["gt_x", "est_x"])
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(gt_vel[:,0], gt_vel[:,2], linewidth=1)   # y 
# plt.plot(est_data[:,0], est_data[:,4], linewidth=1)   # y 
# plt.legend(["gt_y", "est_y"])
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(gt_vel[:,0], gt_vel[:,3], linewidth=1)   # z
# plt.plot(est_data[:,0], est_data[:,5], linewidth=1)   # z
# plt.grid()
# plt.legend(["gt_z", "est_z"])
# plt.xlabel("time(s)")
# plt.ylabel("velocity (rad/s)")

# plt.show()

# for each event batch, inter using t1 and t2 get R_gt of t2_t1 
# reference to wiki https://en.wikipedia.org/wiki/Homography_(computer_vision)
#%% calculate ARPE ARRE 
import numpy as np 
import scipy.linalg as lg 

def angAxis2mat(v):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
def mat2angAxis(m):
    return np.array([m[2,1],m[0,2],-m[1,0]])


RRE = []


# abs pos from cam to world
gt_pose = np.loadtxt(gt_filename)  # (timestamp px py pz qx qy qz qw)

gt_quat_interp = Slerp(gt_pose[:,0], Rotation.from_quat(gt_pose[:,4:]))


# RRE 
gt_euler_list, est_euler_list = [], []
for i in range(3, est_data.shape[0]-3):  # 避免超出插值范围
    t1, t2 = est_data[i,1], est_data[i,2]
    delta_t = t2 - t1

    gt_quat_t1 = gt_quat_interp(t1) 
    gt_quat_t2 = gt_quat_interp(t2) 
    gt_ang_mat = (gt_quat_t1.inv() * gt_quat_t2)      # from camera to world 
    # est_ang_mat = Rotation.from_euler('xyz',est_data[i, 3:6]*delta_t ,degrees=False)  # from camera to world 
    est_ang_mat = Rotation.from_rotvec(est_data[i, 3:6]*delta_t)  # from camera to world 
    relative_ang_axis = mat2angAxis(lg.logm((est_ang_mat.inv()*gt_ang_mat).as_matrix())) 
    RRE.append(lg.norm(relative_ang_axis))

    gt_ang_euler = (gt_quat_t1.inv() * gt_quat_t2).as_euler('xyz')      # from camera to world 
    gt_euler_list.append(np.hstack(((t1+t2)/2, gt_ang_euler)))
    est_euler_list.append(np.hstack(((t1+t2)/2, est_data[i, 3:6]*delta_t)))
    pass


gt_euler_list = np.array(gt_euler_list)
est_euler_list = np.array(est_euler_list)

ARRE = np.mean(RRE)
print("ARRE {:.6f} ".format(ARRE))



plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.title("abs rotation angualr")
plt.plot(gt_euler_list[:,0], gt_euler_list[:,1], linewidth=1)   # x 
plt.plot(est_euler_list[:,0], est_euler_list[:,1], linewidth=1)   # x 
plt.legend(["gt_x", "est_x"])
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(gt_euler_list[:,0], gt_euler_list[:,2], linewidth=1)   # y 
plt.plot(est_euler_list[:,0], est_euler_list[:,2], linewidth=1)   # y 
plt.legend(["gt_y", "est_y"])
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(gt_euler_list[:,0], gt_euler_list[:,3], linewidth=1)   # z
plt.plot(est_euler_list[:,0], est_euler_list[:,3], linewidth=1)   # z
plt.grid()
plt.legend(["gt_z", "est_z"])
plt.xlabel("time(s)")
plt.ylabel("rotation (rad)")


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

