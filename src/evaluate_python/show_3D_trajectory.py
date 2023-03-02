
#%% read data 
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt 
import numpy as np 

#%% gt part 

# data_path = "/home/hxy/Documents/rosbag/Zhu-RAL/indoor_flying1/"
# filename = "gt_trans_rot_velocity.txt"
# est_data = np.loadtxt(data_path + filename, dtype=np.float32)  # t0, wx, wy, wz, vx, vy, vz     

# pose_list = []
# for i in range(est_data.shape[0]-1):
#     delta_t = est_data[i+1,0] - est_data[i, 0]
#     if i == 0:
#         quat_01 = Rotation.from_rotvec(est_data[i, 1:4]*delta_t)
#         trans_01 = est_data[i, 4:7] * delta_t
#         last_pos = np.hstack((est_data[i, 0], trans_01, quat_01.as_quat()))
#         pose_list.append(last_pos)
#         continue

#     quat_01 = Rotation.from_quat(last_pos[4:])
#     trans_01 = last_pos[1:4]
#     quat_12 = Rotation.from_rotvec(est_data[i, 1:4]*delta_t)
#     trans_12 = est_data[i, 4:] * delta_t

#     quat_02 = quat_01 * quat_12
#     trans_02 = np.matmul(quat_01.as_matrix(), trans_12) + trans_01

#     last_pos = np.hstack((est_data[i, 0], trans_02, quat_02.as_quat()))
#     pose_list.append(last_pos)

# # pose_list = np.hstack((est_data[:,0]-delta_t_list/2, pose_list))

# np.savetxt(data_path + "gt_trans_rot_velocity_traj.txt", pose_list, "%6f" + " %6f"*7)  # time + xyz + quat



#%% 
# EMIN 
# data_path = "/home/hxy/Desktop/EventEMin/output/6dof/indoor_flying1/"
# filename = "test3d_scale100_incTsallis_size15k_estimates.txt"
# data_path = "/home/hxy/Desktop/EventEMin/output/6dof/outdoor_driving1/"
# filename = "neg_guanting_10_200_ApproTsllis_estimates.txt"


# HXY 
# data_path = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/indoor_flying1/"
# filename = "inner_norm0.01_ransac_size30k_double_warp_30000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise2.txt"
data_path = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/outdoor_driving1/"
filename = "inner_norm0.01_ransac_size30k_double_warp_30000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise2.txt"



est_data = np.loadtxt(data_path + filename, dtype=np.float32)  # idx, t0, t1, wx, wy, wz, vx, vy, vz     
est_data[:, 0] = (est_data[:, 1] + est_data[:, 2]) / 2



pose_list = []
last_pos = []
for i in range(0, est_data.shape[0]):  # 避免超出插值范围
    t1, t2 = est_data[i, 1], est_data[i, 2]
    delta_t = t2 - t1

    if i == 0:
        quat_01 = Rotation.from_rotvec(est_data[i, 3:6]*delta_t)
        trans_01 = est_data[i, 6:9] * delta_t
        last_pos = np.hstack((est_data[i, 0], trans_01, quat_01.as_quat()))
        pose_list.append(last_pos)
        continue

    quat_01 = Rotation.from_quat(last_pos[4:])
    trans_01 = last_pos[1:4]
    quat_12 = Rotation.from_rotvec(est_data[i, 3:6]*delta_t)
    trans_12 = est_data[i, 6:9] * delta_t

    quat_02 = quat_01 * quat_12
    trans_02 = np.matmul(quat_01.as_matrix(), trans_12) + trans_01

    last_pos = np.hstack((est_data[i, 0], trans_02, quat_02.as_quat()))
    pose_list.append(last_pos)

np.savetxt(data_path + "_traj.txt", pose_list, "%6f" + " %6f"*7)  # time + xyz + quat


exit()
#%% accumulate part 

delta_t = velcity_txt[:,2] - velcity_txt[:,1] 
delta_pose_list = np.zeros((size, 6))
for i in range(6):   # relative ang and pos 
    delta_pose_list[:, i] =  velcity_txt[:, 3+i] * delta_t

pose_list = np.zeros((size, 6), dtype=np.float32) 
pre_pose = np.zeros((6), dtype=np.float32)
for i in range(size):
    pose_list[i] = pre_pose + delta_pose_list[i]
    pre_pose = pose_list[i]


plt.figure()
plt.subplot(3, 1, 1)
plt.title("accumulated angular ")
plt.plot(time_t, pose_list[:, 0] * 180/3.14)
plt.subplot(3, 1, 2)
plt.plot(time_t, pose_list[:, 1] * 180/3.14)
plt.subplot(3, 1, 3)
plt.plot(time_t, pose_list[:, 2] * 180/3.14)
plt.xlabel("time(s)")
plt.ylabel("velocity (degree)")

plt.show()

exit()

pose_list = np.concatenate((np.expand_dims(time_t, 0).transpose(), pose_list), axis=1)
np.savetxt(data_path + "est_pose.txt", pose_list, fmt="%.6f" + " %.6f"*7)  # kitti only need 12 transform matrix 


pass


