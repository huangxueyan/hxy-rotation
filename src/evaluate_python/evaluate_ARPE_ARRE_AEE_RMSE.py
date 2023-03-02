
# %% Average Relative Pose Errors(ARPE) in rad
import scipy.linalg as lg
from scipy.spatial.transform import Slerp
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import math
from scipy.spatial.transform import Rotation

# rpg gt in TUM format
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/poster_translation/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/poster_6dof/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/poster_hdr/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/simulation_3walls/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Mono-unzip/simulation_3planes/groundtruth.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Zhu-RAL/indoor_flying1/indoor_flying1_left_pose_tum.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Zhu-RAL/outdoor_driving1/outdoor_driving1_left_pose_tum.txt"

save_flag = False

# EMIN
# HOMO poster_trans ARPE 27.198909 ARRE 0.014113 RMSw 24.9 # RMS using STPPP 
# HOMO poster_6dof  ARPE 69.799378 ARRE 0.018368 RMSw 43.8 # RMS using STPPP 
# HOMO poster_hdr   ARPE 94.032964 ARRE 0.024313 RMSw 46.2 # RMS using STPPP 
# 6DOF indoor1      ARPE 14.658866 ARRE 0.007729 AEE 0.161016 RMSt 0.112890 RMSw 2.275805
# 6DOF outdoor1     ARPE           ARRE FAILED
# est_filename = "/home/hxy/Desktop/EventEMin/output/homography/poster_translation/whiten_ApproPoten_size30k_estimates_RPE27.19_RRE0.014.txt"
# est_filename = "/home/hxy/Desktop/EventEMin/output/homography/poster_6dof/neg_trans_whiten_ApproPoten_size30k_estimates_RPE69.83_RRE0.018_rms43.8ppp.txt"
# est_filename = "/home/hxy/Desktop/EventEMin/output/homography/poster_hdr/neg_whiten_ApproPoten_size20k_estimates_failed_rms64.txt"
# est_filename = "/home/hxy/Desktop/EventEMin/output/6dof/indoor_flying1/test3d_scale100_incTsallis_size15k_estimates.txt"
# est_filename = "/home/hxy/Desktop/EventEMin/output/6dof/outdoor_driving1/neg_outdor_10_200_ApproTsllis_estimates.txt"


# HXY
# HOMO trans     ARPE 18.98349 ARRE 0.011214 RMSw 23.7  # RMS using STPPP 
# HOMO 6dof      ARPE 26.63638 ARRE 0.013190 RMSw 29.22 # RMS using STPPP 
# HOMO hdr       ARPE 45.34167 ARRE 0.015057 RMSw 43.54  # RMS using STPPP 
# 6DOF indoor1   ARPE 6.904698 ARRE 0.005956 AEE 0.07641 RMSt 0.050894 RMSw 1.633804
# 6DOF outdoor1  ARPE 11.69259 ARRE 0.008256 AEE 1.23532 RMSt 0.936438 RMSw 4.328851 # caused by outlier
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/planar_estimation/poster_translation/double_inv_exactly_norm3d_size30k_regular10_10000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/planar_estimation/poster_6dof/single_inv_exactly_norm3d_size30k_regular10_30000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/planar_estimation/poster_hdr/double_inv_exactly_norm3d_size30k_regular10_30000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4.txt"
est_filename = "/home/hxy/Desktop/hxy-rotation/data/planar_estimation/simulation_3walls/single_inv_exactly_norm3d_size10k_regular10_3000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/planar_estimation/simulation_3planes/single_inv_exactly_norm3d_size10k_regular10_3000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/indoor_flying1/inner_norm0.01_ransac_size30k_single_warp_1000_timerange(0.8-0.8)_iter3_ceres3_gaussan5_sigma1.0_denoise2.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/outdoor_driving1/iter10_10/inner_norm0.01_ransac_size30k_double_warp_10000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise2.txt"

# Pytorch-CMax
# HOMO trans     ARPE 39.23117 ARRE 0.010896 RMSw 17.14 # RMS using STPPP 
# HOMO 6dof      ARPE 48.58074 ARRE 0.015718 RMSw 51.47 # RMS using STPPP 
# HOMO hdr       ARPE 51.57235 ARRE 0.010547 RMSw 26.34 # RMS using STPPP 
# 6DOF indoor1   ARPE 11.40762 ARRE 0.006783 AEE 0.11032 RMSt 0.070512 RMSw 1.737737
# 6DOF outdoor1  ARPE 16.79299 ARRE 0.022551 AEE 1.66247 RMSt 1.304980 RMSw 15.23551
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/poster_translation/cmax_size_30k_RPE39.23_RRE_0.010_RMSw_17.14.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/poster_6dof/poster_6dof_cmax_size_30k_homo.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/poster_hdr/poster_hdr_cmax_size_30k_hdr_RPE45_RRE0.012_rmsw22.08.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/indoor_flying1/remove_inner_cmax_size_30k_6dof.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/outdoor_driving1/remo_inner_cmax_size_30k_6dof.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/simulation_3planes/simulation_3planes_cmax_size_30k_output.txt"

# Kim-CMax
# HOMO trans     ARPE 39.23117 ARRE 0.010896 RMSw 17.14 # RMS using STPPP 
# HOMO 6dof      ARPE 48.58074 ARRE 0.015718 RMSw 51.47 # RMS using STPPP 
# HOMO hdr       ARPE 51.57235 ARRE 0.010547 RMSw 26.34 # RMS using STPPP 
# 6DOF indoor1   ARPE 11.40762 ARRE 0.006783 AEE 0.11032 RMSt 0.070512 RMSw 1.737737
# 6DOF outdoor1  ARPE 16.79299 ARRE 0.022551 AEE 1.66247 RMSt 1.304980 RMSw 15.23551
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/poster_translation/cmax_size_30k_RPE39.23_RRE_0.010_RMSw_17.14.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/poster_6dof/poster_6dof_cmax_size_30k_homo.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/poster_hdr/poster_hdr_cmax_size_30k_hdr_RPE45_RRE0.012_rmsw22.08.txt"
# est_filename = "/home/hxy/Desktop/kim_rotation/output/indoor_flying1/indoor_iter100_step0.005.txt"
# est_filename = "/home/hxy/Desktop/Event-ST-PPP/output/outdoor_driving1/remo_inner_cmax_size_30k_6dof.txt"



# read estimated data
est_data = np.loadtxt(est_filename)
est_data[:, 0] = (est_data[:, 1] + est_data[:, 2]) / 2

# %% visualize
# rotation part, gt file is acquired from ZHU read_gtvelocity.py

# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Mono-unzip/poster_translation/gt_trans_rot_velocity.txt")
# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Mono-unzip/poster_6dof/gt_trans_rot_velocity.txt")
# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Mono-unzip/poster_hdr/gt_trans_rot_velocity.txt")
# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Zhu-RAL/indoor_flying1/gt_trans_rot_velocity.txt")
# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Zhu-RAL/outdoor_driving1/gt_trans_rot_velocity.txt")
# gt_vel = np.loadtxt("/home/hxy/Documents/rosbag/Zhu-RAL/outdoor_night1/gt_trans_rot_velocity.txt")

# plt.figure(figsize=(15, 9))
# plt.subplot(3, 1, 1)
# plt.title("relative rotation velocity (rad/s)")
# plt.plot(gt_vel[:, 0], gt_vel[:, 1], linewidth=1)   # x
# plt.plot(est_data[:, 0], est_data[:, 3], linewidth=1)   # x
# plt.legend(["gt_x", "est_x"])
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(gt_vel[:, 0], gt_vel[:, 2], linewidth=1)   # y
# plt.plot(est_data[:, 0], est_data[:, 4], linewidth=1)   # y
# plt.legend(["gt_y", "est_y"])
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(gt_vel[:, 0], gt_vel[:, 3], linewidth=1)   # z
# plt.plot(est_data[:, 0], est_data[:, 5], linewidth=1)   # z
# plt.grid()
# plt.legend(["gt_z", "est_z"])
# plt.xlabel("time(s)")
# plt.ylabel("velocity (rad/s)")


# # translation
# plt.figure(figsize=(15, 9))
# plt.subplot(3, 1, 1)
# plt.title("estimated translation velocity (m/s)")
# plt.plot(gt_vel[:, 0], gt_vel[:, 4], linewidth=1)   # x
# plt.plot(est_data[:, 0], est_data[:, 6], linewidth=1)   # x
# plt.legend(["gt_x", "est_x"])
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(gt_vel[:, 0], gt_vel[:, 5], linewidth=1)   # y
# plt.plot(est_data[:, 0], est_data[:, 7], linewidth=1)   # y
# plt.legend(["gt_y", "est_y"])
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(gt_vel[:, 0], gt_vel[:, 6], linewidth=1)   # z
# plt.plot(est_data[:, 0], est_data[:, 8], linewidth=1)   # z
# plt.grid()
# plt.legend(["gt_z", "est_z"])
# plt.xlabel("time(s)")
# plt.ylabel("velocity (m/s)")

# plt.show()

# for each event batch, inter using t1 and t2 get R_gt of t2_t1
# reference to wiki https://en.wikipedia.org/wiki/Homography_(computer_vision)
# %% calculate ARPE ARRE


def angAxis2mat(v):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])


def mat2angAxis(m):
    return np.array([m[2, 1], m[0, 2], -m[1, 0]])


# load gt and init interpolate
gt_pose = np.loadtxt(gt_filename)  # (timestamp px py pz qx qy qz qw)

gt_px_interp = interp1d(gt_pose[:, 0], gt_pose[:, 1], kind='linear') 
gt_py_interp = interp1d(gt_pose[:, 0], gt_pose[:, 2], kind='linear') 
gt_pz_interp = interp1d(gt_pose[:, 0], gt_pose[:, 3], kind='linear') 
gt_quat_interp = Slerp(gt_pose[:, 0], Rotation.from_quat(gt_pose[:, 4:]))


gt_trans_rotvec_list, est_trans_rotvec_list = [], []   # for visualize
RPE, ABS_Err_t, ABS_Err_w, RRE = [], [], [], []
for i in range(5, est_data.shape[0]-5):  # 避免超出插值范围
    t1, t2 = est_data[i, 1], est_data[i, 2]
    delta_t = t2 - t1

    gt_trans_t1 = np.array([gt_px_interp(t1), gt_py_interp(t1), gt_pz_interp(t1)])
    gt_trans_t2 = np.array([gt_px_interp(t2), gt_py_interp(t2), gt_pz_interp(t2)])

    gt_quat_t1 = gt_quat_interp(t1)
    gt_quat_t2 = gt_quat_interp(t2)

    # RPE
    gt_trans = np.matmul(gt_quat_t1.inv().as_matrix(), gt_trans_t2 - gt_trans_t1)
    est_trans = est_data[i, 6:9] * delta_t  # velocity * time = abs translation
    norm_t1_t2 = np.linalg.norm(est_trans) * np.linalg.norm(gt_trans)
    prod_t1_t2 = np.matmul(est_trans, np.transpose(gt_trans))
    RPE.append(np.math.acos(prod_t1_t2 / norm_t1_t2))

    # RRE
    gt_ang_rot = (gt_quat_t1.inv() * gt_quat_t2)      # from camera to world
    # est_ang_rot = Rotation.from_euler('xyz', est_data[i, 3:6]*delta_t, degrees=False)  # from camera to world
    est_ang_rot = Rotation.from_rotvec(est_data[i, 3:6]*delta_t)  # from camera to world
    relative_ang_axis = mat2angAxis(lg.logm((est_ang_rot.inv()*gt_ang_rot).as_matrix()))
    RRE.append(lg.norm(relative_ang_axis))

    # trans and rot RMSE
    ABS_Err_w.append(gt_ang_rot.as_rotvec()/delta_t - est_data[i, 3:6])
    ABS_Err_t.append(gt_trans/delta_t - est_data[i, 6:9])

    # visualize data
    gt_ang_rotvec = gt_ang_rot.as_rotvec()
    gt_trans_rotvec_list.append(np.hstack((est_data[i, 0], gt_ang_rotvec, gt_trans)))
    est_trans_rotvec_list.append(np.hstack((est_data[i, 0], est_data[i, 3:6]*delta_t, est_trans)))

# RMSE be consistent with EMIN 
RMSt = np.sqrt(np.mean(np.power(np.array(ABS_Err_t),2), axis=0))
RMSw = np.sqrt(np.mean(np.power(np.array(ABS_Err_w),2), axis=0))

print("RMSw {:.6f} degrees".format(np.mean(RMSw)*180/3.14))
print("RMSt {:.6f} ".format(np.mean(RMSt)))


ARPE = np.mean(RPE)
print("ARPE {:.6f} degrees".format(ARPE*180/3.14))

ARRE = np.mean(RRE)
print("ARRE {:.6f} ".format(ARRE))

AEE = np.mean(lg.norm(np.abs(np.array(ABS_Err_t)),axis=1))
print("AEE {:.6f} m/s".format(AEE))




# visualize the relative pose from frame2 to frame1
gt_trans_rotvec_list = np.array(gt_trans_rotvec_list)
est_trans_rotvec_list = np.array(est_trans_rotvec_list)

# rotation part
plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.title("relative rotation angualr (rad)")
plt.plot(gt_trans_rotvec_list[:, 0],
         gt_trans_rotvec_list[:, 1], linewidth=1)   # x
plt.plot(est_trans_rotvec_list[:, 0],
         est_trans_rotvec_list[:, 1], linewidth=1)   # x
plt.legend(["gt_x", "est_x"])
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(gt_trans_rotvec_list[:, 0],
         gt_trans_rotvec_list[:, 2], linewidth=1)   # y
plt.plot(est_trans_rotvec_list[:, 0],
         est_trans_rotvec_list[:, 2], linewidth=1)   # y
plt.legend(["gt_y", "est_y"])
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(gt_trans_rotvec_list[:, 0],
         gt_trans_rotvec_list[:, 3], linewidth=1)   # z
plt.plot(est_trans_rotvec_list[:, 0],
         est_trans_rotvec_list[:, 3], linewidth=1)   # z
plt.grid()
plt.legend(["gt_z", "est_z"])
plt.xlabel("time(s)")
plt.ylabel("rotation (rad)")

if save_flag:
    plt.savefig(est_filename[:-4] + "_RRE{:.5f}.png".format(ARRE))

# translation part
plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.title("relative translation (m)")
plt.plot(gt_trans_rotvec_list[:, 0],
         gt_trans_rotvec_list[:, 4], linewidth=1)   # x
plt.plot(est_trans_rotvec_list[:, 0],
         est_trans_rotvec_list[:, 4], linewidth=1)   # x
plt.legend(["gt_x", "est_x"])
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(gt_trans_rotvec_list[:, 0],
         gt_trans_rotvec_list[:, 5], linewidth=1)   # y
plt.plot(est_trans_rotvec_list[:, 0],
         est_trans_rotvec_list[:, 5], linewidth=1)   # y
plt.legend(["gt_y", "est_y"])
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(gt_trans_rotvec_list[:, 0],
         gt_trans_rotvec_list[:, 6], linewidth=1)   # z
plt.plot(est_trans_rotvec_list[:, 0],
         est_trans_rotvec_list[:, 6], linewidth=1)   # z
plt.grid()
plt.legend(["gt_z", "est_z"])
plt.xlabel("time(s)")
plt.ylabel("translation (m)")

if save_flag:
    plt.savefig(est_filename[:-4] + "_RPE{:.2f}.png".format(ARPE*180/3.14))


plt.show()

exit()

# %% utiles
# # Average Relative Rotation Errors(ARRE) in degree


def angAxis2mat(v):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])


def mat2angAxis(m):
    return np.array([m[2, 1], m[0, 2], -m[1, 0]])


r_est = np.array([0.4894532, 0.4894532, 0.1249779])
r_gt = np.array([1, 2, 3])


R_est = lg.expm(angAxis2mat(r_est))
R_gt = lg.expm(angAxis2mat(r_gt))

RRE = lg.norm(mat2angAxis(lg.logm(R_est.transpose()*R_gt)))


# %% utiles
# %% ARPE calculation

t_est = np.array([1, 2, 3])
t_gt = np.array([1, 2, 3])
norm_t1_t2 = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
prod_t1_t2 = np.matmul(t_est, np.transpose(t_gt))

RPE = np.math.acos(prod_t1_t2 / norm_t1_t2)
