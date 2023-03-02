"""
模拟心电图
"""
from cv2 import repeat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg as lg
from scipy.spatial.transform import Slerp
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import math
from scipy.spatial.transform import Rotation

gt_filename = "/home/hxy/Documents/rosbag/Zhu-RAL/indoor_flying1/indoor_flying1_left_pose_tum.txt"
# gt_filename = "/home/hxy/Documents/rosbag/Zhu-RAL/outdoor_driving1/outdoor_driving1_left_pose_tum.txt"
est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/indoor_flying1/RT_inner_norm0.01_ransac_size30k_single_warp_1000_timerange(0.8-0.8)_iter3_ceres3_gaussan5_sigma1.0_denoise2.txt"
est_filename_AE = "/home/hxy/Desktop/EventEMin/output/6dof/indoor_flying1/NoWhiten_ApproTsallis_size30k_estimates.txt"
# est_filename = "/home/hxy/Desktop/hxy-rotation/data/6dof_estimation/outdoor_driving1/iter10_10/inner_norm0.01_ransac_size30k_double_warp_10000_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise2.txt"

est_data = np.loadtxt(est_filename)
est_data[:, 0] = (est_data[:, 1] + est_data[:, 2]) / 2
est_data_AE = np.loadtxt(est_filename_AE)
est_data_AE[:, 0] = (est_data_AE[:, 1] + est_data_AE[:, 2]) / 2
gt_pose = np.loadtxt(gt_filename)  # (timestamp px py pz qx qy qz qw)

gt_px_interp = interp1d(gt_pose[:, 0], gt_pose[:, 1], kind='linear') 
gt_py_interp = interp1d(gt_pose[:, 0], gt_pose[:, 2], kind='linear') 
gt_pz_interp = interp1d(gt_pose[:, 0], gt_pose[:, 3], kind='linear') 
gt_quat_interp = Slerp(gt_pose[:, 0], Rotation.from_quat(gt_pose[:, 4:]))


gt_trans_velocity_list, est_trans_rotvec_list = [], []   # for visualize
RPE, ABS_Err_t, ABS_Err_w, RRE = [], [], [], []
for i in range(1, est_data.shape[0]):  # 避免超出插值范围
    t1, t2 = est_data[i, 1], est_data[i, 2]
    delta_t = t2 - t1

    gt_trans_t1 = np.array([gt_px_interp(t1), gt_py_interp(t1), gt_pz_interp(t1)])
    gt_trans_t2 = np.array([gt_px_interp(t2), gt_py_interp(t2), gt_pz_interp(t2)])

    gt_quat_t1 = gt_quat_interp(t1)
    gt_quat_t2 = gt_quat_interp(t2)

    gt_trans = np.matmul(gt_quat_t1.inv().as_matrix(), gt_trans_t2 - gt_trans_t1)
    gt_trans_velocity_list.append(gt_trans/delta_t)

gt_trans_velocity_list = np.array(gt_trans_velocity_list)

# plt.plot(est_data[5:-5,0], gt_trans_velocity_list[:,0])
# plt.show()


# mp.figure('Signal', facecolor='lightgray')
# mp.title('Signal', fontsize=16)
# mp.xlim(0, 30)
# mp.ylim(-3, 3)
# mp.grid(linestyle=':')
# pl = mp.plot([],[], color='dodgerblue', label='Signal')[0]
# pl_bg = mp.plot([],[], color='red', label='2')[0]
# pl_bg.set_data(est_data[5:-5,0], gt_trans_velocity_list[:,0])


# 启动动画
xdata, y1data, y2data, y3data= [], [], [], []
xdata_AE, y1data_AE, y2data_AE, y3data_AE= [], [], [], []
AE_i = 1
for i in range(2,len(est_data)):
    print("processing ", i)
    t, y1, y2, y3 = est_data[i,0], est_data[i,6], est_data[i,7], est_data[i,8]
    t_AE, y1_AE, y2_AE, y3_AE = est_data_AE[AE_i,0], est_data_AE[AE_i,6], est_data_AE[AE_i,7], est_data_AE[AE_i,8]
    
    xdata.append(t)
    y1data.append(y1)
    y2data.append(y2)
    y3data.append(y3)

    # xdata_AE.append(t_AE)
    # y1data_AE.append(y1_AE)
    # y2data_AE.append(y2_AE)
    # y3data_AE.append(y3_AE)

    plt.figure(figsize=(15, 9))  
    plt.cla() 
    plt.subplot(3, 1, 1)
    plt.plot(est_data[1:,0], gt_trans_velocity_list[:,0],linewidth=1,color='darkgray')
    plt.plot(xdata, y1data,linewidth=2)
    plt.plot(est_data_AE[1:,0], est_data_AE[1:,6],linewidth=1)
    # plt.plot(xdata_AE, y1data_AE,linewidth=2)
    plt.xlim(3,67)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.cla() 
    plt.plot(est_data[1:,0], gt_trans_velocity_list[:,1],linewidth=1,color='darkgray')
    plt.plot(xdata, y2data,linewidth=2)
    plt.plot(est_data_AE[1:,0], est_data_AE[1:,7],linewidth=1)
    # plt.plot(xdata_AE, y2data_AE,linewidth=2)
    plt.xlim(3,67)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.cla() 
    plt.plot(est_data[1:,0], gt_trans_velocity_list[:,2],linewidth=1,color='darkgray')
    plt.plot(xdata, y3data,linewidth=2)
    plt.plot(est_data_AE[1:,0], est_data_AE[1:,8],linewidth=1)
    # plt.plot(xdata_AE, y3data_AE,linewidth=2)
    plt.xlim(3,67)
    plt.grid()

    plt.savefig("hello_{:03d}.png".format(i))
    plt.close()
    # plt.show()

# def update(data):
#     t, y1, y2, y3 = data
#     xdata.append(t)
#     y1data.append(y1)
#     y2data.append(y2)
#     y3data.append(y3)
#     print(" receive ", t, " value,", y1)
#     # 重新绘制图像
#     line1.set_data(xdata, y1data)
#     line2.set_data(xdata, y2data)
#     line3.set_data(xdata, y3data)
#     return line
    # 移动坐标轴
    # if x[-1]>5:
    #     mp.xlim(x[-1]-5, x[-1]+5)
 
# time_i = 4
# def generator():
#     global time_i
#     if time_i < 180:
#         time_i += 1
#     else:
#         return
#     x = est_data[time_i,0]
#     yield (x, est_data[time_i,6], est_data[time_i,7], est_data[time_i,8])
#     print("time ", x, "value ", est_data[time_i,6] )
 
# anim = animation.FuncAnimation(fig, update, generator, interval=167)
# anim = animation.FuncAnimation(fig, update, generator, interval=20)

# anim = animation.FuncAnimation(fig, update,  , interval=20)
# writervideo = animation.FFMpegWriter(fps=60) 
# anim.save("animation.mp4", writer=writervideo)
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# plt.show()