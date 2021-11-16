
from operator import truediv
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import numpy as np 
from scipy.interpolate import interp1d
import math


## shapes 
filename1 = "/home/hxy/Desktop/hxy-rotation/data/boxes_rotation/imu.txt"
# filename1 = "/home/hxy/Desktop/hxy-rotation/data/poster_gt_theta_velocity_python.txt"
# filename1 = "/home/hxy/Desktop/hxy-rotation/data/boxes_gt_theta_velocity_python.txt"
# filename1 = "/home/hxy/Desktop/hxy-rotation/data/dynamic_gt_theta_velocity_python.txt"

# filename2 = "/home/hxy/Desktop/hxy-rotation/data/dynamic_velocity.txt"
# filename2 = "/home/hxy/Desktop/hxy-rotation/data/ransac_velocity.txt"

filename2 = "/home/hxy/Desktop/hxy-rotation/data/boxes_rotation/ransac_doublewarp_init0.txt"
# filename2 = "/home/hxy/Desktop/Event-ST-PPP/dataset/boxes_rotation/ppp_30k_output.txt"


imu_vel = []
imu_data = np.loadtxt(filename1)
for imu_info in imu_data:
    temp = np.array(
        [imu_info[0] - 0.0024, imu_info[4], imu_info[5], imu_info[6]])
    imu_vel.append(temp)

data_gt_raw = np.array(imu_vel)


def get_interpolate(gt, est):
    """
    inputs data: raw gt [t,vx,vy,vz]
    return interpolated gt
    ----------
    """

    dst = est[:,0] ## time 
    for i in range(gt.shape[1]-1):
        t = gt[:,0]
        x = gt[:,i+1]
        f = interp1d(t, x, kind = 'linear')
        new_x = f(est[:,0])
        dst = np.c_[dst, new_x]
    return np.array(dst)

data_est_raw = np.loadtxt(filename2) 
data_est_t = (data_est_raw[:, 1] + data_est_raw[:,2])/2
data_est = np.c_[data_est_t, data_est_raw[:,-3:]]
data_gt_intered = get_interpolate(data_gt_raw, data_est)

mean_err_x = np.abs(data_est[:,1] - data_gt_intered[:,1])
mean_err_y = np.abs(data_est[:,2] - data_gt_intered[:,2])
mean_err_z = np.abs(data_est[:,3] - data_gt_intered[:,3])

def get_rms(gt, est):
    vec_gt = np.reshape(gt[:, 1:], (-1,))
    vec_est = np.reshape(est[:, 1:], (-1,))
    return np.sqrt(np.mean(np.power(vec_gt-vec_est, 2)))

rms = get_rms(data_gt_intered, data_est)

# visualize 
fig = plt.figure(figsize=(20,10)) # cols, rows
# spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 3])
# ax0 = fig.add_subplot(spec[0])
# ax1 = fig.add_subplot(spec[1])
ax0 = fig.add_subplot()

ax0.plot(data_est[:,0], mean_err_x/3.14*180+800,'b')
ax0.plot(data_est[:,0], mean_err_y/3.14*180+800,'r')
ax0.plot(data_est[:,0], mean_err_z/3.14*180+800,'y')
ax0.set_title("abs err")
ax0.legend(["x","y","z"])
ax0.set_xlabel("time (s)")
ax0.set_ylabel("(degree/s)")


# plot match points marker reference 
# ax0.plot(data_gt_raw[:,0], data_gt_raw[:,1]/3.14*180,marker = "1",markersize=5, c='b',linestyle="--", linewidth=1)
# ax0.plot(data_gt_raw[:,0], data_gt_raw[:,2]/3.14*180,marker = "1",markersize=5, c='r',linestyle="--", linewidth=1)
# ax0.plot(data_gt_raw[:,0], data_gt_raw[:,3]/3.14*180,marker = "1",markersize=5, c='y',linestyle="--", linewidth=1)
ax0.plot(data_gt_intered[:,0], data_gt_intered[:,1]/3.14*180,marker = ".",markersize=7, c='b',linestyle="--", linewidth=1)
ax0.plot(data_gt_intered[:,0], data_gt_intered[:,2]/3.14*180,marker = ".",markersize=7, c='r',linestyle="--", linewidth=1)
ax0.plot(data_gt_intered[:,0], data_gt_intered[:,3]/3.14*180,marker = ".",markersize=7, c='y',linestyle="--", linewidth=1)
ax0.plot(data_est[:,0], data_est[:,1]/3.14*180,marker = "+",markersize=7, c='b',linestyle="-", linewidth=1)
ax0.plot(data_est[:,0], data_est[:,2]/3.14*180,marker = "+",markersize=7, c='r',linestyle="-", linewidth=1)
ax0.plot(data_est[:,0], data_est[:,3]/3.14*180,marker = "+",markersize=7, c='y',linestyle="-", linewidth=1)



plt.title("rms {:.2f} degree/s".format(rms/3.14*180))
plt.legend(["gt_x","gt_y","gt_z","est_x","est_y","est_z"])
plt.xlabel("time (s)")
plt.ylabel("velocity (degree/s)")

plt.show()


