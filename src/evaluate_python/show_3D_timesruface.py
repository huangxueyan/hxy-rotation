import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np 
import pandas as pd 
import torch
from scipy.spatial.transform import Rotation

# 画出event的3D空间与时间图

#%% warp events 
def warp_event(pose, events, ref_t):
    ang_vel_matrix = torch.zeros(3, 3, dtype=torch.float)
    ang_vel_matrix[0, 1] = -pose[2]
    ang_vel_matrix[0, 2] = pose[1]
    ang_vel_matrix[1, 0] = pose[2]
    ang_vel_matrix[1, 2] = -pose[0]
    ang_vel_matrix[2, 0] = -pose[1]
    ang_vel_matrix[2, 1] = pose[0]

    N = events.shape[0]
    fx, fy, cx, cy = 199.092366542, 198.82882047, 132.192071378, 110.712660011
    X = torch.from_numpy((events[:, 1] - cx) / fx)
    Y = torch.from_numpy((events[:, 2] - cy) / fy)
    Z = torch.ones(N, dtype=torch.float)
    point_3d = torch.stack((X, Y, Z), dim=1).float()

    delta_t = torch.from_numpy(events[:, 0] - ref_t).float()
    point_3d_rotated = torch.matmul(point_3d, ang_vel_matrix)  # it should be torch.matmul(angular_vel_matrix, point_3d)
    delta_r = torch.mul(torch.t(delta_t.repeat(3, 1)), point_3d_rotated)
    coordinate_3d = point_3d - delta_r

    warped_x = coordinate_3d[:, 0] * fx / coordinate_3d[:, 2]+ cx
    warped_y = coordinate_3d[:, 1] * fy / coordinate_3d[:, 2]+ cy
    warped_events = torch.stack((warped_x, warped_y), dim=1)
    return warped_events



#%% processing data


START_LINE, COUNT = int(411e4), int(9e3)
filename = "/home/hxy/Documents/rosbag/Mono-unzip/shapes_rotation/"

# ["ts", "x", "y", "p"]
events_pd = pd.read_csv(filename + "events.txt", sep=" ", header=None, skiprows=START_LINE ,nrows=COUNT)
events = events_pd.to_numpy()

# get gt pose 
from scipy.interpolate import interp1d
imu_file = np.loadtxt(filename + "imu.txt")
imu_data = []
for cam_info in imu_file:
    temp = np.array([cam_info[0] - 0.0024, cam_info[4], cam_info[5], cam_info[6]])
    imu_data.append(temp)
imu_data = np.array(imu_data)
fx = interp1d(imu_data[:,0], imu_data[:,1], kind = 'linear')
fy = interp1d(imu_data[:,0], imu_data[:,2], kind = 'linear')
fz = interp1d(imu_data[:,0], imu_data[:,3], kind = 'linear')

ang_axis_t0 = np.array([fx(events[0,0]), fy(events[0,0]), fz(events[0,0])])
ang_axis_t1 = np.array([fx(events[-1,0]), fy(events[-1,0]), fz(events[-1,0])])
ang_axis  = (ang_axis_t1 + ang_axis_t0) / 2

## warp events 
# events = events[events[:, 1]<200]
warped_t0 = warp_event(ang_axis, events, events[0,0]) 
mask = np.bitwise_and(warped_t0[:,0]<220, warped_t0[:,1]<172)
warped_t0 = warped_t0[mask.numpy().astype(np.bool8)]
events = events[mask.numpy().astype(np.bool8),:]


# ang_axis = np.array([0,0,0])
STEP = int(events.shape[0] / 2)
warped_t0 = warp_event(ang_axis, events, events[0,0]) 
warped_t1 = warp_event(ang_axis, events, events[STEP,0]) 
warped_t2 = warp_event(ang_axis, events, events[-1,0]) 


ang_axis = np.array([0,0,0])
unwarp_t0 = warp_event(ang_axis, events, events[0,0]) 


fig = plt.figure(dpi=150)
ax = Axes3D(fig)
 
samples = 3
t = events[::samples, 0]
x = events[::samples, 1]
y = events[::samples, 2]
p = events[::samples, 3]

plot_raw = ax.scatter(x, y, t, c=t, marker='.', s=5, label='', cmap='coolwarm')
# ax.scatter(warped_t0[::5,0], warped_t0[::5,1], zs=events[0, 0],      c='#0000ff', marker='.', s=5, label='begin',)
ax.scatter(warped_t0[::5,0], warped_t0[::5,1], zs=events[0, 0],      c='#0000ff', marker='.', s=5, label='begin',)
# ax.scatter(warped_t1[:STEP*2,0], warped_t1[:STEP*2,1], zs=events[0, 0],   c='#7da0f9', marker='.', s=5, label='middle',)
# ax.scatter(warped_t1[::5,0], warped_t1[::5,1], zs=events[STEP*1, 0], c='#d75444', marker='.', s=5, label='middle',)
ax.scatter(warped_t2[::5,0], warped_t2[::5,1], zs=events[STEP*2-1, 0],     c='#b50927', marker='.', s=5, label='end')

fig.colorbar(plot_raw, shrink=0.5, aspect=5)
# Hide grid lines
ax.grid(False)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.figure(dpi=150)
plt.scatter(warped_t0[:,0], warped_t0[:,1], c=events[:, 0], marker='.', s=5, label='begin',cmap='coolwarm')
plt.xlim([-10,240])
plt.ylim([-10,180])
plt.yticks([0,50,100,150])

plt.figure(dpi=150)

plt.scatter(unwarp_t0[:,0], unwarp_t0[:,1], c= events[:, 0], marker='.', s=5, cmap='coolwarm')
plt.xlim([-10,240])
plt.ylim([-10,180])
plt.yticks([0,50,100,150])



plt.show()

