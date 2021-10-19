
from operator import truediv
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import numpy as np 
import math


## shapes 
# filename1 = "/home/hxt/Desktop/hxy-rotation/data/shapes_gt_theta_velocity_python.txt"
# filename1 = "/home/hxt/Desktop/hxy-rotation/data/poster_gt_theta_velocity_python.txt"
# filename1 = "/home/hxt/Desktop/hxy-rotation/data/boxes_gt_theta_velocity_python.txt"
filename1 = "/home/hxt/Desktop/hxy-rotation/data/dynamic_gt_theta_velocity_python.txt"

# filename2 = "/home/hxt/Desktop/hxy-rotation/data/dynamic_velocity.txt"
filename2 = "/home/hxt/Desktop/hxy-rotation/data/ransac_velocity.txt"

# filename2 = "/home/hxt/Desktop/hxy-rotation/data/shapes_cm_theta_velocity.txt"
# filename2 = "/home/hxt/Desktop/hxy-rotation/data/dynamic_ransac_theta_velocity.txt"


data_num_gt = []
with open(filename1,'r') as f:
# with open(filename,'r') as f:
    data_str = f.readlines()
    # t, x,y,z, qx,qy,qz,qw
    for i in data_str:
        data_num_gt.append(list(map(float,i.split())))
data_num_gt = np.array(data_num_gt)


data_num_est = []
with open(filename2,'r') as f:
# with open(filename,'r') as f:
    data_str = f.readlines()

    for i in data_str:
        data_num_est.append(list(map(float,i.split())))
data_num_est = np.array(data_num_est)

# visual two curves
# plt.figure()
# plt.plot(data_num_gt[:,0], data_num_gt[:,1]/3.14*180,'b')
# plt.plot(data_num_gt[:,0], data_num_gt[:,2]/3.14*180,'r')
# plt.plot(data_num_gt[:,0], data_num_gt[:,3]/3.14*180,'y')
# plt.plot(data_num_est[:,0], data_num_est[:,1]/3.14*180,'b--')
# plt.plot(data_num_est[:,0], data_num_est[:,2]/3.14*180,'r--')
# plt.plot(data_num_est[:,0], data_num_est[:,3]/3.14*180,'y--')

# plt.title("solid gt, dash est")
# plt.legend(["x","y","z"])
# plt.xlabel("time(s)")
# plt.ylabel("velocity (degree/s)")
# plt.show()


# visual rms in time

def get_interpolate(data1, data2, target_t):
    """
    inputs data: [t,x,y,z]
    return interpolated x,y,z
    ----------
    """
    t_1, t_2 = data1[0], data2[0]
    x_1, x_2 = data1[1], data2[1]
    y_1, y_2 = data1[2], data2[2]
    z_1, z_2 = data1[3], data2[3]
    
    ratio = (target_t - t_1) / (t_2 - t_1)
    
    x = (1-ratio) * x_1 +  ratio * x_2
    y = (1-ratio) * y_1 +  ratio * y_2
    z = (1-ratio) * z_1 +  ratio * z_2
    
    return x, y, z

rms_list, inter_gt_list, inter_est_list = [], [], []
rms_tuple = [] 

def search_bias(bias, store = False):
    search_pos = 0
    rms = 0
    for i in data_num_est[3:-5:]:
        t_est = i[0] + bias
        x_est, y_est, z_est = i[1], i[2], i[3]  

        while search_pos < (len(data_num_gt)-1) and t_est > data_num_gt[search_pos,0]:
            search_pos += 1
        
        x_gt, y_gt, z_gt = get_interpolate(data_num_gt[search_pos-1],data_num_gt[search_pos], t_est)
        # x_gt, y_gt, z_gt = data_num_gt[search_pos-1,1], data_num_gt[search_pos-1,2], data_num_gt[search_pos-1,3]

        rms += (x_gt-x_est)**2 + (y_gt-y_est)**2 + (z_gt-z_est)**2
        if store :
            rms_list.append([t_est, abs(x_gt-x_est), abs(y_gt-y_est), abs(z_gt-z_est)])
            inter_gt_list.append([t_est, x_gt, y_gt, z_gt])
            inter_est_list.append([t_est, x_est, y_est, z_est])
    rms_tuple.append((rms, bias))

for bias in np.linspace(-0.05, 0.06, 80):
    search_bias(bias, store = False)
    print("bias {:.3f}, rms {:.2f}".format(rms_tuple[-1][1],rms_tuple[-1][0]))



_temp_idx = np.argmin(np.array(rms_tuple)[:,0])
bias = rms_tuple[_temp_idx][1]
rms = rms_tuple[_temp_idx][0]
print("best bias {}, rms {}".format(bias,math.sqrt(rms/len(data_num_est))/3.14*180))
search_bias(bias, store = True)

rms_list = np.array(rms_list)
inter_gt_list = np.array(inter_gt_list)
inter_est_list = np.array(inter_est_list)

# visualize 
fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=2,
                        height_ratios=[1, 3])
ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])

ax0.plot(rms_list[:,0], rms_list[:,1]/3.14*180,'b')
ax0.plot(rms_list[:,0], rms_list[:,2]/3.14*180,'r')
ax0.plot(rms_list[:,0], rms_list[:,3]/3.14*180,'y')
ax0.set_title("abs error")
ax0.legend(["x","y","z"])
ax0.set_xlabel("time (s)")
ax0.set_ylabel("(degree/s)")


# plot match points 
ax1.plot(inter_gt_list[:,0], inter_gt_list[:,1]/3.14*180,marker = "",markersize=5, c='b',linestyle="--", linewidth=1)
ax1.plot(inter_gt_list[:,0], inter_gt_list[:,2]/3.14*180,marker = "",markersize=5, c='r',linestyle="--", linewidth=1)
ax1.plot(inter_gt_list[:,0], inter_gt_list[:,3]/3.14*180,marker = "",markersize=5, c='y',linestyle="--", linewidth=1)
ax1.plot(inter_est_list[:,0]+bias, inter_est_list[:,1]/3.14*180,marker = "x",markersize=2, c='b',linestyle="-", linewidth=1)
ax1.plot(inter_est_list[:,0]+bias, inter_est_list[:,2]/3.14*180,marker = "x",markersize=2, c='r',linestyle="-", linewidth=1)
ax1.plot(inter_est_list[:,0]+bias, inter_est_list[:,3]/3.14*180,marker = "x",markersize=2, c='y',linestyle="-", linewidth=1)


plt.title("rms {:.2f} degree/s".format(math.sqrt(rms/len(data_num_est))/3.14*180))
plt.legend(["gt_x","gt_y","gt_z","est_x","est_y","est_z"])
plt.xlabel("time (s)")
plt.ylabel("velocity (degree/s)")

plt.show()


