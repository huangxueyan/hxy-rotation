# using linear interpolation to get rmse errors 

from scipy.spatial.transform import Rotation 
import numpy as np 
import math 
import matplotlib.pyplot as plt

def toEulerAngles(qx, qy, qz, qw):
    """
    inputs: qx,qy,qz,qw
    ----------
    """
    
    angles_x, angles_y, angles_z = 0, 0, 0  
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    angles_x = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        angles_y = math.pi/2 if sinp>0 else -math.pi/2  # use 90 degrees if out of range
    else:
        angles_y = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    angles_z = math.atan2(siny_cosp, cosy_cosp)

    return [angles_x, angles_y, angles_z]


# filename = "/home/hxt/Documents/rosbag/Mono-unzip/shapes_rotation/groundtruth.txt"
# filename = "/home/hxt/Documents/rosbag/Mono-unzip/poster_rotation/groundtruth.txt"
# filename = "/home/hxt/Documents/rosbag/Mono-unzip/boxes_rotation/groundtruth.txt"
filename = "/home/hxt/Documents/rosbag/Mono-unzip/dynamic_rotation/groundtruth.txt"


data_num = [] 
init_rotation, init_time = [], []
with open(filename,'r') as f:
    data_str = f.readlines()
    # t, x,y,z, qx,qy,qz,qw

    for index, i in enumerate(data_str):
        elements = list(map(float,i.split()))
        if index == 0:
            init_rotation = Rotation.from_quat([elements[7], elements[4],elements[5],elements[6]])
            init_time = elements[0]
            data_num.append( [init_time, 0,0,0])
            continue

        t = elements[0] 
        rotation = Rotation.from_quat([elements[7], elements[4],elements[5],elements[6]])
        
        relative_r = rotation * init_rotation.inv()

        qx, qy, qz, qw = relative_r.as_quat()
        euler = toEulerAngles(qx, qy, qz,qw)

        # euler = toEulerAngles(elements[4],elements[5],elements[6],elements[7])
        data_num.append( [t] + [euler[2], -euler[1], -euler[0]])   # gt中 x_gt=z, y_gt=-y, z=-x

data_num = np.array(data_num)

# data_num[data_num[:,3]>0,3] -= 2*math.pi

# abs position 
plt.figure()
plt.plot(data_num[:,0], data_num[:,1]*180/3.14,'b')
plt.plot(data_num[:,0], data_num[:,2]*180/3.14,'r')
plt.plot(data_num[:,0], data_num[:,3]*180/3.14,'y')
plt.legend(["x","y","z"])
plt.xlabel("time(s)")
plt.ylabel("theta (degree)")
plt.show()

# velocity 

# time angle_x, angle_y, angle_z
data_num2 = []
# filename = "/home/hxt/Desktop/hxy-rotation/data/shapes_gt_theta_velocity_python.txt"
# filename = "/home/hxt/Desktop/hxy-rotation/data/poster_gt_theta_velocity_python.txt"
# filename = "/home/hxt/Desktop/hxy-rotation/data/boxes_gt_theta_velocity_python.txt"
filename = "/home/hxt/Desktop/hxy-rotation/data/dynamic_gt_theta_velocity_python.txt"

interval = 5
with open(filename,'w') as f:
    for i in range(interval, len(data_num[:,0])-interval-1): 
        time_stamp = (data_num[i+interval,0] + data_num[i-interval,0]) / 2
        delta_t = data_num[i+interval,0] - data_num[i-interval,0]
        velocity_x = (data_num[i+interval,1] - data_num[i-interval,1]) / delta_t
        velocity_y = (data_num[i+interval,2] - data_num[i-interval,2]) / delta_t
        velocity_z = (data_num[i+interval,3] - data_num[i-interval,3]) / delta_t
        data_num2.append([time_stamp, velocity_x, velocity_y, velocity_z]) 
        f.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(time_stamp, velocity_x, velocity_y, velocity_z))

data_num2 = np.array(data_num2)


plt.figure()
plt.plot(data_num2[:,0], data_num2[:,1]*180/3.14,'b')
plt.plot(data_num2[:,0], data_num2[:,2]*180/3.14,'r')  
plt.plot(data_num2[:,0], data_num2[:,3]*180/3.14,'y')
plt.legend(["x","y","z"])
plt.xlabel("time(s)")
plt.ylabel("velocity (degree/s)")
plt.show()







