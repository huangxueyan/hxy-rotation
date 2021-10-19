
import matplotlib.pyplot as plt 
import numpy as np 
import math

data_num1 = []
data_num2 = []

## shapes 
# filename1 = "/home/hxt/Desktop/hxy-rotation/data/poster_gt_theta_velocity_python.txt"
filename1 = "/home/hxt/Desktop/hxy-rotation/data/shapes_gt_theta_velocity_python.txt"
# filename1 = "/home/hxt/Desktop/hxy-rotation/data/boxes_gt_theta_velocity_python.txt"
filename2 = "/home/hxt/Desktop/hxy-rotation/data/ransac_velocity.txt"
# filename2 = "/home/hxt/Desktop/hxy-rotation/data/poster_cm10000_theta_velocity.txt"
# filename2 = "/home/hxt/Desktop/hxy-rotation/data/poster_cm_theta_velocity.txt"




with open(filename1,'r') as f:
# with open(filename,'r') as f:
    data_str = f.readlines()
    # t, x,y,z, qx,qy,qz,qw
    for i in data_str:
        data_num1.append(list(map(float,i.split())))

data_num1 = np.array(data_num1)


with open(filename2,'r') as f:
# with open(filename,'r') as f:
    data_str = f.readlines()

    for i in data_str:
        data_num2.append(list(map(float,i.split())))
data_num2 = np.array(data_num2)


plt.plot(data_num1[:,0], data_num1[:,1]/3.14*180,marker = "",markersize=4, c='b',linestyle="-")
plt.plot(data_num1[:,0], data_num1[:,2]/3.14*180,marker = "",markersize=4, c='r',linestyle="-")
plt.plot(data_num1[:,0], data_num1[:,3]/3.14*180,marker = "",markersize=4, c='y',linestyle="-")


plt.plot(data_num2[:,0], data_num2[:,1]/3.14*180,marker = "x",markersize=4, c='b',linestyle="")
plt.plot(data_num2[:,0], data_num2[:,2]/3.14*180,marker = "x",markersize=4, c='r',linestyle="")
plt.plot(data_num2[:,0], data_num2[:,3]/3.14*180,marker = "x",markersize=4, c='y',linestyle="")

plt.title("dash gt, solid est")
plt.legend(["x","y","z"])
plt.xlabel("time(s)")
if "velocity" in filename1:
    plt.ylabel("velocity (degree/s)")
else:
    plt.ylabel("theta (degree)")

plt.show()





