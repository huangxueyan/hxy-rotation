
import matplotlib.pyplot as plt 
import numpy as np 


data_str = []
data_num = []

## shapes 
# filename = "/home/hxt/Desktop/hxy-rotation/data/cm_shape_theta_velocity.txt"
# filename = "/home/hxt/Desktop/hxy-rotation/data/gt_shape_theta_velocity.txt"


## current 
# filename = "/home/hxt/Desktop/hxy-rotation/data/gt_theta_velocity.txt"
filename = "/home/hxt/Desktop/hxy-rotation/data/ransac_velocity.txt"


with open(filename,'r') as f:
# with open(filename,'r') as f:
    data_str = f.readlines()

    for i in data_str:
        data_num.append(list(map(float,i.split())))

data_num = np.array(data_num)

# if "velocity" not in filename:
#     data_num[:, 2][data_num[:, 2]<0] = data_num[:, 2][data_num[:, 2]<0] + 3.14 * 2
#     data_num[:, 2] -= 3.14 / 2

begin_tstamp = np.min(data_num[:,0])

plt.plot(data_num[:,0]+data_num[:,1]/1e9, data_num[:,2]/3.14*180,'b')
plt.plot(data_num[:,0]+data_num[:,1]/1e9, data_num[:,3]/3.14*180,'r')
plt.plot(data_num[:,0]+data_num[:,1]/1e9, data_num[:,4]/3.14*180,'y')


plt.legend(["x","y","z"])
plt.xlabel("time(s)")
if "velocity" in filename:
    plt.ylabel("velocity (degree/s)")
else:
    plt.ylabel("theta (degree)")

plt.show()





