from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt 

folder = "/home/hxy/Desktop/ECCV22-all/hxy-rotation/data/rotation_estimation/dynamic_rotation/"
gt_filename = "second_order_size30k_double_warp_1500_timerange(0.8-0.8)_iter10_ceres10_gaussan5_sigma1.0_denoise4_defaultval1.0.txt"
imu_data = np.loadtxt(folder + gt_filename)

# pre_value = np.array([0, 0, 0], dtype=np.float)
# norm_list = []
# max_norm = 0
# for cam_info in imu_data:
#     if np.linalg.norm(pre_value) == 0:
#         pre_value = np.array([cam_info[3], cam_info[4], cam_info[5]])
#         continue
#     cur_value = np.array([cam_info[3], cam_info[4], cam_info[5]])
#     norm_list.append(np.linalg.norm(cur_value - pre_value)) 
#     pre_value = cur_value

# norm_list = np.array(norm_list)
# max_norm = np.max(norm_list)
# max_idx = np.where(norm_list == max_norm)[0].item()

# print("avg {:.2f}, max {:.2f}, idx {}".format(np.mean(norm_list), max_norm, max_idx))


t_list, vnorm_list, size_list = [], [], []
max_norm = 0
for cam_info in imu_data:
    cur_value = np.array([cam_info[3], cam_info[4], cam_info[5]])
    vnorm_list.append(np.linalg.norm(cur_value))
    t_list.append(cam_info[2] - cam_info[1])
    # size_list.append(cam_info[6])
    size_list.append(3)

vnorm_list = np.array(vnorm_list)
t_list = np.array(t_list)
size_list = np.array(size_list)

size_t_list = size_list / t_list


plt.plot(size_t_list, vnorm_list, 'o')
plt.show()

pass