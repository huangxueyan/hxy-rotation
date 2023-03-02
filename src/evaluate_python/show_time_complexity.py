from cProfile import label
from turtle import color
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from brokenaxes import brokenaxes
# batch num: poster 5645, dynamic 2378, boxes 6190

# time and RMS% 
EMin_list   = np.array([[60, 2.03], [4.38e5, 0.55], [4168, 0.623], [632, 1.47]])  #  RT-inc, EMIN, AEMIN, INC, 
Ours_d_list = np.array([            [386, 0.723], [504, 0.541], [1536, 0.522], [4288, 0.518]]) #                      iter10_10   0.1k, 1k 10k 30k
Ours_s_list = np.array([[60, 0.95], [335, 1.139], [370, 0.626], [1022, 0.566], [2415, 0.560]]) # RT:sample1k_iter3_10 iter10_10: 0.1k, 1k 10k 30k

CM_list = np.array([[60, 2.76] ]) #, RT-CM, RT-ours


EMin_list[:, 0] = np.log(EMin_list[:,0]/60)
# EMin_list[1,0] = 8.2

Ours_d_list[:, 0] = np.log(Ours_d_list[:,0]/60)
Ours_s_list[:, 0] = np.log(Ours_s_list[:,0]/60)
CM_list[:, 0] = np.log(CM_list[:,0]/60)

plt.figure(figsize=(8, 5), dpi=100)
baxes = brokenaxes(xlims=((-0.3,4.5),(8,9.01)), ylims=((0.3,3.0),), d=0.015)
# plt.xlim([-0.3, 6])
# plt.ylim([0.3, 3.0])
# plt.plot(np.zeros((5,)),np.linspace(0, 3.5, 5),'--', color='#4b4947', alpha=0.5)
baxes.scatter(EMin_list[:,0], EMin_list[:,1], s=50, c='#015598', marker='*', label='EMin' )
baxes.scatter(Ours_d_list[:,0], Ours_d_list[:,1], s=50, c='red', marker='^', label='Ours-d' )
baxes.scatter(Ours_s_list[:,0], Ours_s_list[:,1], s=50, c='#f9bf0f', marker='o', label='Ours-s' )
baxes.scatter(CM_list[:,0], CM_list[:,1], s=50, c=['k'], marker='s', label='CM' )
# plt.scatter()
# plt.scatter()
baxes.legend()    # 显示散点图中的label标签
# baxes.xlabel("Real-time ratio in log scale")
# baxes.ylabel("Average RMS error (%)")
plt.show()
