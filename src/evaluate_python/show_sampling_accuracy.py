from cProfile import label
from turtle import color
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 

# batch num: poster 5645, dynamic 2378, boxes 6190

# time and RMS% 
Ours_d_list = np.array([[0.1, 0.717], [0.3, 0.590], [1, 0.543], [3, 0.5258], [10, 0.519], [20, 0.51748], [30, 0.5172]]) #  0.1k 0.3k 1k 3k 10k 20k 30k
Ours_s_list = np.array([[0.1, 1.296], [0.3, 0.763], [1, 0.626], [3, 0.5803], [10, 0.566], [20, 0.56244], [30, 0.5615]])  #  0.1k 0.3k 1k 3k 10k 20k 30k


plt.figure(figsize=(4, 4), dpi=150)
# plt.xlim([-0.9, 5])
plt.ylim([0.3, 3.0])
# plt.plot(np.zeros((5,)),np.linspace(0, 3.5, 5),'--', color='#4b4947', alpha=0.5)
# plt.scatter(EMin_list[:,0], EMin_list[:,1], s=50, c='#015598', marker='*', label='EMin' )
plt.scatter(Ours_d_list[:,0], Ours_d_list[:,1], s=50, c='red', marker='^', label='Ours-d' )
plt.scatter(Ours_s_list[:,0], Ours_s_list[:,1], s=50, c='#f9bf0f', marker='o', label='Ours-s' )
# plt.scatter(CM_list[:,0], CM_list[:,1], s=50, c=['k'], marker='s', label='CM' )
# plt.scatter()
# plt.scatter()
plt.legend()    # 显示散点图中的label标签
plt.xlabel("Sampling number ")
plt.ylabel("Average RMS error (%)")
plt.show()
