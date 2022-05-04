from cmath import exp
from matplotlib import pyplot as plt
import numpy as np



f1 = []
f2 = []
f3 = []
for i in range(200) :
    f1.append(0)
    f2.append(1+i/200*0.55)
    f3.append(6-4*exp(-i/300))

len_1 = len(f1)
len_2 = len(f2)
len_3 = len(f3)


plt.ylim((-0.05, 1.05))
my_y_ticks = np.arange(0, 1.25, 0.25)
plt.yticks(my_y_ticks)
plt.yticks(fontsize=16)
scale_ls = [len_1/2,len_1+len_2/2,len_1+len_2+len_3/2]
index_ls = ["ID","semi-OOD","full-OOD"]
plt.xticks(scale_ls,index_ls)
plt.xticks(fontsize=12)
plt.xlabel("Data distribution",fontsize=20,fontweight='normal',fontfamily='Times New Roman')
plt.ylabel("Uncertainty",fontsize=20,fontweight='normal',fontfamily='Times New Roman')

# visualizatoin
ID = plt.scatter(np.arange(len_1), [i/10*1.5 for i in f1], s=5, c='r',label='ID')
semi_OOD = plt.scatter(np.arange(len_2)+len_1, [i/10*1.5 for i in f2], s=5, c='b',label='semi-OOD')
full_OOD = plt.scatter(np.arange(len_3)+len_1+len_2, [i/10*1.5 for i in f3], s=5, c='g',label='full-OOD')

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 16,
}
legend = plt.legend(handles=[ID,semi_OOD,full_OOD],prop=font1,loc='upper left')

# plt.show()
plt.savefig('cluster_visualization.jpg',bbox_inches = 'tight')
