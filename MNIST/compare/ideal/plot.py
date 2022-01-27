from cmath import exp
from matplotlib import pyplot as plt
import numpy as np

dir_name = "/home/ssy/BDNN/MNIST/compare/ideal"

f1 = []
f2 = []
f3 = []
for i in range(9000) :
    f1.append(0)
    f2.append(1+3*i/9000)
    f3.append(8-3*exp(-i/2700))

len_1 = len(f1)
len_2 = len(f2)
len_3 = len(f3)


plt.ylim((-0.1, 1.1))
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
ID = plt.scatter(np.arange(len_1), [i/10 for i in f1], s=5, c='r',label='ID')
semi_OOD = plt.scatter(np.arange(len_2)+len_1, [i/10 for i in f2], s=5, c='b',label='semi-OOD')
full_OOD = plt.scatter(np.arange(len_3)+len_1+len_2, [i/10 for i in f3], s=5, c='g',label='full-OOD')

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 16,
}
legend = plt.legend(handles=[ID,semi_OOD,full_OOD],prop=font1,loc='upper left')

# plt.show()
plt.savefig('%s/cluster_visualization.jpg' % dir_name,bbox_inches = 'tight')
