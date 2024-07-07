# original code is from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
# Modeified by Shiyu Shen
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.cluster import KMeans


def tpr95(dir_name, task='OOD'):
    # calculate the falsepositive error when tpr is 95%
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' %
                           dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' %
                           dir_name, delimiter=',')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start)/200000  # precision:200000

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.96 and tpr >= 0.94:
            fpr += error2
            total += 1
    if total == 0:
        print('corner case')
        fprBase = 1
    else:
        fprBase = fpr/total

    return fprBase


def auroc(dir_name, task='OOD'):
    # calculate the AUROC
    if task == 'OOD':
        f1 = open('%s/Update_Base_ROC_tpr.txt' % dir_name, 'w')
        f2 = open('%s/Update_Base_ROC_fpr.txt' % dir_name, 'w')
        cifar = np.loadtxt('%s/confidence_Base_In.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' %
                           dir_name, delimiter=',')
    elif task == 'mis':
        f1 = open('%s/Update_Base_ROC_tpr_mis.txt' % dir_name, 'w')
        f2 = open('%s/Update_Base_ROC_fpr_mis.txt' % dir_name, 'w')
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' %
                           dir_name, delimiter=',')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start)/200000

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        f1.write("{}\n".format(tpr))
        f2.write("{}\n".format(fpr))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    f1.close()
    f2.close()
    return aurocBase


def auprIn(dir_name, task='OOD'):
    # calculate the AUPR
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' %
                           dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' %
                           dir_name, delimiter=',')

    precisionVec = []
    recallVec = []
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start)/200000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta))  # / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta))  # / np.float(len(Y1))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def auprOut(dir_name, task='OOD'):
    # calculate the AUPR
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' %
                           dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' %
                           dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start)/200000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta))  # / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta))  # / np.float(len(Y1))
        if tp + fp == 0:
            break
        precision = tp / (tp+fp)
        recall = tp/np.float(len(Y1))
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def detection(dir_name, task='OOD'):
    # calculate the minimum detection error
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' %
                           dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' %
                           dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' %
                           dir_name, delimiter=',')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start)/200000

    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

    return errorBase


def confusion_matrix(dir_name):
    # calculate confusion matrix and print
    f1 = np.loadtxt('%s/confidence_Base_In.txt' %
                    dir_name, delimiter=',')
    f2 = np.loadtxt('%s/confidence_Base_semi.txt' %
                    dir_name, delimiter=',')
    f3 = np.loadtxt('%s/confidence_Base_Out.txt' %
                    dir_name, delimiter=',')
    # # reform std to positive uncertainty
    # f1 = list(-f1)
    # f1.sort()
    # len_1 = len(f1)
    # f2 = list(-f2)
    # f2.sort()
    # len_2 = len(f2)
    # f3 = list(-f3)
    # f3.sort()
    # len_3 = len(f3)
    # f = f1+f2+f3
    # f = np.array(f)

    # reform probability to uncertainty in [0,10] 
    f1_ = list(f1)
    f1 = [(1-i) * 10 for i in f1_]
    f1.sort()
    f2_ = list(f2)
    f2 = [(1-i) * 10 for i in f2_]
    f2.sort()
    f3_ = list(f3)
    f3 = [(1-i) * 10 for i in f3_]
    f3.sort()

    # avoid Tianji horse racing, take first 9000 samples
    f1 = f1[0:9000]
    f2 = f2[0:9000]
    f3 = f3[0:9000]

    len_1 = len(f1)
    len_2 = len(f2)
    len_3 = len(f3)
    
    # visualizatoin

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

    ID = plt.scatter(np.arange(len_1), [i/10 for i in f1], s=5, c='r',label='ID')
    semi_OOD = plt.scatter(np.arange(len_2)+len_1, [i/10 for i in f2], s=5, c='b',label='semi-OOD')
    full_OOD = plt.scatter(np.arange(len_3)+len_1+len_2, [i/10 for i in f3], s=5, c='g',label='full-OOD')

    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 16,
    }
    legend = plt.legend(handles=[ID,semi_OOD,full_OOD],prop=font1,loc='upper left')

    # plt.show()
    plt.savefig('%s/cluster_visualization_WOODS.jpg' % dir_name,bbox_inches = 'tight')
    plt.close()

    #distribution

    plt.xlabel("Uncertainty",fontsize=20,fontweight='normal',fontfamily='Times New Roman')
    plt.ylabel("Frequency",fontsize=20,fontweight='normal',fontfamily='Times New Roman')

    nbins = 50
    n1, bins1, rects1 = plt.hist(f1, bins=nbins,range=(0,8), density=True,alpha=0.5,label="ID")
    for r1 in rects1:
        r1.set_height(r1.get_height()*r1.get_width())

    n2, bins2, rects2 = plt.hist(f2, bins=nbins,range=(0,8), density=True,alpha=0.5,label="semi-OOD")
    for r2 in rects2:
        r2.set_height(r2.get_height()*r2.get_width())

    n3, bins3, rects3 = plt.hist(f3, bins=nbins,range=(0,8), density=True,alpha=0.5,label="full-OOD")
    for r3 in rects3:
        r3.set_height(r3.get_height()*r3.get_width())

    # plt.grid(True)
    plt.ylim((0.0, 1.0))
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(prop=font1,loc='upper right')
    plt.savefig('%s/distribution_WOODS.jpg' % dir_name,bbox_inches = 'tight')
    plt.close()

    # k-means
    f = f1+f2+f3
    f = np.array(f)
    clf = KMeans(n_clusters=3)
    clf.fit(f.reshape(-1, 1))
    centers = clf.cluster_centers_
    labels = clf.labels_
    # find the class of cluster centers
    idx = sorted(enumerate(centers[:, 0]), key=lambda x: x[1])

    # construct confusion matrix
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(0, len_1):
        if labels[i] == idx[0][0]:
            confusion[0][0] += 1
        elif labels[i] == idx[1][0]:
            confusion[0][1] += 1
        else:
            confusion[0][2] += 1
    for i in range(len_1, len_1+len_2):
        if labels[i] == idx[0][0]:
            confusion[1][0] += 1
        elif labels[i] == idx[1][0]:
            confusion[1][1] += 1
        else:
            confusion[1][2] += 1
    for i in range(len_1+len_2, len_1+len_2+len_3):
        if labels[i] == idx[0][0]:
            confusion[2][0] += 1
        elif labels[i] == idx[1][0]:
            confusion[2][1] += 1
        else:
            confusion[2][2] += 1

    # print results
    print("Confusion Matrix:")
    tabl = PrettyTable(["\\", "ID", "semi-OOD", "OOD"])
    tabl.add_row(["ID"]+confusion[0])
    tabl.add_row(["semi-OOD"]+confusion[1])
    tabl.add_row(["OOD"]+confusion[2])
    print(tabl)


def metric(dir_name, task):
    print("{}{:>34}".format(task, "Performance of Baseline detector"))
    fprBase = tpr95(dir_name, task)
    print("{:20}{:13.3f}%".format("TNR at TPR 95%:", (1-fprBase)*100))
    aurocBase = auroc(dir_name, task)
    print("{:20}{:13.3f}%".format("AUROC:", aurocBase*100))
    errorBase = detection(dir_name, task)
    print("{:20}{:13.3f}%".format("Detection acc:", (1-errorBase)*100))
    auprinBase = auprIn(dir_name, task)
    print("{:20}{:13.3f}%".format("AUPR In:", auprinBase*100))
    auproutBase = auprOut(dir_name, task)
    print("{:20}{:13.3f}%".format("AUPR Out:", auproutBase*100))