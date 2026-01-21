from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np

# 加载 lnc_dis 矩阵
lnc_dis = np.loadtxt("../data/dataset1/lnc_dis_association.txt")
#lnc_dis = lnc_dis.T

if np.any(lnc_dis == 1):
    print("mi_dis contains 1s")
else:
    print("mi_dis does not contain 1s")
# 初始化正样本和负样本的列表
positive_samples = []
negative_samples = []

# 遍历 lnc_dis 矩阵，根据值分类为正样本和负样本
lnc_dis_shape = lnc_dis.shape
for lnc_id in range(lnc_dis_shape[0]):
    for dis_id in range(lnc_dis_shape[1]):
        if lnc_dis[lnc_id, dis_id] == 1:
            positive_samples.append((lnc_id, dis_id))
        else:
            negative_samples.append((lnc_id, dis_id))

# 将正样本和负样本转换为 numpy 数组
positive_samples = np.array(positive_samples)
negative_samples = np.array(negative_samples)

# 打乱正样本
np.random.shuffle(positive_samples)

# 将正样本分为5个子集
positive_subsets = np.array_split(positive_samples, 5)
#print(positive_subsets)

# 为每个子集生成训练集和测试集，并保存到文件
for i in range(5):
    # 当前子集作为测试集，剩余子集作为训练集
    test_positive = positive_subsets[i]
    train_positive = np.vstack([positive_subsets[j] for j in range(5) if j != i])

    # 从负样本中随机选择与训练正样本等大小的负样本用于训练
    train_negative_indices = np.random.choice(len(negative_samples), len(train_positive), replace=False)
    train_negative = negative_samples[train_negative_indices]

    '''
    # 剩余的负样本用于测试
    remaining_negative_indices = np.array(
        [index for index in range(len(negative_samples)) if index not in train_negative_indices])
    test_negative = negative_samples[remaining_negative_indices]
    '''

    #'''
    # 从剩余的负样本中随机选择与测试正样本等大小的负样本用于测试
    remaining_negative_indices = np.array(
        [index for index in range(len(negative_samples)) if index not in train_negative_indices])
    remaining_negative_samples = negative_samples[remaining_negative_indices]
    test_negative_indices = np.random.choice(len(remaining_negative_samples), len(test_positive)*1, replace=False)
    test_negative = remaining_negative_samples[test_negative_indices]
    #'''

    # 合并正负样本形成训练集和测试集
    #print("train_positive shape:", train_positive.shape)
    #print("train_negative shape:", train_negative.shape)
    train_data = np.vstack((train_positive, train_negative))
    test_data = np.vstack((test_positive, test_negative))

    # 保存训练集和测试集到文件
    print("train_positive shape:", train_positive.shape)
    print("train_negative shape:", train_negative.shape)
    print("test_positive shape:", test_positive.shape)
    print("test_negative shape:", test_negative.shape)
    np.savetxt(f"../data/dataset1/datasplit/data1.1/lnc_dis_train_id{i + 1}.txt", train_data, fmt='%d')
    np.savetxt(f"../data/dataset1/datasplit/data1.1/lnc_dis_test_id{i + 1}.txt", test_data, fmt='%d')