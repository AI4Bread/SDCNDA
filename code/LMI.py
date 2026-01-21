from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
'''Read dataset1, LDA prediction'''
'''Read dataset1'''
def read_file1():

    train_id = np.loadtxt("../data/dataset1/datasplit/data1.1/mi_lnc_train_id1.txt")
    test_id = np.loadtxt("../data/dataset1/datasplit/data1.1/mi_lnc_test_id1.txt")

    low_A = np.loadtxt("../result/dataset1/data1.1/embedding.txt")
    mi_lnc = np.loadtxt("../data/dataset1/yuguoxian_lnc_mi.txt").T
    neg_id=[]
    mi_feature = low_A[645: ]
    lnc_feature = low_A[ :240]
    return train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, neg_id

def get_feature(A_feature, B_feature, index, adi_matrix):
    input = []
    output = []
    for i in range(index.shape[0]):
        A_i = int(index[i][0])
        B_j = int(index[i][1])
        feature = np.hstack((A_feature[A_i], B_feature[B_j]))
        input.append(feature.tolist())
        label = adi_matrix[[A_i],[B_j]].tolist()
        # print(type(label))
        # label = label.tolist()
        # print(label)
        output.append(label)
    output = np.array(output)
    output = output.ravel()
    return np.array(input), output

def aupr_f1(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = average_precision_score(y_true,y_pred)
    y_pred_label = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    return aupr, f1


'''miRNA-lncRNA'''
train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, negtive_id = read_file1()
#train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, negtive_id = read_file2()
train_input, train_output = get_feature(mi_feature, lnc_feature, train_id, mi_lnc)  # (2328, dim)
test_input, test_output = get_feature(mi_feature, lnc_feature,test_id, mi_lnc)


# --------------------------------- Exploring the performance of different classifiers------------------------------

'''XGBoost'''
#flag = 0
flag = 1
if flag:
    xgb = XGBClassifier(n_estimators = 200, eta = 0.1, max_depth = 10)
    xgb.fit(train_input,train_output)
    y_pred = xgb.predict_proba(test_input)[:,1]
    np.save('../result/dataset1_result/data1.1/LMI/xgb_test_output.npy', test_output)
    np.save('../result/dataset1_result/data1.1/LMI/xgb_y_pred.npy', y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("xgb auc:", auc, "aupr:", aupr, "F1:", f1)



'''MLP'''
#flag = 1
if flag:
    mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(512,2),activation='relu', learning_rate_init=0.0001,max_iter=1000)
    mlp.fit(train_input,train_output)
    y_pred = mlp.predict_proba(test_input)[:,1]
    np.save('../result/dataset1_result/data1.1/LMI/MLP_test_output.npy', test_output)
    np.save('../result/dataset1_result/data1.1/LMI/MLP_y_pred.npy', y_pred)
    #print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("MLP auc:", auc, "aupr:", aupr, "F1:", f1)
