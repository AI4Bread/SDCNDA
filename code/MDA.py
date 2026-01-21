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
'''Read dataset1'''
def read_file1():
    train_id = np.loadtxt("../data/dataset1/datasplit/data1.1/mi_dis_train_id1.txt")
    test_id = np.loadtxt("../data/dataset1/datasplit/data1.1/mi_dis_test_id1.txt")
    neg_id=[]
    low_A = np.loadtxt("../result/dataset1/data1.1/embedding.txt")
    mi_dis = np.loadtxt("../data/dataset1/mi_dis.txt")
    mi_feature = low_A[645: ]
    dis_feature = low_A[240:645]

    return train_id, test_id, low_A, mi_dis, mi_feature, dis_feature,neg_id

def get_feature(A_feature, B_feature, index, adi_matrix):
    input = []
    output = []
    for i in range(index.shape[0]):
        A_i = int(index[i][0])
        B_j = int(index[i][1])
        feature = np.hstack((A_feature[A_i], B_feature[B_j]))
        input.append(feature.tolist())
        label = adi_matrix[[A_i],[B_j]].tolist()
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

'''miRNA-disease'''
train_id, test_id, low_A, mi_dis, mi_feature, dis_feature,negtive_id = read_file1()
#train_id, test_id, low_A, mi_dis, mi_feature, dis_feature, negtive_id = read_file2()
train_input, train_output = get_feature(mi_feature, dis_feature, train_id, mi_dis)
test_input, test_output = get_feature(mi_feature, dis_feature, test_id, mi_dis)


# --------------------------------- Exploring the performance of different classifiers------------------------------
'''XGBoost'''
#flag = 0
flag = 1
if flag:
    xgb = XGBClassifier(n_estimators = 200, eta = 0.1, max_depth = 7)
    xgb.fit(train_input,train_output)
    y_pred = xgb.predict_proba(test_input)[:,1]
    #print(y_pred)
    np.save('../result/dataset1_result/data1.1/MDA/xgb_test_output.npy', test_output)
    np.save('../result/dataset1_result/data1.1/MDA/xgb_y_pred.npy', y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("xgb auc:", auc, "aupr:", aupr, "F1:", f1)

'''MLP'''
#flag = 0
#flag = 1
if flag:
    mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(512,2),activation='relu', learning_rate_init=0.0001,max_iter=1000)
    mlp.fit(train_input,train_output)
    y_pred = mlp.predict_proba(test_input)[:,1]
    np.save('../result/dataset1_result/data1.1/MDA/MLP_test_output.npy', test_output)
    np.save('../result/dataset1_result/data1.1/MDA/MLP_y_pred.npy', y_pred)
    #np.savetxt('../plot/MTP_roc_pr_MDA_dataset1.txt', np.column_stack((test_output, y_pred)), delimiter=',')
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("MLP auc:", auc, "aupr:", aupr, "F1:", f1)

