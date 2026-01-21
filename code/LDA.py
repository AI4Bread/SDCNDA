from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import auc
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#import pdb

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

import torch
import torch.nn as nn
import torch.optim as optim

'''Read dataset1, LDA prediction'''
def read_file1():
    train_id = np.loadtxt("../data/dataset1/datasplit/data1.1/lnc_dis_train_id1.txt")
    test_id = np.loadtxt("../data/dataset1/datasplit/data1.1/lnc_dis_test_id1.txt")
    low_A = np.loadtxt("../result/dataset1/data1.1/embedding.txt")
    lnc_dis = np.loadtxt("../data/dataset1/lnc_dis_association.txt")

    negtive_id=[]
    lnc_feature = low_A[:240]
    dis_feature = low_A[240:645]
    print(train_id.shape,test_id.shape)
    return train_id, test_id, low_A, lnc_dis, lnc_feature, dis_feature, negtive_id


def aupr_f1(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = average_precision_score(y_true,y_pred)
    y_pred_label = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    return aupr, f1
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


'''lncRNA-disease'''
train_id, test_id, low_A, lnc_dis, lnc_feature, dis_feature, _ = read_file1()
#train_id, test_id, low_A, lnc_dis, lnc_feature, dis_feature,negtive_id = read_file2()
train_input, train_output = get_feature(lnc_feature, dis_feature, train_id, lnc_dis)
test_input, test_output = get_feature(lnc_feature, dis_feature, test_id, lnc_dis)

#----------------------------Exploring the performance of different classifiers------------------------------

'''XGBoost'''
#flag = 0
flag = 1
if flag:
    xgb = XGBClassifier(n_estimators = 200, eta = 0.1, max_depth = 7)
    xgb.fit(train_input,train_output)
    y_pred = xgb.predict_proba(test_input)[:,1]
    np.save('../result/gamma/dataset1_result/data1.1/LDA/xgb_test_output.npy', test_output)
    np.save('../result/gamma/dataset1_result/data1.1/LDA/xgb_y_pred.npy', y_pred)
    #print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("xgb auc:",auc,"aupr:",aupr,"F1:",f1)


'''MLP'''
#flag = 0
flag = 1
if flag:
    mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(512,2),activation='relu', learning_rate_init=0.0001,max_iter=1500)
    mlp.fit(train_input,train_output)
    y_pred = mlp.predict_proba(test_input)[:,1]
    np.save('../result/dataset1_result/data1.1/LDA/MLP_test_output.npy', test_output)
    np.save('../result/dataset1_result/data1.1/LDA/MLP_y_pred.npy', y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("MLP auc:",auc,"aupr:",aupr,"F1:",f1)
