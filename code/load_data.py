from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import torch_geometric.transforms as T
import scipy.io
import csv
import pandas as pd
import json
from os import path
import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from model import *

def rand_train_test_idx(label, train_prop, valid_prop, test_prop, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:train_num + valid_num + test_num]

    train_idx = train_indices
    valid_idx = val_indices
    test_idx = test_indices

    return {'train': train_idx.numpy(), 'valid': valid_idx.numpy(), 'test': test_idx.numpy()}

def index_to_mask(splits_lst, num_nodes):
    mask_len = len(splits_lst)
    train_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)

    for i in range(mask_len):
        train_mask[i][splits_lst[i]['train']] = True
        val_mask[i][splits_lst[i]['valid']] = True
        test_mask[i][splits_lst[i]['test']] = True

    return train_mask.T, val_mask.T, test_mask.T

from typing import Optional, Callable
import os.path as osp
import torch
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data

# import gdown
def load_dataset(dataname, train_prop, valid_prop, test_prop, num_masks):
    #666
    assert dataname in ('ncrna_disease'), 'Invalid dataset'
    if dataname == 'ncrna_disease':
        data = load_ncrna_disease_dataset('dataset1/data/data1.1', 0.6, 0.2, 0.2, 5)
    else:
        data = load_dataset(dataname, 0.6, 0.2, 0.2, 5)
    return data

import torch
import numpy as np
from torch_geometric.data import Data
from load_data import rand_train_test_idx, index_to_mask

def load_ncrna_disease_dataset(dataset_name='data1.1', train_prop=0.6, valid_prop=0.2, test_prop=0.2, num_masks=5):

    import os
    
    data_dir = f'../data/{dataset_name}/'
    
    print(f"Loading {dataset_name} dataset from {data_dir}")
    
    # 加载邻接矩阵 A
    A_path = os.path.join(data_dir, 'matrix_A.npy')
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"matrix_A.npy not found at {A_path}")
    matrix_A = np.load(A_path)
    print(f"✅ Loaded matrix_A: {matrix_A.shape}")
    
    # 加载标准化后的特征矩阵 (用作节点特征)
    feature_path = os.path.join(data_dir, 'feature.pt')
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"feature.pt not found at {feature_path}")
    features = torch.load(feature_path)
    print(f"✅ Loaded features: {features.shape}")
    
    # 加载节点标签
    label_path = os.path.join(data_dir, 'label.pt')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"label.pt not found at {label_path}")
    labels = torch.load(label_path)  
    print(f"✅ Loaded labels: {labels.shape}")
    
    # 🔥 直接从 labels 生成 type_vec（labels 就是节点类型！）
    type_vec = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels.copy()
    print(f"✅ Generated type_vec from labels: {type_vec.shape}")
    print(f"📊 Node types: lncRNA={np.sum(type_vec==0)}, disease={np.sum(type_vec==1)}, miRNA={np.sum(type_vec==2)}")
    
    # 从 matrix_A 构造边信息
    def construct_edge_indices_from_A(A, type_vec):
        """从邻接矩阵A和type_vec构造同层边和跨层边"""
        # 获取非零边
        row, col = np.where(A > 0)
        edge_index_full = torch.tensor([row, col], dtype=torch.long)
        
        if len(row) == 0:
            print("⚠️ Warning: No edges found in matrix_A")
            # 返回空的边索引
            edge_index_same = torch.zeros((2, 0), dtype=torch.long)
            edge_index_cross = torch.zeros((2, 0), dtype=torch.long)
            return edge_index_same, edge_index_cross, edge_index_full
        
        type_vec_tensor = torch.tensor(type_vec, dtype=torch.long)
        
        # 获取每条边两端节点的类型
        row_types = type_vec_tensor[edge_index_full[0]]
        col_types = type_vec_tensor[edge_index_full[1]]
        
        # 同层边：两端节点类型相同 (语义视角)
        same_type_mask = (row_types == col_types)
        # 跨层边：两端节点类型不同 (结构视角)
        cross_type_mask = ~same_type_mask
        
        edge_index_same = edge_index_full[:, same_type_mask]   # 同层边（语义）
        edge_index_cross = edge_index_full[:, cross_type_mask] # 跨层边（结构）
        
        print(f"📊 Edge statistics:")
        print(f"   - Total edges: {edge_index_full.shape[1]}")
        print(f"   - Same-type edges (semantic): {edge_index_same.shape[1]}")
        print(f"   - Cross-type edges (structure): {edge_index_cross.shape[1]}")
        
        return edge_index_same, edge_index_cross, edge_index_full
    
    # 构造边索引
    edge_index_same, edge_index_cross, edge_index_full = construct_edge_indices_from_A(matrix_A, type_vec)
    
    # 生成训练/验证/测试划分
    splits_lst = [rand_train_test_idx(labels, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                  for _ in range(num_masks)]
    train_mask, val_mask, test_mask = index_to_mask(splits_lst, len(labels))
    
    # 创建 Data 对象
    data = Data(
        x=features,                    # 节点特征 (归一化后的A矩阵)
        edge_index=edge_index_full,    # 完整的边索引
        y=labels,                      # 节点标签
        train_mask=train_mask,
        val_mask=val_mask, 
        test_mask=test_mask,
        num_nodes=len(labels)
    )
    
    # 添加额外的边信息 (供 S3GCL 使用)
    data.edge_index_same = edge_index_same     # 同层边 (语义视角)
    data.edge_index_cross = edge_index_cross   # 跨层边 (结构视角)
    data.type_vec = torch.tensor(type_vec, dtype=torch.long)  # 节点类型 (就是labels的副本)
    data.matrix_A = torch.tensor(matrix_A, dtype=torch.float)  # 原始邻接矩阵
    
    print(f"🎉 Successfully loaded {dataset_name}:")
    print(f"   - Nodes: {data.num_nodes}")
    print(f"   - Features: {data.x.shape[1]}")
    print(f"   - Classes: {len(torch.unique(labels))}")
    print(f"   - Node types: lncRNA={torch.sum(data.type_vec==0)}, disease={torch.sum(data.type_vec==1)}, miRNA={torch.sum(data.type_vec==2)}")
    
    return data