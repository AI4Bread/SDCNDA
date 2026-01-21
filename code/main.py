import time
from collections import Counter
from args import get_args
from load_data import load_dataset
from model import Model_our
from model import LogReg
import statistics
import torch_geometric
import torch
import torch as th
import torch.nn as nn
import numpy as np
import warnings
import random
import pdb
import scipy.sparse as sp
import os

seed = 1024
warnings.filterwarnings('ignore')
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

args = get_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = 'ncrna_disease'  # args.dataname
    hid_dim = args.hid_dim  # args.hid_dim (你这里手动写死了)
    out_dim = args.hid_dim  # ✅ 建议：若 num_MLP=0，则把 out_dim 设成 hid_dim；否则确保 num_MLP>0
    n_layers = args.n_layers
    temp = args.temp
    epochs = 500  # args.epochs
    lr1, wd1 = args.lr1, args.wd1
    lr2, wd2 = args.lr2, args.wd2
    device = args.device

    # 🔥 加载数据 - 统一的接口
    if dataname == 'ncrna_disease':
        data = load_dataset(dataname, 0.6, 0.2, 0.2, 5)
        print("✅ Loaded ncRNA-disease dataset with enhanced edge information")

        # 数据里提供了同层/跨层边
        edge_index_same = data.edge_index_same  # [2, E_same] (CPU)
        edge_index_cross = data.edge_index_cross  # [2, E_cross] (CPU)

        # （可选但推荐）把两类边都变成无向
        from torch_geometric.utils import to_undirected

        edge_index_same = to_undirected(edge_index_same)
        edge_index_cross = to_undirected(edge_index_cross)

    else:
        data = load_dataset(dataname, 0.6, 0.2, 0.2, 5)
        print(f"✅ Loaded standard dataset: {dataname}")
        # 标准数据没有分同/跨层，就先都用原始边
        edge_index_same = data.edge_index
        edge_index_cross = data.edge_index

    import dgl
    from torch_geometric.utils import to_undirected
    import time
    import torch

    # 处理原始图结构（用于 ChebNetII 的谱滤波）
    data.edge_index = to_undirected(data.edge_index)

    # 用 DGL 构建图（仅为了你现有的 conv 调用里使用 graph.edge_index）
    g = dgl.graph((data.edge_index[0], data.edge_index[1]))
    # 加/去自环（DGL 是 out-of-place API，记得接返回值）
    g = g.remove_self_loop().add_self_loop()

    # 设备与特征
    feat = data.x.to(device)  # [N, F]
    labels = data.y  # （目前不用，不搬也行）
    g = g.to(device)

    # 给 graph 挂一个 PyG 风格的 edge_index，供 ChebNetII 使用
    u, v = g.edges()
    # DGL 这里通常返回在同一 device 上的张量，但稳妥起见显式转型
    graph = g
    graph.edge_index = torch.stack([u, v]).to(device=device, dtype=torch.long)

    num_class = (torch.max(labels) + 1).item()
    print("Nodes:", feat.shape[0])
    print("Features:", feat.shape[1])
    print("Classes:", num_class)

    in_dim = feat.shape[1]

    # ✅ 重要：如果 num_MLP == 0，请把 out_dim 设为 hid_dim，避免维度不匹配
    if (args.num_MLP == 0) and (out_dim != hid_dim):
        print(f"[Note] num_MLP=0 且 out_dim({out_dim})!=hid_dim({hid_dim})，将 out_dim 重置为 hid_dim 以匹配相似度维度。")
        out_dim = hid_dim

    # 初始化模型（直接把 edge_index_same/cross 传进去就好）
    model = Model_our(
        in_dim, hid_dim, out_dim, n_layers, temp,
        args.use_mlp, args.num_MLP, args.gamma, args.k,
        edge_index_same=edge_index_same,  # 先在 CPU，forward 里会自动转 device
        edge_index_cross=edge_index_cross
    ).to(device)

    # ❌ 不再需要：model.set_edge_indices(...)

    model.batch_size = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)

    print("=== Training ===")
    start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(graph, feat)  # graph.edge_index 会被 ChebNetII 使用
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')
    end = time.time()
    print(f"Training time: {end - start:.2f}s")


    # GPU 显存统计（可选）
    def to_MB(byte):
        return byte / 1024.0 / 1024.0


    if torch.cuda.is_available() and 'cuda' in str(device):
        print(f"Max GPU memory: {to_MB(torch.cuda.max_memory_allocated(device)):.2f} MB")

    print("=== Evaluation ===")
    # 再次确保图有自环（按你流程）
    graph = graph  # 已经在上面处理过；如果要严格一致可：g = g.remove_self_loop().add_self_loop()

    # 获取节点嵌入（Z_f）
    embeds = model.get_embedding(graph, feat)  # [N, out_dim]
    np.savetxt('../result/dataset1/data1.1/embdding.txt', embeds.cpu().detach().numpy())
    print("Embeddings shape:", embeds.shape)
    results = []
