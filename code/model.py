import dgl.sampling
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
import copy
import random
from sklearn.neighbors import kneighbors_graph
from scipy import sparse

sigma = 1e-6
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)

torch.cuda.manual_seed_all(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_index_to_mask(edge_index, num_nodes, device, fill_diag: bool = True):
    """将 edge_index 转为稠密 0/1 掩码 (float) [N, N]，并放到 device 上"""
    M = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index is None or edge_index.numel() == 0:
        if fill_diag:
            M.fill_diagonal_(1.0)
        return M
    # 关键：把索引搬到同一 device，且用 long
    edge_index = edge_index.to(device=device, dtype=torch.long)
    r, c = edge_index[0], edge_index[1]
    M[r, c] = 1.0
    if fill_diag:
        M.fill_diagonal_(1.0)
    return M

from torch.nn import Linear

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_ln=False):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.use_ln = use_ln
        self.lns = nn.ModuleList()

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            for i in range(n_layers - 1):
                self.lns.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            if not self.use_ln:
                x = F.relu(self.convs[i](graph, x))
            else:
                x = F.relu(self.lns[i](self.convs[i](graph, x)))

        x = self.convs[-1](graph, x)

        return x


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(MLP_generator, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        neighbor_embedding = self.linears[self.num_layers - 1](h)
        return neighbor_embedding


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.LayerNorm([h_dim * 2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]


class Gate(nn.Module):
    def __init__(self, hidden) -> None:
        super().__init__()

        self.cs = ConditionalSelection(hidden, hidden)
        self.context = 0
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, context=None, tau=1, hard=False):

        gate_in = rep

        if context != None:
            context = F.normalize(context, p=2, dim=1)
            self.context = torch.tile(context, (rep.shape[0], 1))

        if self.context != None:
            gate_in = rep * self.context

        pm, gm = self.cs(gate_in, tau=tau, hard=hard)

        rep_p = rep * pm
        rep_g = rep * gm
        return rep_p, rep_g

    def set_context(self, head):
        headw_p = head.weight.data.clone()
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)


def extract_batch_mask(edge_index, batch_nodes, num_nodes):
    """从edge_index中提取batch对应的子图掩码"""
    # 创建节点映射
    node_map = {node.item(): i for i, node in enumerate(batch_nodes)}

    # 筛选边
    r, c = edge_index
    batch_set = set(batch_nodes.tolist())
    mask = torch.tensor([r[i].item() in batch_set and c[i].item() in batch_set
                         for i in range(len(r))], dtype=torch.bool)

    if mask.sum() == 0:  # 如果没有边，返回单位矩阵
        sub_mask = torch.eye(len(batch_nodes), device=edge_index.device)
        return sub_mask

    sub_r = r[mask]
    sub_c = c[mask]

    # 重新映射节点索引
    remapped_r = torch.tensor([node_map[node.item()] for node in sub_r], device=edge_index.device)
    remapped_c = torch.tensor([node_map[node.item()] for node in sub_c], device=edge_index.device)

    # 创建子图掩码
    sub_mask = torch.zeros((len(batch_nodes), len(batch_nodes)), device=edge_index.device)
    sub_mask[remapped_r, remapped_c] = 1
    sub_mask.fill_diagonal_(1)  # 添加自环

    return sub_mask


class Model_our(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp, use_mlp=False, num_MLP=1,
                 gamma=0.5, k=6, edge_index_same=None, edge_index_cross=None, batch_size: int = 0):
        super(Model_our, self).__init__()

        # 统一：始终提供 full-pass 的 MLP 编码器（Z_f）
        self.encoder_target = MLP(in_dim, hid_dim, out_dim, use_bn=True)

        # 原来的 encoder 保留（现在 forward 没用到它，不影响）
        if use_mlp:
            self.encoder = MLP(in_dim, hid_dim, out_dim, use_bn=True)
        else:
            self.encoder = GCN(in_dim, hid_dim, out_dim, num_layers)

        # projector：把 Z_f(out_dim) 投到 hid_dim，好与 Zh/Zl 对齐
        self.num_MLP = num_MLP
        if num_MLP and num_MLP > 0:
            self.projector = MLP_generator(out_dim, hid_dim, num_MLP)


        self.temp = temp
        self.out_dim = out_dim
        self.gamma = gamma

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 存储边信息（同层/跨层）
        self.edge_index_same = edge_index_same
        self.edge_index_cross = edge_index_cross

        self.batch_size = int(batch_size) if batch_size is not None else 0

    def get_embedding(self, graph, feat):
        # 只导出 MLP 的全通表示 Z_f
        trans_feature = self.encoder_target(graph, feat)
        return trans_feature.detach()

    def infonce(self, anchor, sample, pos_mask, neg_mask, tau, eps: float = 1e-12):
        # 行级多正样本 InfoNCE
        a = F.normalize(anchor, p=2, dim=1)
        b = F.normalize(sample, p=2, dim=1)
        sim = (a @ b.t()) / tau
        pos = pos_mask.float() if pos_mask.dtype == torch.bool else pos_mask
        neg = neg_mask.float() if neg_mask.dtype == torch.bool else neg_mask
        exp_sim = torch.exp(sim)
        logZ = torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)
        log_prob = sim - logZ
        pos_sum = (log_prob * pos).sum(dim=1)
        denom = pos.sum(dim=1).clamp_min(1.0)  # 防除零
        loss = -(pos_sum / denom).mean()
        return loss

    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        return F.normalize(h1, dim=1) @ F.normalize(h2, dim=1).t()

    def forward(self, graph, feat):

        device = feat.device
        N = feat.size(0)

        # —— 三种表征 —— #
        Zf = self.encoder_target(graph, feat)  # [N, out_dim]


        # 使用Zf自身进行对比
        Zh = Zf
        Zl = Zf

        # projector（把 Zf 投到 hid_dim），并全部显式 normalize(dim=1)
        if self.num_MLP and self.num_MLP > 0:
            Zf_use = F.normalize(self.projector(Zf), dim=1)  # [N, hid_dim]
        else:
            # 若不投影，则必须保证 out_dim == hid_dim，否则相似度维度不配
            if Zf.size(1) != Zh.size(1):
                raise ValueError(f"Dim mismatch: Zf({Zf.size(1)}) vs Zh/Zl({Zh.size(1)}) "
                                 f"— 请启用 projector 或把 out_dim 设成 {Zh.size(1)}")
            Zf_use = F.normalize(Zf, dim=1)

        Zh = F.normalize(Zh, dim=1)
        Zl = F.normalize(Zl, dim=1)

        # —— 确保边在同一 device —— #
        if self.edge_index_same is not None:
            self.edge_index_same = self.edge_index_same.to(device=device, dtype=torch.long)
        if self.edge_index_cross is not None:
            self.edge_index_cross = self.edge_index_cross.to(device=device, dtype=torch.long)

        # —— 全图 / mini-batch 路径 —— #
        if (self.batch_size == 0) or (self.batch_size > N):
            # 全图掩码（语义=同层，结构=跨层）
            pos_mask_sem = edge_index_to_mask(self.edge_index_same, N, device, fill_diag=True)
            pos_mask_str = edge_index_to_mask(self.edge_index_cross, N, device, fill_diag=True)
            neg_mask_sem = 1.0 - pos_mask_sem
            neg_mask_str = 1.0 - pos_mask_str

            loss_FH = self.infonce(Zf_use, Zh, pos_mask_sem, neg_mask_sem, self.temp)  # Full–High
            loss_FL = self.infonce(Zf_use, Zl, pos_mask_str, neg_mask_str, self.temp)  # Full–Low
            loss = (1.0 - self.gamma) * loss_FH + self.gamma * loss_FL
            return loss

        else:
            # mini-batch 子图（用你现有的 split_batch / extract_batch_mask）
            node_idxs = list(range(N))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            total_loss = 0.0

            for b in batches:
                batch_nodes = torch.tensor(b, device=device, dtype=torch.long)
                weight = len(b) / N

                pos_sem = extract_batch_mask(self.edge_index_same, batch_nodes, N)
                pos_str = extract_batch_mask(self.edge_index_cross, batch_nodes, N)
                # extract_batch_mask 产物在 edge_index.device；前面我们把 edge_index_* 移到了 model device
                # 所以这里的子图掩码与 Zf/Zh/Zl 设备一致，OK
                neg_sem = 1.0 - pos_sem
                neg_str = 1.0 - pos_str

                loss_FH = self.infonce(Zf_use[batch_nodes], Zh[batch_nodes], pos_sem, neg_sem, self.temp)
                loss_FL = self.infonce(Zf_use[batch_nodes], Zl[batch_nodes], pos_str, neg_str, self.temp)

                total_loss = total_loss + ((1.0 - self.gamma) * loss_FH + self.gamma * loss_FL) * weight

            return total_loss