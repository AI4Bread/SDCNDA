from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split


DATASET_META = {
    "dataset1": {
        "n_lnc": 240,
        "n_dis": 405,
        "n_mi": 495,
        "lnc_dis_file": "lnc_dis_association.txt",
        "mi_dis_file": "mi_dis.txt",
        "mi_lnc_file": "yuguoxian_lnc_mi.txt",
        "dis_sem_file": "dis_sem_sim.txt",
    },
    "dataset2": {
        "n_lnc": 665,
        "n_dis": 316,
        "n_mi": 295,
        "lnc_dis_file": "dis_lnc.txt",
        "mi_dis_file": "dis_mi.txt",
        "mi_lnc_file": "mi_lnc.txt",
        "dis_sem_file": "disease_semantic_sim.txt",
    },
}


@dataclass
class FoldArtifacts:
    fold: int
    fold_dir: Path
    train_matrix_dir: Path
    graph_dir: Path
    pair_dir: Path
    embedding_dir: Path
    log_dir: Path
    compensation_dir: Path


def dataset1_dir(root_dir: Path) -> Path:
    return root_dir / "data" / "dataset1"


def get_dataset_meta(dataset: str = "dataset1") -> dict[str, object]:
    dataset = dataset.lower()
    if dataset not in DATASET_META:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_META[dataset]


def dataset_dir(root_dir: Path, dataset: str = "dataset1") -> Path:
    return root_dir / "data" / dataset.lower()


def split_dir(root_dir: Path, dataset: str = "dataset1") -> Path:
    return dataset_dir(root_dir, dataset) / "datasplit" / "data1.1"


def load_dataset_raw(root_dir: Path, dataset: str = "dataset1") -> dict[str, np.ndarray]:
    meta = get_dataset_meta(dataset)
    data_dir = dataset_dir(root_dir, dataset)
    lnc_dis = np.loadtxt(data_dir / meta["lnc_dis_file"], dtype=np.float32)
    mi_dis = np.loadtxt(data_dir / meta["mi_dis_file"], dtype=np.float32)
    mi_lnc = np.loadtxt(data_dir / meta["mi_lnc_file"], dtype=np.float32)

    if dataset.lower() == "dataset1":
        mi_lnc = mi_lnc.T
    else:
        lnc_dis = lnc_dis.T
        mi_dis = mi_dis.T

    return {
        "lnc_dis": lnc_dis.astype(np.float32),
        "mi_dis": mi_dis.astype(np.float32),
        "mi_lnc": mi_lnc.astype(np.float32),
    }


def load_dataset1_raw(root_dir: Path) -> dict[str, np.ndarray]:
    return load_dataset_raw(root_dir, dataset="dataset1")


def get_fold_ids(root_dir: Path, fold: int, dataset: str = "dataset1", relation_prefix: str = "mi_dis") -> tuple[np.ndarray, np.ndarray]:
    base = split_dir(root_dir, dataset)
    train_pairs = np.loadtxt(base / f"{relation_prefix}_train_id{fold}.txt", dtype=int)
    test_pairs = np.loadtxt(base / f"{relation_prefix}_test_id{fold}.txt", dtype=int)
    return np.atleast_2d(train_pairs), np.atleast_2d(test_pairs)


def zero_pairs(matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    train_matrix = matrix.copy()
    for i, j in np.atleast_2d(pairs):
        train_matrix[int(i), int(j)] = 0.0
    return train_matrix


def assert_no_test_edges(train_matrix: np.ndarray, test_pairs: np.ndarray, name: str) -> None:
    leaked = 0
    for i, j in np.atleast_2d(test_pairs):
        if train_matrix[int(i), int(j)] != 0:
            leaked += 1
    if leaked:
        raise ValueError(f"{name}: found {leaked} leaked test edges")


def construct_graph(
    lnc_dis: np.ndarray,
    mi_dis: np.ndarray,
    mi_lnc: np.ndarray,
    lnc_sim: np.ndarray,
    mi_sim: np.ndarray,
    dis_sim: np.ndarray,
) -> np.ndarray:
    lnc_block = np.hstack((lnc_sim, lnc_dis, mi_lnc.T))
    dis_block = np.hstack((lnc_dis.T, dis_sim, mi_dis.T))
    mi_block = np.hstack((mi_lnc, mi_dis, mi_sim))
    return np.vstack((lnc_block, dis_block, mi_block)).astype(np.float32)


def symmetric_normalize(adj: np.ndarray) -> np.ndarray:
    degree = np.array(adj.sum(1), dtype=np.float32)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = np.power(degree[mask], -0.5)
    d_mat = np.diag(inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).astype(np.float32)


def build_edge_views(matrix_a: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = np.where(matrix_a > 0)
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    same_mask = label_tensor[edge_index[0]] == label_tensor[edge_index[1]]
    edge_index_same = edge_index[:, same_mask]
    edge_index_cross = edge_index[:, ~same_mask]
    return edge_index, edge_index_same, edge_index_cross


def build_labels(dataset: str = "dataset1") -> np.ndarray:
    meta = get_dataset_meta(dataset)
    return np.array([0] * meta["n_lnc"] + [1] * meta["n_dis"] + [2] * meta["n_mi"], dtype=np.int64)


def split_train_val_pairs(
    train_pairs: np.ndarray,
    mi_dis_full: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([mi_dis_full[int(i), int(j)] for i, j in np.atleast_2d(train_pairs)], dtype=np.int64)
    idx = np.arange(len(train_pairs))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_pairs[train_idx], labels[train_idx], train_pairs[val_idx], labels[val_idx]


def pair_to_global_indices(
    pairs: np.ndarray,
    dataset: str = "dataset1",
    left_type: str = "mi",
    right_type: str = "dis",
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.atleast_2d(pairs)
    meta = get_dataset_meta(dataset)
    offsets = {
        "lnc": 0,
        "dis": meta["n_lnc"],
        "mi": meta["n_lnc"] + meta["n_dis"],
    }
    left_nodes = offsets[left_type] + pairs[:, 0].astype(int)
    right_nodes = offsets[right_type] + pairs[:, 1].astype(int)
    return left_nodes, right_nodes


def make_fold_artifacts(base_dir: Path, fold: int) -> FoldArtifacts:
    fold_dir = base_dir / f"fold{fold}"
    train_matrix_dir = fold_dir / "train_matrices"
    graph_dir = fold_dir / "graph"
    pair_dir = fold_dir / "pairs"
    embedding_dir = fold_dir / "embeddings"
    log_dir = fold_dir / "logs"
    compensation_dir = fold_dir / "compensation"
    for path in (train_matrix_dir, graph_dir, pair_dir, embedding_dir, log_dir, compensation_dir):
        path.mkdir(parents=True, exist_ok=True)
    return FoldArtifacts(
        fold=fold,
        fold_dir=fold_dir,
        train_matrix_dir=train_matrix_dir,
        graph_dir=graph_dir,
        pair_dir=pair_dir,
        embedding_dir=embedding_dir,
        log_dir=log_dir,
        compensation_dir=compensation_dir,
    )


def save_matrix_txt(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.8f")


def save_pairs(path: Path, pairs: np.ndarray) -> None:
    np.savetxt(path, np.atleast_2d(pairs), fmt="%d")


def write_log(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
