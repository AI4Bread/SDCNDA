"""
Ours strict ReSim - Embedding Generation
========================================
Strict fold-wise leakage-free similarity reconstruction and graph contrastive embedding.

- No path compensation features.
- No pseudo edge augmentation.
- Uses reconstructed similarity matrices from fold-wise train data.

Project layout assumption (all paths are computed from this file's location):
    ROOT_DIR      = parents[1]  (= project root)
    DATA_DIR      = ROOT_DIR / "data" / <dataset>
    PROCESSED_DIR = ROOT_DIR / "data_processed" / Dataset / Task / <run_name>
    RESULTS_DIR   = ROOT_DIR / "results"      / Dataset / Task
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_undirected

# Project root (parent of Code/)
ROOT_DIR = Path(__file__).resolve().parents[1]
# Make this directory importable so `model` etc. resolve when running directly
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.model import Model_our  # noqa: E402
from Code.fold_utils import (  # noqa: E402
    FoldArtifacts,
    assert_no_test_edges,
    build_edge_views,
    build_labels,
    construct_graph,
    get_fold_ids,
    load_dataset_raw,
    make_fold_artifacts,
    save_matrix_txt,
    save_pairs,
    symmetric_normalize,
    write_log,
    zero_pairs,
)
from Code.similarity_reconstruction import (  # noqa: E402
    reconstruct_lda_similarity,
    reconstruct_lmi_similarity,
    reconstruct_mda_similarity,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ours strict ReSim: leakage-free fold-wise embedding generation"
    )
    parser.add_argument("--dataset", type=str, choices=["dataset1", "dataset2"], default="dataset1")
    parser.add_argument("--task", type=str, choices=["MDA", "LDA", "LMI"], default="MDA")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-ids", type=int, nargs="*", default=None)
    parser.add_argument("--contrastive-epochs", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hid-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num-mlp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--lr1", type=float, default=5e-4)
    parser.add_argument("--wd1", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def get_device(args: argparse.Namespace) -> torch.device:
    if args.gpu != -1 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def process_fold_mda(args: argparse.Namespace, fold: int, processed_root: Path) -> tuple[FoldArtifacts, dict]:
    raw = load_dataset_raw(ROOT_DIR, dataset=args.dataset)
    mi_dis_full = raw["mi_dis"]
    train_pairs, test_pairs = get_fold_ids(ROOT_DIR, fold, dataset=args.dataset, relation_prefix="mi_dis")

    mi_dis_train = zero_pairs(mi_dis_full, test_pairs)
    assert_no_test_edges(mi_dis_train, test_pairs, "MD_train")

    sim_bundle = reconstruct_mda_similarity(ROOT_DIR, raw, mi_dis_train, dataset=args.dataset)

    fold_artifacts = make_fold_artifacts(processed_root, fold)
    save_matrix_txt(fold_artifacts.train_matrix_dir / "MD_train.txt", mi_dis_train)
    save_matrix_txt(fold_artifacts.compensation_dir / "mi_fusion_sim_fold.txt", sim_bundle["mi_sim"])
    save_matrix_txt(fold_artifacts.compensation_dir / "lnc_fusion_sim_fold.txt", sim_bundle["lnc_sim"])
    save_matrix_txt(fold_artifacts.compensation_dir / "dis_fusion_sim_fold.txt", sim_bundle["dis_sim"])
    save_pairs(fold_artifacts.pair_dir / f"train_pairs_{args.task}.txt", train_pairs)
    save_pairs(fold_artifacts.pair_dir / f"test_pairs_{args.task}.txt", test_pairs)

    matrix_a = construct_graph(
        raw["lnc_dis"],
        mi_dis_train,  # no pseudo edges
        raw["mi_lnc"],
        sim_bundle["lnc_sim"],
        sim_bundle["mi_sim"],
        sim_bundle["dis_sim"],
    )
    edge_index, edge_same, edge_cross = build_edge_views(matrix_a, build_labels(args.dataset))
    feature = symmetric_normalize(matrix_a)

    np.save(fold_artifacts.graph_dir / "matrix_A.npy", matrix_a)
    torch.save(edge_index, fold_artifacts.graph_dir / "edges.pt")
    torch.save(edge_same, fold_artifacts.graph_dir / "same_edges.pt")
    torch.save(edge_cross, fold_artifacts.graph_dir / "cross_edges.pt")
    torch.save(torch.tensor(feature, dtype=torch.float32), fold_artifacts.graph_dir / "feature.pt")
    torch.save(torch.tensor(build_labels(args.dataset), dtype=torch.int64), fold_artifacts.graph_dir / "label.pt")

    write_log(
        fold_artifacts.log_dir / "leakage_check.txt",
        f"Fold {fold}\nMD test edges removed: passed\n"
        f"pseudo_edges: 0 (disabled)\n"
        f"path_features: 0 (disabled)\n"
        f"sim_mix_alpha: 0.0 (not used)\n"
        f"mi_sim: fold-wise recon\n"
        f"dis_sim: fold-wise recon\n"
        f"lnc_sim: fold-wise recon\n",
    )
    write_log(
        fold_artifacts.log_dir / "graph_stats.txt",
        f"matrix_A shape: {matrix_a.shape}\n"
        f"total edges: {edge_index.shape[1]}\n"
        f"same-type edges: {edge_same.shape[1]}\n"
        f"cross-type edges: {edge_cross.shape[1]}\n"
        f"pseudo_edges: 0\n"
        f"sim_mix_alpha: 0.0 (not used)\n",
    )

    payload = {
        "feature": feature,
        "edge_index": edge_index,
        "edge_same": edge_same,
        "edge_cross": edge_cross,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
    }
    return fold_artifacts, payload


def process_fold_lda(args: argparse.Namespace, fold: int, processed_root: Path) -> tuple[FoldArtifacts, dict]:
    raw = load_dataset_raw(ROOT_DIR, dataset=args.dataset)
    lnc_dis_full = raw["lnc_dis"]
    train_pairs, test_pairs = get_fold_ids(ROOT_DIR, fold, dataset=args.dataset, relation_prefix="lnc_dis")

    lnc_dis_train = zero_pairs(lnc_dis_full, test_pairs)
    assert_no_test_edges(lnc_dis_train, test_pairs, "LD_train")

    sim_bundle = reconstruct_lda_similarity(ROOT_DIR, raw, lnc_dis_train, dataset=args.dataset)
    # mi_sim: use RAW (no fold-wise recon) — unchanged, fold-wise recon would need mi_dis train matrix
    # lnc_sim: fold-wise recon (sim_mix_alpha=0, pure recon)
    # dis_sim: fold-wise recon (sim_mix_alpha=0, pure recon)

    fold_artifacts = make_fold_artifacts(processed_root, fold)
    save_matrix_txt(fold_artifacts.train_matrix_dir / "LD_train.txt", lnc_dis_train)
    save_matrix_txt(fold_artifacts.compensation_dir / "lnc_fusion_sim_fold.txt", sim_bundle["lnc_sim"])
    save_matrix_txt(fold_artifacts.compensation_dir / "dis_fusion_sim_fold.txt", sim_bundle["dis_sim"])
    save_pairs(fold_artifacts.pair_dir / f"train_pairs_{args.task}.txt", train_pairs)
    save_pairs(fold_artifacts.pair_dir / f"test_pairs_{args.task}.txt", test_pairs)

    # Reference exact: use FULL mi_dis and mi_lnc (no fold-wise zeroing).
    # Only the TARGET relation lnc_dis is fold-wise zeroed.
    matrix_a = construct_graph(
        lnc_dis_train,
        raw["mi_dis"],      # FULL, not fold-wise zeroed
        raw["mi_lnc"],     # FULL, not fold-wise zeroed
        sim_bundle["lnc_sim"],
        raw["mi_sim"],     # RAW, not reconstructed
        sim_bundle["dis_sim"],
    )
    edge_index, edge_same, edge_cross = build_edge_views(matrix_a, build_labels(args.dataset))
    feature = symmetric_normalize(matrix_a)

    np.save(fold_artifacts.graph_dir / "matrix_A.npy", matrix_a)
    torch.save(edge_index, fold_artifacts.graph_dir / "edges.pt")
    torch.save(edge_same, fold_artifacts.graph_dir / "same_edges.pt")
    torch.save(edge_cross, fold_artifacts.graph_dir / "cross_edges.pt")
    torch.save(torch.tensor(feature, dtype=torch.float32), fold_artifacts.graph_dir / "feature.pt")
    torch.save(torch.tensor(build_labels(args.dataset), dtype=torch.int64), fold_artifacts.graph_dir / "label.pt")

    write_log(
        fold_artifacts.log_dir / "leakage_check.txt",
        f"Fold {fold}\nLD test edges removed: passed\n"
        f"pseudo_edges: 0 (disabled)\n"
        f"path_features: 0 (disabled)\n"
        f"sim_mix_alpha: 0.0 (not used)\n"
        f"lnc_sim: fold-wise recon\n"
        f"dis_sim: fold-wise recon\n"
        f"mi_sim: RAW (no recon)\n"
        f"mi_dis: FULL (no fold-wise zeroing)\n"
        f"mi_lnc: FULL (no fold-wise zeroing)\n",
    )
    write_log(
        fold_artifacts.log_dir / "graph_stats.txt",
        f"matrix_A shape: {matrix_a.shape}\n"
        f"total edges: {edge_index.shape[1]}\n"
        f"same-type edges: {edge_same.shape[1]}\n"
        f"cross-type edges: {edge_cross.shape[1]}\n"
        f"pseudo_edges: 0\n"
        f"sim_mix_alpha: 0.0 (not used)\n",
    )

    payload = {
        "feature": feature,
        "edge_index": edge_index,
        "edge_same": edge_same,
        "edge_cross": edge_cross,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
    }
    return fold_artifacts, payload


def process_fold_lmi(args: argparse.Namespace, fold: int, processed_root: Path) -> tuple[FoldArtifacts, dict]:
    raw = load_dataset_raw(ROOT_DIR, dataset=args.dataset)
    mi_lnc_full = raw["mi_lnc"]
    train_pairs, test_pairs = get_fold_ids(ROOT_DIR, fold, dataset=args.dataset, relation_prefix="mi_lnc")

    mi_lnc_train = zero_pairs(mi_lnc_full, test_pairs)
    assert_no_test_edges(mi_lnc_train, test_pairs, "LM_train")

    # LMI: use fold-wise reconstructed sim (no mix with raw), matching reference
    sim_bundle = reconstruct_lmi_similarity(ROOT_DIR, raw, mi_lnc_train, dataset=args.dataset)
    mi_sim_for_graph = sim_bundle["mi_sim"]
    lnc_sim_for_graph = sim_bundle["lnc_sim"]
    dis_sim_for_graph = sim_bundle["dis_sim"]

    fold_artifacts = make_fold_artifacts(processed_root, fold)
    save_matrix_txt(fold_artifacts.train_matrix_dir / "LM_train.txt", mi_lnc_train)
    save_matrix_txt(fold_artifacts.compensation_dir / "mi_fusion_sim_fold.txt", sim_bundle["mi_sim"])
    save_matrix_txt(fold_artifacts.compensation_dir / "lnc_fusion_sim_fold.txt", sim_bundle["lnc_sim"])
    save_pairs(fold_artifacts.pair_dir / f"train_pairs_{args.task}.txt", train_pairs)
    save_pairs(fold_artifacts.pair_dir / f"test_pairs_{args.task}.txt", test_pairs)

    # LMI: fold-wise recon sim, FULL lnc_dis and mi_dis, only mi_lnc is fold-wise zeroed
    matrix_a = construct_graph(
        raw["lnc_dis"],         # FULL
        raw["mi_dis"],          # FULL
        mi_lnc_train,
        lnc_sim_for_graph,     # fold-wise recon
        mi_sim_for_graph,      # fold-wise recon
        dis_sim_for_graph,     # fold-wise recon
    )
    edge_index, edge_same, edge_cross = build_edge_views(matrix_a, build_labels(args.dataset))
    feature = symmetric_normalize(matrix_a)

    np.save(fold_artifacts.graph_dir / "matrix_A.npy", matrix_a)
    torch.save(edge_index, fold_artifacts.graph_dir / "edges.pt")
    torch.save(edge_same, fold_artifacts.graph_dir / "same_edges.pt")
    torch.save(edge_cross, fold_artifacts.graph_dir / "cross_edges.pt")
    torch.save(torch.tensor(feature, dtype=torch.float32), fold_artifacts.graph_dir / "feature.pt")
    torch.save(torch.tensor(build_labels(args.dataset), dtype=torch.int64), fold_artifacts.graph_dir / "label.pt")

    write_log(
        fold_artifacts.log_dir / "leakage_check.txt",
        f"Fold {fold}\nLM test edges removed: passed\n"
        f"pseudo_edges: 0 (disabled)\n"
        f"path_features: 0 (disabled)\n"
        f"sim: fold-wise recon (LMI, no mix)\n"
        f"lnc_dis: FULL (no fold-wise zeroing)\n"
        f"mi_dis: FULL (no fold-wise zeroing)\n",
    )
    write_log(
        fold_artifacts.log_dir / "graph_stats.txt",
        f"matrix_A shape: {matrix_a.shape}\n"
        f"total edges: {edge_index.shape[1]}\n"
        f"same-type edges: {edge_same.shape[1]}\n"
        f"cross-type edges: {edge_cross.shape[1]}\n"
        f"pseudo_edges: 0\n"
        f"sim: fold-wise recon (LMI)\n",
    )

    payload = {
        "feature": feature,
        "edge_index": edge_index,
        "edge_same": edge_same,
        "edge_cross": edge_cross,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
    }
    return fold_artifacts, payload


def train_contrastive_one_fold(
    args: argparse.Namespace,
    device: torch.device,
    fold_artifacts: FoldArtifacts,
    payload: dict,
) -> torch.Tensor:
    edge_index = to_undirected(payload["edge_index"])
    edge_same = to_undirected(payload["edge_same"])
    edge_cross = to_undirected(payload["edge_cross"])

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = g.remove_self_loop().add_self_loop().to(device)
    u, v = g.edges()
    g.edge_index = torch.stack([u, v]).to(device=device, dtype=torch.long)

    feat = torch.tensor(payload["feature"], dtype=torch.float32, device=device)
    model = Model_our(
        in_dim=feat.shape[1],
        hid_dim=args.hid_dim,
        out_dim=args.hid_dim,
        num_layers=args.n_layers,
        temp=args.temp,
        use_mlp=False,
        num_MLP=args.num_mlp,
        gamma=args.gamma,
        k=10,
        edge_index_same=edge_same,
        edge_index_cross=edge_cross,
        batch_size=args.batch_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    for epoch in range(args.contrastive_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(g, feat)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == args.contrastive_epochs - 1:
            print(f"Fold {fold_artifacts.fold}: contrastive epoch {epoch:03d}, loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embedding = model.get_embedding(g, feat).cpu()
    torch.save(embedding, fold_artifacts.embedding_dir / "node_embedding.pt")
    return embedding


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"{args.dataset}_{args.task.lower()}"

    set_seed(args.seed)
    device = get_device(args)

    dataset_name = args.dataset.capitalize()
    processed_root = ROOT_DIR / "data_processed" / dataset_name / args.task / args.run_name

    process_fold_fn = {
        "MDA": process_fold_mda,
        "LDA": process_fold_lda,
        "LMI": process_fold_lmi,
    }[args.task]

    rows = []
    fold_ids = args.fold_ids if args.fold_ids else list(range(1, args.folds + 1))

    for fold in fold_ids:
        print(f"\n{'=' * 60}")
        print(f"Processing {args.dataset}/{args.task} Fold {fold}")
        print(f"{'=' * 60}")

        fold_artifacts, payload = process_fold_fn(args, fold, processed_root)
        embedding = train_contrastive_one_fold(args, device, fold_artifacts, payload)

        rows.append({
            "fold": fold,
            "embedding_dim": embedding.shape[1],
            "num_nodes": embedding.shape[0],
        })
        print(f"Fold {fold}: embedding shape = {embedding.shape}")

    df = pd.DataFrame(rows)
    print(f"\nEmbedding generation complete for {len(rows)} folds")
    print(df.to_string(index=False))

    config_path = processed_root / "embedding_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - start:.2f}s")