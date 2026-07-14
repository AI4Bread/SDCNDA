"""
Ours strict ReSim Classifier (XGBoost / MLP)
============================================
XGBoost or MLP classifier using ReSim embeddings with no path compensation and no pseudo edges.

Feature vector: endpoint embeddings + absolute difference + element-wise product.

XGBoost parameters: n_estimators=300, max_depth=6, learning_rate=0.03.
MLP parameters: hidden_dim=256, epochs=100, lr=0.001.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.fold_utils import load_dataset_raw, pair_to_global_indices  # noqa: E402


TASK_CONFIGS = {
    "MDA": {
        "matrix_key": "mi_dis",
        "pair_file_suffix": "MDA",
        "left_type": "mi",
        "right_type": "dis",
    },
    "LDA": {
        "matrix_key": "lnc_dis",
        "pair_file_suffix": "LDA",
        "left_type": "lnc",
        "right_type": "dis",
    },
    "LMI": {
        "matrix_key": "mi_lnc",
        "pair_file_suffix": "LMI",
        "left_type": "mi",
        "right_type": "lnc",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ours strict ReSim classifier (XGBoost / MLP)"
    )
    parser.add_argument("--dataset", type=str, choices=["dataset1", "dataset2"], default="dataset1")
    parser.add_argument("--task", type=str, choices=["MDA", "LDA", "LMI"], default="MDA")
    parser.add_argument("--embedding-run-name", type=str, required=True,
                        help="Name of the embedding run (e.g., dataset1_mda)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1024)
    # Classifier selection
    parser.add_argument("--classifier", type=str, choices=["xgboost", "mlp"], default="xgboost",
                        help="Choose classifier: xgboost (default) or mlp")
    # XGBoost parameters
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--xgb-device", type=str, choices=["cpu", "cuda"], default="cpu")
    # MLP parameters
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-lr", type=float, default=0.001)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    # Common parameters
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="(Unused in this baseline; kept for interface compatibility.)")
    parser.add_argument("--run-prefix", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_pair_features(
    embedding: torch.Tensor,
    pairs: np.ndarray,
    dataset: str,
    left_type: str,
    right_type: str,
) -> np.ndarray:
    """Build pair features: [left_emb, right_emb, |left-right|, left*right]"""
    left_idx, right_idx = pair_to_global_indices(
        pairs, dataset=dataset, left_type=left_type, right_type=right_type
    )
    left = embedding[left_idx]
    right = embedding[right_idx]
    return torch.cat([left, right, torch.abs(left - right), left * right], dim=1).numpy()


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_score),
        "aupr": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "acc": accuracy_score(y_true, y_pred),
        "threshold": threshold,
    }


def make_xgb(args: argparse.Namespace, fold: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=args.xgb_device,
        n_jobs=args.n_jobs,
        random_state=args.seed + fold,
    )


class MLPClassifier(nn.Module):
    """MLP classifier for link prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_name = args.dataset.capitalize()
    config = TASK_CONFIGS[args.task]

    embedding_root = ROOT_DIR / "data_processed" / dataset_name / args.task / args.embedding_run_name

    raw = load_dataset_raw(ROOT_DIR, dataset=args.dataset)
    label_matrix = raw[config["matrix_key"]]

    run_name = f"{args.task.lower()}" if not args.run_prefix else f"{args.run_prefix}_{args.task.lower()}"
    out_data_root = ROOT_DIR / "data_processed" / dataset_name / args.task / run_name

    # Choose score directory based on classifier
    score_subdir = "xgboost_scores" if args.classifier == "xgboost" else "mlp_scores"
    score_root = out_data_root / score_subdir
    pair_root = out_data_root / "pairs"
    score_root.mkdir(parents=True, exist_ok=True)
    pair_root.mkdir(parents=True, exist_ok=True)

    results_root = ROOT_DIR / "results" / dataset_name / args.task
    results_root.mkdir(parents=True, exist_ok=True)

    # Determine device for MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []

    for fold in range(1, args.folds + 1):
        print(f"\n{'=' * 60}")
        classifier_name = "XGB-300d6" if args.classifier == "xgboost" else f"MLP-{args.mlp_hidden_dim}"
        print(f"Training {classifier_name} for {args.dataset}/{args.task} Fold {fold}")
        print(f"{'=' * 60}")

        pair_dir = embedding_root / f"fold{fold}" / "pairs"
        train_pairs = np.loadtxt(pair_dir / f"train_pairs_{config['pair_file_suffix']}.txt", dtype=int)
        test_pairs = np.loadtxt(pair_dir / f"test_pairs_{config['pair_file_suffix']}.txt", dtype=int)

        embedding_path = embedding_root / f"fold{fold}" / "embeddings" / "node_embedding.pt"
        embedding = torch.load(embedding_path, map_location="cpu")

        train_x = build_pair_features(embedding, train_pairs, args.dataset, config["left_type"], config["right_type"])
        test_x = build_pair_features(embedding, test_pairs, args.dataset, config["left_type"], config["right_type"])

        y_train = np.array([label_matrix[int(i), int(j)] for i, j in train_pairs], dtype=np.int64)
        y_test = np.array([label_matrix[int(i), int(j)] for i, j in test_pairs], dtype=np.int64)

        fold_pair_dir = pair_root / f"fold{fold}"
        fold_pair_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(fold_pair_dir / f"train_pairs_{config['pair_file_suffix']}.txt", train_pairs, fmt="%d")
        np.savetxt(fold_pair_dir / f"test_pairs_{config['pair_file_suffix']}.txt", test_pairs, fmt="%d")

        if args.classifier == "xgboost":
            model = make_xgb(args, fold)
            model.fit(train_x, y_train)
            scores = model.predict_proba(test_x)[:, 1]
        else:
            # MLP training
            input_dim = train_x.shape[1]
            model = MLPClassifier(input_dim=input_dim, hidden_dim=args.mlp_hidden_dim).to(device)

            train_x_tensor = torch.FloatTensor(train_x).to(device)
            train_y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
            test_x_tensor = torch.FloatTensor(test_x).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
            criterion = nn.BCEWithLogitsLoss()

            model.train()
            for epoch in range(args.mlp_epochs):
                optimizer.zero_grad()
                outputs = model(train_x_tensor)
                loss = criterion(outputs, train_y_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(test_x_tensor)
                scores = torch.sigmoid(logits).cpu().numpy().flatten()

        metrics = evaluate_binary(y_test, scores, args.decision_threshold)

        np.save(score_root / f"fold{fold}_test_scores.npy", scores)

        train_pos = int(y_train.sum())
        test_pos = int(y_test.sum())
        row = {
            "fold": fold,
            "train_pos": train_pos,
            "train_neg": int(len(y_train) - train_pos),
            "test_pos": test_pos,
            "test_neg": int(len(y_test) - test_pos),
            **metrics,
        }
        rows.append(row)

        print(
            f"Fold {fold}: train_pos={row['train_pos']} train_neg={row['train_neg']} "
            f"test_pos={row['test_pos']} test_neg={row['test_neg']}\n"
            f"  AUC={metrics['auc']:.4f} AUPR={metrics['aupr']:.4f} "
            f"F1={metrics['f1']:.4f} ACC={metrics['acc']:.4f}"
        )

        del train_x, test_x, embedding, model

    df = pd.DataFrame(rows)
    metric_cols = [col for col in df.columns if col != "fold"]
    mean_row = {"fold": "mean", **df[metric_cols].mean().to_dict()}
    std_row = {"fold": "std", **df[metric_cols].std().to_dict()}
    summary = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    out_csv = results_root / f"{run_name}.csv"
    out_json = results_root / f"{run_name}.json"
    summary.to_csv(out_csv, index=False)

    config_json = vars(args).copy()
    config_json.update({
        "dataset": args.dataset,
        "task": args.task,
        "embedding_run_name": args.embedding_run_name,
        "feature_mode": "pair_interact",
        "classifier_features": "endpoint_embeddings + abs_diff + product",
        "pseudo_edges": "disabled",
        "path_features": "disabled",
    })
    out_json.write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Results Summary: {args.dataset}/{args.task}")
    print(f"{'=' * 60}")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - start:.2f}s")