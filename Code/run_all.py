"""
Ours strict ReSim - Combined Pipeline Runner (XGBoost / MLP)
=============================================================
Runs the full pipeline from 5-fold pair files to final metrics:

  build_fold_split  ->  run_embedding  ->  run_classifier

Run::

    python Code/run_all.py --dataset dataset1 --tasks MDA LDA LMI
    python Code/run_all.py --dataset dataset1 --tasks MDA --classifier mlp
    python Code/run_all.py --dataset dataset1 --tasks MDA --skip-embedding --build-folds
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent


def run_subprocess(cmd: list, description: str) -> None:
    print(f"\n{'#' * 60}")
    print(f"# {description}")
    print(f"{'#' * 60}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"SUCCESS: {description} completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ours strict ReSim - combined pipeline runner (XGBoost / MLP)"
    )
    parser.add_argument("--dataset", type=str, choices=["dataset1", "dataset2"], default="dataset1")
    parser.add_argument("--tasks", nargs="+", choices=["MDA", "LDA", "LMI"], default=["MDA", "LDA", "LMI"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embedding-epochs", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hid-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding generation, reuse existing embeddings.")
    parser.add_argument("--build-folds", action="store_true",
                        help="(Re)generate 5-fold pair files via Code/build_fold_split.py first.")
    parser.add_argument("--classifier", type=str, choices=["xgboost", "mlp"], default="xgboost",
                        help="Choose classifier: xgboost (default) or mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset.capitalize()

    print(f"\n{'=' * 60}")
    classifier_name = "XGBoost" if args.classifier == "xgboost" else "MLP"
    print(f"Ours strict ReSim {classifier_name} Pipeline")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Tasks:       {args.tasks}")
    print(f"  Folds:       {args.folds}")
    print(f"  Classifier:  {classifier_name}")
    print(f"  GPU:         {args.gpu}")
    print(f"  Hid dim:     {args.hid_dim}")
    print(f"  Seed:        {args.seed}")
    print(f"  Build folds: {args.build_folds}")
    print(f"  Skip embed:  {args.skip_embedding}")
    print(f"{'=' * 60}")

    # Step 0: optionally regenerate 5-fold pair files
    if args.build_folds:
        cmd = [
            sys.executable,
            str(CODE_DIR / "build_fold_split.py"),
            "--dataset", args.dataset,
            "--seed", str(args.seed),
        ]
        run_subprocess(cmd, f"Build 5-fold splits for {args.dataset}")

    # Step 1: generate embeddings for each task
    if not args.skip_embedding:
        for task in args.tasks:
            cmd = [
                sys.executable,
                str(CODE_DIR / "run_embedding.py"),
                "--dataset", args.dataset,
                "--task", task,
                "--folds", str(args.folds),
                "--contrastive-epochs", str(args.embedding_epochs),
                "--gpu", str(args.gpu),
                "--hid-dim", str(args.hid_dim),
                "--seed", str(args.seed),
                "--run-name", f"{args.dataset}_{task.lower()}",
            ]
            run_subprocess(cmd, f"Embedding generation for {task}")

    # Step 2: run classification for each task
    for task in args.tasks:
        task_emb_name = f"{args.dataset}_{task.lower()}"
        cmd = [
            sys.executable,
            str(CODE_DIR / "run_classifier.py"),
            "--dataset", args.dataset,
            "--task", task,
            "--embedding-run-name", task_emb_name,
            "--folds", str(args.folds),
            "--seed", str(args.seed),
            "--classifier", args.classifier,
            "--decision-threshold", "0.5",
        ]
        # Add classifier-specific parameters
        if args.classifier == "xgboost":
            cmd += [
                "--n-estimators", "300",
                "--max-depth", "6",
                "--learning-rate", "0.03",
                "--subsample", "0.8",
                "--colsample-bytree", "0.8",
            ]
        else:
            cmd += [
                "--mlp-epochs", "100",
                "--mlp-lr", "0.001",
                "--mlp-hidden-dim", "256",
            ]
        classifier_desc = "XGB" if args.classifier == "xgboost" else "MLP"
        run_subprocess(cmd, f"{classifier_desc} classification for {task}")

    print(f"\n{'=' * 60}")
    print("COMPLETE PIPELINE FINISHED")
    print(f"{'=' * 60}")
    print(f"\nResults saved under:")
    for task in args.tasks:
        result_path = PROJECT_ROOT / "results" / dataset_name / task
        print(f"  - {result_path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - start:.2f}s")