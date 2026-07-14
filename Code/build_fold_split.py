"""5-fold split generator.

Reads the three required adjacency matrices (mi_dis, lnc_dis, mi_lnc) from
``data/<dataset>/`` and writes 30 pair files under
``data/<dataset>/datasplit/data1.1`` (5 folds x 3 relations x train/test).

For every relation matrix this writes:

    <relation>_train_id1.txt ... <relation>_train_id5.txt
    <relation>_test_id1.txt  ... <relation>_test_id5.txt

Each file has the shape ``(N, 2)`` and is laid out as
``[positive_pairs | negative_pairs]``.

Run::

    python Code/build_fold_split.py --dataset dataset1
    python Code/build_fold_split.py --dataset dataset2 --seed 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SEED = 1024
N_FOLDS = 5


def load_dataset1(data_dir: Path) -> dict[str, np.ndarray]:
    return {
        "lnc_dis": np.loadtxt(data_dir / "lnc_dis_association.txt"),
        "mi_dis": np.loadtxt(data_dir / "mi_dis.txt"),
        "mi_lnc": np.loadtxt(data_dir / "yuguoxian_lnc_mi.txt").T,
    }


def load_dataset2(data_dir: Path) -> dict[str, np.ndarray]:
    return {
        "lnc_dis": np.loadtxt(data_dir / "dis_lnc.txt").T,
        "mi_dis": np.loadtxt(data_dir / "dis_mi.txt").T,
        "mi_lnc": np.loadtxt(data_dir / "mi_lnc.txt"),
    }


DATASET_LOADERS = {
    "dataset1": load_dataset1,
    "dataset2": load_dataset2,
}


def make_balanced_folds(
    matrix: np.ndarray,
    output_dir: Path,
    prefix: str,
    seed: int,
) -> None:
    """Split a (n_left, n_right) association matrix into 5 balanced folds.

    For each fold:
      - positives are split 1/5 test, 4/5 train.
      - negatives: sample ``len(train_pos)`` train negatives without replacement,
        then ``len(test_pos)`` test negatives from the remaining pool.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_samples = np.argwhere(matrix > 0)
    negative_samples = np.argwhere(matrix == 0)

    if len(positive_samples) == 0:
        raise ValueError(f"{prefix} has no positive samples")

    positive_samples = rng.permutation(positive_samples)
    positive_subsets = np.array_split(positive_samples, N_FOLDS)

    for fold_idx in range(N_FOLDS):
        test_positive = positive_subsets[fold_idx]
        train_positive = np.vstack(
            [positive_subsets[j] for j in range(N_FOLDS) if j != fold_idx]
        )

        train_negative_indices = rng.choice(
            len(negative_samples), size=len(train_positive), replace=False
        )
        train_negative = negative_samples[train_negative_indices]

        remaining_negative_mask = np.ones(len(negative_samples), dtype=bool)
        remaining_negative_mask[train_negative_indices] = False
        remaining_negative = negative_samples[remaining_negative_mask]
        test_negative_indices = rng.choice(
            len(remaining_negative), size=len(test_positive), replace=False
        )
        test_negative = remaining_negative[test_negative_indices]

        train_data = np.vstack((train_positive, train_negative)).astype(int)
        test_data = np.vstack((test_positive, test_negative)).astype(int)

        np.savetxt(output_dir / f"{prefix}_train_id{fold_idx + 1}.txt", train_data, fmt="%d")
        np.savetxt(output_dir / f"{prefix}_test_id{fold_idx + 1}.txt", test_data, fmt="%d")

        print(
            f"  {prefix} fold {fold_idx + 1}: "
            f"train_pos={len(train_positive)} train_neg={len(train_negative)} "
            f"test_pos={len(test_positive)} test_neg={len(test_negative)}"
        )


def build_splits(dataset: str, data_dir: Path, output_dir: Path, seed: int) -> None:
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    matrices = DATASET_LOADERS[dataset](data_dir)

    make_balanced_folds(matrices["lnc_dis"], output_dir, "lnc_dis", seed)
    make_balanced_folds(matrices["mi_dis"], output_dir, "mi_dis", seed + 1)
    make_balanced_folds(matrices["mi_lnc"], output_dir, "mi_lnc", seed + 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 5-fold train/test pair files from raw adjacency matrices."
    )
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Master seed; relations use seed, seed+1, seed+2.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override data root (default: <project_root>/data/<dataset>)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override split dir (default: <project_root>/data/<dataset>/datasplit/data1.1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    data_dir = args.data_dir if args.data_dir else (project_root / "data" / args.dataset)
    output_dir = args.output_dir if args.output_dir else (data_dir / "datasplit" / "data1.1")

    print(f"Dataset: {args.dataset}")
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Seed:       {args.seed}")

    if not data_dir.exists():
        print(f"ERROR: data dir does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    build_splits(args.dataset, data_dir, output_dir, args.seed)
    print(f"\nGenerated 30 pair files under {output_dir}")


if __name__ == "__main__":
    main()