from __future__ import annotations

from pathlib import Path

import numpy as np


def mix_similarity(raw_sim: np.ndarray, recon_sim: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if raw_sim.shape != recon_sim.shape:
        raise ValueError(f"raw/recon similarity shape mismatch: {raw_sim.shape} vs {recon_sim.shape}")
    return (alpha * raw_sim + (1.0 - alpha) * recon_sim).astype(np.float32)


def calculate_kernel_bandwidth(matrix: np.ndarray) -> float:
    ip_sum = 0.0
    for i in range(matrix.shape[0]):
        ip_sum += np.square(np.linalg.norm(matrix[i]))
    return 1.0 / ((1.0 / matrix.shape[0]) * ip_sum)


def calculate_gaussian_kernel_sim(matrix: np.ndarray) -> np.ndarray:
    bandwidth = calculate_kernel_bandwidth(matrix)
    sim = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float32)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            diff = matrix[i] - matrix[j]
            sim[i, j] = np.exp(-bandwidth * np.square(np.linalg.norm(diff)))
    return sim


def pbpa(rna_i: int, rna_j: int, disease_sim: np.ndarray, rna_disease: np.ndarray) -> float:
    disease_set_i = rna_disease[rna_i] > 0
    disease_set_j = rna_disease[rna_j] > 0
    disease_sim_ij = disease_sim[disease_set_i][:, disease_set_j]
    if disease_sim_ij.shape[0] == 0 or disease_sim_ij.shape[1] == 0:
        return 0.0
    return (
        np.max(disease_sim_ij, axis=0).sum() + np.max(disease_sim_ij, axis=1).sum()
    ) / (disease_sim_ij.shape[0] + disease_sim_ij.shape[1])


def get_rna_functional_sim(rna_len: int, disease_sim: np.ndarray, rna_disease: np.ndarray) -> np.ndarray:
    sim = np.zeros((rna_len, rna_len), dtype=np.float32)
    for i in range(rna_len):
        for j in range(i + 1, rna_len):
            sim[i, j] = sim[j, i] = pbpa(i, j, disease_sim, rna_disease)
    sim += np.eye(rna_len, dtype=np.float32)
    return sim


def rna_fusion_sim(gip_1: np.ndarray, gip_2: np.ndarray, functional_sim: np.ndarray) -> np.ndarray:
    fusion = np.zeros_like(functional_sim, dtype=np.float32)
    avg = (gip_1 + gip_2) / 2.0
    for i in range(fusion.shape[0]):
        for j in range(fusion.shape[1]):
            fusion[i, j] = functional_sim[i, j] if functional_sim[i, j] > 0 else avg[i, j]
    return fusion


def dis_fusion_sim(gip_1: np.ndarray, gip_2: np.ndarray, dis_sem: np.ndarray) -> np.ndarray:
    return ((dis_sem + (gip_1 + gip_2) / 2.0) / 2.0).astype(np.float32)


def _load_dis_sem(root_dir: Path, dataset: str) -> np.ndarray:
    if dataset.lower() == "dataset1":
        dis_sem_file = "dis_sem_sim.txt"
    elif dataset.lower() == "dataset2":
        dis_sem_file = "disease_semantic_sim.txt"
    else:
        raise ValueError(f"Unsupported dataset for similarity reconstruction: {dataset}")
    data_dir = root_dir / "data" / dataset.lower()
    return np.loadtxt(data_dir / dis_sem_file, dtype=np.float32)


def reconstruct_mda_similarity(root_dir: Path, raw: dict[str, np.ndarray], mi_dis_train: np.ndarray, dataset: str = "dataset1") -> dict[str, np.ndarray]:
    dis_sem = _load_dis_sem(root_dir, dataset)

    mi_gip_from_md = calculate_gaussian_kernel_sim(mi_dis_train)
    mi_gip_from_ml = calculate_gaussian_kernel_sim(raw["mi_lnc"])
    mi_functional_sim = get_rna_functional_sim(mi_dis_train.shape[0], dis_sem, mi_dis_train)
    mi_sim_fold = rna_fusion_sim(mi_gip_from_md, mi_gip_from_ml, mi_functional_sim)

    lnc_gip_from_ld = calculate_gaussian_kernel_sim(raw["lnc_dis"])
    lnc_gip_from_lm = calculate_gaussian_kernel_sim(raw["mi_lnc"].T)
    lnc_functional_sim = get_rna_functional_sim(raw["lnc_dis"].shape[0], dis_sem, raw["lnc_dis"])
    lnc_sim_fold = rna_fusion_sim(lnc_gip_from_ld, lnc_gip_from_lm, lnc_functional_sim)

    dis_gip_from_ld = calculate_gaussian_kernel_sim(raw["lnc_dis"].T)
    dis_gip_from_md = calculate_gaussian_kernel_sim(mi_dis_train.T)
    dis_sim_fold = dis_fusion_sim(dis_gip_from_ld, dis_gip_from_md, dis_sem)

    return {
        "mi_sim": mi_sim_fold.astype(np.float32),
        "lnc_sim": lnc_sim_fold.astype(np.float32),
        "dis_sim": dis_sim_fold.astype(np.float32),
        "mi_functional_sim": mi_functional_sim.astype(np.float32),
        "lnc_functional_sim": lnc_functional_sim.astype(np.float32),
        "dis_sem": dis_sem.astype(np.float32),
        "mi_gip_from_md": mi_gip_from_md.astype(np.float32),
        "mi_gip_from_ml": mi_gip_from_ml.astype(np.float32),
        "lnc_gip_from_ld": lnc_gip_from_ld.astype(np.float32),
        "lnc_gip_from_lm": lnc_gip_from_lm.astype(np.float32),
        "dis_gip_from_ld": dis_gip_from_ld.astype(np.float32),
        "dis_gip_from_md": dis_gip_from_md.astype(np.float32),
    }


def reconstruct_dataset1_mda_similarity(root_dir: Path, raw: dict[str, np.ndarray], mi_dis_train: np.ndarray) -> dict[str, np.ndarray]:
    return reconstruct_mda_similarity(root_dir, raw, mi_dis_train, dataset="dataset1")


def reconstruct_lda_similarity(
    root_dir: Path,
    raw: dict[str, np.ndarray],
    lnc_dis_train: np.ndarray,
    dataset: str = "dataset1",
) -> dict[str, np.ndarray]:
    dis_sem = _load_dis_sem(root_dir, dataset)

    lnc_gip_from_ld = calculate_gaussian_kernel_sim(lnc_dis_train)
    lnc_gip_from_lm = calculate_gaussian_kernel_sim(raw["mi_lnc"].T)
    lnc_functional_sim = get_rna_functional_sim(lnc_dis_train.shape[0], dis_sem, lnc_dis_train)
    lnc_sim_fold = rna_fusion_sim(lnc_gip_from_ld, lnc_gip_from_lm, lnc_functional_sim)

    dis_gip_from_ld = calculate_gaussian_kernel_sim(lnc_dis_train.T)
    dis_gip_from_md = calculate_gaussian_kernel_sim(raw["mi_dis"].T)
    dis_sim_fold = dis_fusion_sim(dis_gip_from_ld, dis_gip_from_md, dis_sem)

    return {
        "lnc_sim": lnc_sim_fold.astype(np.float32),
        "dis_sim": dis_sim_fold.astype(np.float32),
        "lnc_functional_sim": lnc_functional_sim.astype(np.float32),
        "dis_sem": dis_sem.astype(np.float32),
        "lnc_gip_from_ld": lnc_gip_from_ld.astype(np.float32),
        "lnc_gip_from_lm": lnc_gip_from_lm.astype(np.float32),
        "dis_gip_from_ld": dis_gip_from_ld.astype(np.float32),
        "dis_gip_from_md": dis_gip_from_md.astype(np.float32),
    }


def reconstruct_dataset1_lda_similarity(root_dir: Path, raw: dict[str, np.ndarray], lnc_dis_train: np.ndarray) -> dict[str, np.ndarray]:
    return reconstruct_lda_similarity(root_dir, raw, lnc_dis_train, dataset="dataset1")


def reconstruct_lmi_similarity(
    root_dir: Path,
    raw: dict[str, np.ndarray],
    mi_lnc_train: np.ndarray,
    dataset: str = "dataset1",
) -> dict[str, np.ndarray]:
    dis_sem = _load_dis_sem(root_dir, dataset)

    mi_gip_from_md = calculate_gaussian_kernel_sim(raw["mi_dis"])
    mi_gip_from_ml = calculate_gaussian_kernel_sim(mi_lnc_train)
    mi_functional_sim = get_rna_functional_sim(raw["mi_dis"].shape[0], dis_sem, raw["mi_dis"])
    mi_sim_fold = rna_fusion_sim(mi_gip_from_md, mi_gip_from_ml, mi_functional_sim)

    lnc_gip_from_ld = calculate_gaussian_kernel_sim(raw["lnc_dis"])
    lnc_gip_from_lm = calculate_gaussian_kernel_sim(mi_lnc_train.T)
    lnc_functional_sim = get_rna_functional_sim(raw["lnc_dis"].shape[0], dis_sem, raw["lnc_dis"])
    lnc_sim_fold = rna_fusion_sim(lnc_gip_from_ld, lnc_gip_from_lm, lnc_functional_sim)

    dis_gip_from_ld = calculate_gaussian_kernel_sim(raw["lnc_dis"].T)
    dis_gip_from_md = calculate_gaussian_kernel_sim(raw["mi_dis"].T)
    dis_sim_fold = dis_fusion_sim(dis_gip_from_ld, dis_gip_from_md, dis_sem)

    return {
        "mi_sim": mi_sim_fold.astype(np.float32),
        "lnc_sim": lnc_sim_fold.astype(np.float32),
        "dis_sim": dis_sim_fold.astype(np.float32),
        "mi_functional_sim": mi_functional_sim.astype(np.float32),
        "lnc_functional_sim": lnc_functional_sim.astype(np.float32),
        "dis_sem": dis_sem.astype(np.float32),
        "mi_gip_from_md": mi_gip_from_md.astype(np.float32),
        "mi_gip_from_ml": mi_gip_from_ml.astype(np.float32),
        "lnc_gip_from_ld": lnc_gip_from_ld.astype(np.float32),
        "lnc_gip_from_lm": lnc_gip_from_lm.astype(np.float32),
        "dis_gip_from_ld": dis_gip_from_ld.astype(np.float32),
        "dis_gip_from_md": dis_gip_from_md.astype(np.float32),
    }


def reconstruct_dataset1_lmi_similarity(root_dir: Path, raw: dict[str, np.ndarray], mi_lnc_train: np.ndarray) -> dict[str, np.ndarray]:
    return reconstruct_lmi_similarity(root_dir, raw, mi_lnc_train, dataset="dataset1")
