"""Per-cell program activity.

Scores every cell against every program two ways: wePAS, the projection onto the
program's gene loadings, and sePAS, a co-expression statistic over the program's
genes. Both are computed from a single pass over the cell chunks.

Reads
-----
    <working_dir>/clique_based_programs/<programs file>
    <working_dir>/ids_cooccur/{ids,correlation}_matrix.hkl
    <working_dir>/anndata_files/raw_adata_chunk_*.h5ad
    <working_dir>/supp_material/clip_vals.npy

Writes
------
    <working_dir>/program_activity/wePAS.h5ad
    <working_dir>/program_activity/sePAS.h5ad
    <working_dir>/clique_based_programs/programs_with_loadings.json
"""

import json
import logging
import anndata as ad
import numpy as np
import pipeline_steps.utils as utils
from pipeline_steps.utils import load_programs
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from scipy.sparse.linalg import eigsh
import glob
import hickle as hkl
import time
from scipy import sparse

ad.settings.allow_write_nullable_strings = True


logger = logging.getLogger(__name__)

MIN_MAX_SCALE_FACTOR = 8

def compute_program_loadings(programs, ids_df, corr_df, working_dir, loading_type=None):
    """Weight each gene within its program by the top eigenvector of the signed IDS submatrix.

    These loadings set how strongly each gene contributes to its program's per-cell
    activity, and are saved so later steps can identify the positively loaded genes.

    Returns a list of (name, genes, loadings) tuples, one per program.
    """

    program_loadings_list = []
    program_names_list = []
    program_genes_list = []

    program_var_explained_list = []
    program_var_explained_names = []

    logger.info("Computing loadings from %s (options: corr_eig, ids_eig)", loading_type)
    for i, program_genes in enumerate(tqdm(programs, desc="Computing program activities")):
        program_genes_in_df = [g for g in program_genes if g in ids_df.index and g in corr_df.index]
        
        if len(program_genes_in_df) == 0:
            raise ValueError(f"None of the program genes for program {i} are present in the matrices")

        if loading_type == 'corr_eig':
            C = corr_df.loc[program_genes_in_df, program_genes_in_df].values
        elif loading_type == 'ids_eig':
            S = corr_df.loc[program_genes_in_df, program_genes_in_df].values
            C = ids_df.loc[program_genes_in_df, program_genes_in_df].values
            C = C * np.sign(S)
        else:
            raise ValueError(f"Unknown loading_type: {loading_type}")

        k = min(C.shape[0], 10)
        if k >= C.shape[0]:
            vals, vecs = np.linalg.eigh(C)
            idx = np.argsort(vals)[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]
        else:
            # Fixed start vector keeps the eigendecomposition reproducible.
            v0 = np.ones(C.shape[0]) / np.sqrt(C.shape[0])
            vals, vecs = eigsh(C, k=k, which='LA', v0=v0)
            idx = np.argsort(vals)[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]

        total = np.sum(vals)
        if total == 0:
            var_explained = np.zeros_like(vals)
        else:
            var_explained = vals / total
        program_var_explained_list.append(var_explained)
        program_var_explained_names.append(f"Program {i}")

        top_eigenvector = vecs[:, 0]
        if np.sum(top_eigenvector > 0) < (len(top_eigenvector) / 2):
            top_eigenvector = -top_eigenvector
        program_loadings_list.append(top_eigenvector)
        program_names_list.append(f"Program {i}")
        program_genes_list.append(program_genes_in_df)

    n_programs = len(program_loadings_list)
    fig, axes = plt.subplots(n_programs, 1, figsize=(6, 1.5 * n_programs), sharex=True)
    if n_programs == 1:
        axes = [axes]
    for idx, (loadings, name) in enumerate(zip(program_loadings_list, program_names_list)):
        ax = axes[idx]
        if loadings.size > 0:
            ax.hist(loadings, bins=20, density=True, alpha=0.7, color='C0')
        n_genes = loadings.size
        ax.set_ylabel(f"{name}\n({n_genes} genes)")
        ax.set_yticks([])
    axes[-1].set_xlabel("Program PC1 gene loadings")
    plt.tight_layout()
    save_path = os.path.join(working_dir, 'program_loadings_distribution.png')
    plt.savefig(save_path, bbox_inches="tight")

    if len(program_var_explained_list) > 0:
        n_programs = len(program_var_explained_list)
        fig, axes = plt.subplots(n_programs, 1, figsize=(6, 2 + 1.5 * n_programs), sharex=True)
        if n_programs == 1:
            axes = [axes]
        for idx, (var_exp, name) in enumerate(zip(program_var_explained_list, program_var_explained_names)):
            ax = axes[idx]
            ax.plot(np.arange(1, len(var_exp) + 1), var_exp, marker='o', color='C0')
            ax.set_ylabel("Variance\nexplained")
            ax.set_title(f"{name} variance explained decay")
            ax.set_xticks(np.arange(1, len(var_exp) + 1))
        axes[-1].set_xlabel("Component")
        plt.tight_layout()
        save_path = os.path.join(working_dir, 'program_variance_explained_decay.png')
        plt.savefig(save_path, bbox_inches="tight")

    program_gene_loadings = []
    for name, genes, loadings in zip(program_names_list, program_genes_list, program_loadings_list):
        program_gene_loadings.append((name, genes, loadings))

    return program_gene_loadings


_SCALE_TRANSFORMS = ("iqr", "clipped_robust_zscore", "clipped_raw_activity", "clipped_zscore",
                     "curvature_clip", "subpop_trim", "robust_zscore", "robust_zscore_trimmed",
                     "robust_zscore_symlog", "robust_zscore_symsqrt", "min_max", "quantile")


def scale_program_activity(adata, programs, program_prefix="new_program_", transform=None):
    """Put every program's activities on a comparable scale.

    Raw activities depend on a program's size and loading magnitudes, so they are
    not comparable across programs. The scaled values are what the report plots,
    the Leiden clustering and the program-program correlations use; the unscaled
    values are kept alongside them.
    """
    program_cols = [col for col in adata.var_names if col.startswith(program_prefix)]
    if not program_cols:
        raise ValueError(f"No program activity columns found in adata.var_names with prefix '{program_prefix}'")

    X_valid = adata.X
    logger.info("Scaling program vectors: %s (options: %s)", transform, ", ".join(_SCALE_TRANSFORMS))

    if transform == "iqr":
        Q1 = np.percentile(X_valid, 25, axis=0)
        Q3 = np.percentile(X_valid, 75, axis=0)
        IQR = Q3 - Q1
        IQR[IQR == 0] = 1
        X_valid = (X_valid - Q1) / IQR
    elif transform == "clipped_robust_zscore":
        Q1 = np.percentile(X_valid, 25, axis=0)
        Q3 = np.percentile(X_valid, 75, axis=0)
        med = np.percentile(X_valid, 50, axis=0)
        IQR = Q3 - Q1
        IQR[IQR == 0] = 1
        X_valid = (X_valid - med) / (IQR / 1.349)  # 1.349 ≈ IQR of standard normal
        X_valid = np.clip(X_valid, -5, 5)
    elif transform == "clipped_raw_activity":
        X_valid = np.clip(X_valid, -8, 8)
    elif transform == "clipped_zscore":
        mean = np.mean(X_valid, axis=0)
        std = np.std(X_valid, axis=0)
        std[std == 0] = 1
        X_valid = (X_valid - mean) / std
        X_valid = np.clip(X_valid, -5, 5)

    elif transform == "curvature_clip":

        mean = np.mean(X_valid, axis=0)
        std = np.std(X_valid, axis=0)
        logger.debug("Means: %s", mean)
        std[std == 0] = 1
        zscores = (X_valid - mean) / std

        # For each column, clip the max z-scores at the z-score of the elbow (max curvature)
        zscores_clipped = zscores.copy()
        n_cols = zscores.shape[1]
        for i in range(n_cols):
            col_z = zscores[:, i]

            k = 1
            n = len(col_z)
            top_n = max(1, int(np.ceil(n * k / 100)))
            top_indices = np.argsort(col_z)[::-1][:top_n]
            if len(top_indices) > 2:
                y = col_z[top_indices]

                x = np.arange(1, len(y) + 1)
                idx, _ = utils.point_of_max_curvature(x, y, normalize=True, trim=0.0)
                if 0 <= idx < len(y):
                    elbow_z = y[idx]
                    # Clip all z-scores in this column at elbow_z (for positive tail)
                    col_z = np.where(col_z > elbow_z, elbow_z, col_z)
            zscores_clipped[:, i] = col_z
        X_valid = zscores_clipped

        logger.debug("Column maxima after curvature_clip: %s", np.max(X_valid, axis=0))

        scale_factors = np.array([np.sqrt(len(genes)) if len(genes) > 0 else 1 for genes in programs])
        X_valid = X_valid * scale_factors

    elif transform == "subpop_trim":
        k = 100
        X_trimmed = X_valid.copy()
        n_rows, n_cols = X_trimmed.shape
        for i in range(n_cols):
            col = X_trimmed[:, i]
            if len(col) > k:
                kth_val = np.partition(col, -k)[-k]
                col = np.minimum(col, kth_val)
            min_val = np.min(col)
            max_val = np.max(col)
            denom = max_val - min_val if max_val != min_val else 1
            X_trimmed[:, i] = (col - min_val) / denom
        X_valid = X_trimmed
    elif transform == "robust_zscore":
        Q1 = np.percentile(X_valid, 25, axis=0)
        Q3 = np.percentile(X_valid, 75, axis=0)
        med = np.percentile(X_valid, 50, axis=0)
        IQR = Q3 - Q1
        IQR[IQR == 0] = 1
        X_valid = (X_valid - med) / (IQR / 1.349)  # 1.349 ≈ IQR of standard normal
    elif transform == "robust_zscore_trimmed":
        med = np.median(X_valid, axis=0)
        mad = np.median(np.abs(X_valid - med), axis=0)
        mad[mad == 0] = 1
        X_valid = (X_valid - med) / (mad * 1.4826)  # 1.4826 ≈ scaling factor for MAD to std
        X_valid = np.clip(X_valid, -8, 8)
    elif transform == "robust_zscore_symlog":
        Q1 = np.percentile(X_valid, 25, axis=0)
        Q3 = np.percentile(X_valid, 75, axis=0)
        med = np.percentile(X_valid, 50, axis=0)
        IQR = Q3 - Q1
        IQR[IQR == 0] = 1
        X_valid = (X_valid - med) / (IQR / 1.349)  # 1.349 ≈ IQR of standard normal
        X_valid = np.sign(X_valid) * np.log1p(np.abs(X_valid))        
    elif transform == "robust_zscore_symsqrt":
        Q1 = np.percentile(X_valid, 25, axis=0)
        Q3 = np.percentile(X_valid, 75, axis=0)
        med = np.percentile(X_valid, 50, axis=0)
        IQR = Q3 - Q1
        IQR[IQR == 0] = 1
        X_valid = (X_valid - med) / (IQR / 1.349)  # 1.349 ≈ IQR of standard normal
        X_valid = np.sign(X_valid) * np.sqrt(np.abs(X_valid))
    elif transform == "min_max":
        min_vals = np.min(X_valid, axis=0)
        max_vals = np.max(X_valid, axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1
        X_valid = (X_valid - min_vals) / denom
    elif transform == "quantile":    
        X_quantiled = np.zeros_like(X_valid)
        n_rows, n_cols = X_valid.shape

        for i in range(n_cols):
            col = X_valid[:, i]
            nonzero_mask = col != 0
            nonzero_vals = col[nonzero_mask]
            if nonzero_vals.size == 0:
                continue
            order = np.argsort(nonzero_vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(nonzero_vals) + 1)
            quantiles = ranks / (len(nonzero_vals) + 1)
            col_quantiled = np.zeros_like(col, dtype=float)
            col_quantiled[nonzero_mask] = quantiles
            X_quantiled[:, i] = col_quantiled
        X_valid = X_quantiled

    X_raw = adata.X
    adata.obsm['program_activity_unscaled'] = X_raw
    adata.obsm['program_activity_scaled'] = X_valid

    return adata


def write_programs_with_loadings_to_json(program_gene_loadings, output_dir):
    """Save each program's genes with their loadings.

    Written before the activity scores are computed, because the co-expression
    score reads it to restrict each program to its positively loaded genes.
    """
    os.makedirs(output_dir, exist_ok=True)
    programs_dict = {}
    for idx, (program_name, gene_list, loadings) in enumerate(program_gene_loadings):
        gene_loading_pairs = sorted(zip(gene_list, loadings), key=lambda x: x[0])
        sorted_genes = [g for g, _ in gene_loading_pairs]
        loadings_dict = {g: float(l) for g, l in gene_loading_pairs}
        programs_dict[str(idx)] = {
            "program": sorted_genes,
            "loadings": loadings_dict
        }
    output_path = os.path.join(output_dir, f"programs_with_loadings.json")
    with open(output_path, "w") as f:
        json.dump(programs_dict, f, indent=2)

def write_program_activity_adata(adata, output_dir, program_prefix="new_program_"):
    """Write wePAS.h5ad, the input to the plotting and summary steps.

    Stores the scaled activities in X, both scaled and unscaled in obsm, and a
    per-program Otsu on/off call (computed on the scaled activities).
    """
    if 'program_activity_scaled' not in adata.obsm:
        raise ValueError("program_activity_scaled not found in adata.obsm. Make sure scale_program_activity was called first.")

    X_scaled = np.asarray(adata.obsm['program_activity_scaled'])
    n_programs = X_scaled.shape[1]
    program_cols = [f'{program_prefix}{i}_activity_scaled' for i in range(n_programs)]

    is_high, otsu_thr = compute_otsu_high(X_scaled, nonzero_only=False)

    program_adata = ad.AnnData(
        X=X_scaled,
        obs=adata.obs.copy(),
        var=pd.DataFrame({"otsu_thr": otsu_thr, "n_high": is_high.sum(0).astype(np.int64)},
                         index=program_cols),
    )

    if 'program_activity_unscaled' in adata.obsm:
        program_adata.obsm['program_activity_unscaled'] = adata.obsm['program_activity_unscaled']
    program_adata.obsm['program_activity_scaled'] = X_scaled
    program_adata.obsm['program_activity_high'] = is_high

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "wePAS.h5ad")
    program_adata.write(out_path)
    logger.info(f"Wrote program activity AnnData to {out_path}")


def _load_hkl(working_dir, filename):
    hkl_path = os.path.join(working_dir, 'ids_cooccur', filename)
    if not os.path.exists(hkl_path):
        raise FileNotFoundError(f"Expected matrix at {hkl_path}")
    df = hkl.load(hkl_path)
    logger.info(f"Loaded matrix from {hkl_path}")
    return df

def get_ids_df(working_dir):
    """Load the IDS matrix written by the dependence step."""
    return _load_hkl(working_dir, 'ids_matrix.hkl')

def get_correlation_df(working_dir):
    """Load the correlation matrix, used to sign the IDS values when computing loadings."""
    return _load_hkl(working_dir, 'correlation_matrix.hkl')

def get_cooccurrence_df(working_dir):
    """Load the cooccurrence matrix written by the dependence step."""
    return _load_hkl(working_dir, 'cooccurrence_matrix.hkl')


# Synchronized Program Activity Score (sePAS): a per-(cell, program) co-expression
# statistic aggregated as a truncated -log10 score and Otsu-binarized.

# Per-gene evidence cap: only admissions at p <= P_MAX_PCT contribute, which
# keeps the null mode a delta at 0 so Otsu sees a bimodal score.
P_MAX_PCT = 5.0

# Grid-based percentile tables, superseded by compute_exact_logp:
# PERCENTILES = np.unique(np.concatenate([
#     np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]),
#     np.arange(1, 101, 1, dtype=float),
# ])).astype(np.float64)
# P = len(PERCENTILES)
# LOG10_PCT_AUG = -np.log10(np.concatenate([PERCENTILES, [100.0]]) / 100.0)
# LOG10_PCT_TRUNC = LOG10_PCT_AUG.copy()
# LOG10_PCT_TRUNC[int(np.searchsorted(PERCENTILES, P_MAX_PCT)):] = 0.0


def otsu_threshold(x, nbins=256, nonzero_only=False):
    """Otsu's between-class-variance threshold.

    `nonzero_only=True` restricts the histogram to score>0 cells, which is
    needed when there is a large mass at 0: it picks the meaningful valley
    above 0 rather than the 0/nonzero valley.
    """
    x = np.asarray(x).ravel()
    if nonzero_only:
        x = x[x > 0]
        if x.size == 0:
            return 0.0
    if np.unique(x).size < 2:
        return float(x[0])
    hist, edges = np.histogram(x, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(np.float64) / hist.sum()
    cum_w = np.cumsum(w)
    cum_wm = np.cumsum(w * centers)
    denom = cum_w * (1 - cum_w)
    sigma_b = np.where(denom > 0,
                       (cum_wm[-1] * cum_w - cum_wm) ** 2 / np.where(denom > 0, denom, 1),
                       0.0)
    return float(centers[int(np.argmax(sigma_b))])


def compute_otsu_high(X, nonzero_only=False):
    """Per-program Otsu on/off call for a cells-by-programs matrix.

    Both wePAS and sePAS call this before saving to mark, per program, the cells
    above that program's Otsu threshold. Returns (is_high, thresholds): a boolean
    cells-by-programs matrix and the per-program thresholds. Use nonzero_only=True
    for scores with a large mass at zero (sePAS).
    """
    X = np.asarray(X)
    n_cells, n_programs = X.shape
    is_high = np.empty((n_cells, n_programs), dtype=bool)
    thresholds = np.empty(n_programs, dtype=np.float64)
    for j in range(n_programs):
        thr = otsu_threshold(X[:, j], nonzero_only=nonzero_only)
        thresholds[j] = thr
        is_high[:, j] = X[:, j] > thr
    return is_high, thresholds


def compute_exact_logp(Ep, p_max=P_MAX_PCT / 100.0):
    """Per-cell, per-gene -log10 of the empirical upper-tail fraction.

    Scores each admission from its exact upper-tail fraction rather than snapping it
    to a fixed percentile grid; admissions above p_max contribute zero.
    """
    n_cells, n_genes = Ep.shape
    out = np.empty((n_cells, n_genes), dtype=np.float64)
    for j in range(n_genes):
        col = Ep[:, j]
        vals_asc, counts = np.unique(col, return_counts=True)
        frac_ge = (n_cells - np.concatenate([[0], np.cumsum(counts)[:-1]])) / n_cells
        logp = np.where(frac_ge <= p_max, -np.log10(frac_ge), 0.0)
        out[:, j] = logp[np.searchsorted(vals_asc, col, side="left")]
    return out


# Grid-based alternative to compute_exact_logp (snaps admissions to PERCENTILES):
# def compute_pmin_idx(Ep):
#     """Per-cell, per-gene admission-percentile bucket index into PERCENTILES."""
#     N, G = Ep.shape
#     pmin_idx = np.full((N, G), P, dtype=np.int32)
#     pct = PERCENTILES / 100.0
#     for j in range(G):
#         col = Ep[:, j]
#         vals_asc, counts = np.unique(col, return_counts=True)
#         cum_at_or_above = N - np.concatenate([[0], np.cumsum(counts)[:-1]])
#         frac_ge = cum_at_or_above.astype(np.float64) / N
#         p_admit = np.searchsorted(pct, frac_ge, side="left").astype(np.int32)
#         pmin_idx[:, j] = p_admit[np.searchsorted(vals_asc, col, side="left")]
#     return pmin_idx


def compute_program_activity(working_dir,
                             program_path_dir="clique_based_programs",
                             program_path_name="programs_after_grow_merge.json",
                             loading_type='ids_eig',
                             program_vector_scaling='robust_zscore_trimmed',
                             weighting="positive"):
    """Read the cell chunks once, then compute and write both activity scores.

    Writes <working_dir>/program_activity/wePAS.h5ad and sePAS.h5ad.
    """
    anndata_dir  = os.path.join(working_dir, 'anndata_files')
    out_dir      = os.path.join(working_dir, 'program_activity')
    programs_dir = os.path.join(working_dir, program_path_dir)
    qc_dir       = os.path.join(working_dir, 'supp_material')

    upper_bound = torch.from_numpy(np.load(os.path.join(qc_dir, 'clip_vals.npy'))).cuda()

    programs = load_programs(os.path.join(programs_dir, program_path_name))
    logger.info("Computing program loadings")
    loadings = compute_program_loadings(programs, get_ids_df(working_dir),
                                        get_correlation_df(working_dir), qc_dir,
                                        loading_type=loading_type)
    write_programs_with_loadings_to_json(loadings, programs_dir)

    programs_json = os.path.join(programs_dir, "programs_with_loadings.json")
    with open(programs_json) as f:
        sepas_genes = sorted({g for v in json.load(f).values() for g in v["program"]})

    obs, activity, counts, library = _read_chunks(anndata_dir, loadings, upper_bound, sepas_genes)

    logger.info("Scoring wePAS")
    compute_wePAS(activity, obs, programs, out_dir, program_vector_scaling)
    logger.info("Scoring sePAS")
    compute_sePAS(counts, library, obs, programs_json, out_dir, weighting=weighting)


def _chunk_files(anndata_dir):
    """Cell chunks in numeric order, so both scores see the same row order."""
    files = sorted(glob.glob(os.path.join(anndata_dir, "raw_adata_chunk_*.h5ad")),
                   key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
    if not files:
        raise FileNotFoundError(f"no cell chunks in {anndata_dir}")
    return files


def _read_chunks(anndata_dir, loadings, upper_bound, sepas_genes):
    """Single pass over the cell chunks, collecting what both scores need.

    Returns (obs, activity, counts, library):
        activity : cells x programs, loading-weighted projection (wePAS input)
        counts   : {gene: raw counts across all cells} (sePAS input)
        library  : per-cell total UMI count (sePAS normalizer)
    """
    program_names = [n for n, _, _ in loadings]
    program_genes = [g for _, g, _ in loadings]
    program_loads = [l for _, _, l in loadings]

    obs_parts, activity_parts, library_parts = [], [], []
    gene_counts = {g: [] for g in sepas_genes}

    for path in tqdm(_chunk_files(anndata_dir), desc="Reading cell chunks"):
        adata = ad.read_h5ad(path)
        obs_parts.append(adata.obs.copy())

        X_raw = adata.X
        var_idx = {v: j for j, v in enumerate(adata.var.index)}
        present = [g for g in sepas_genes if g in var_idx]
        cols = [var_idx[g] for g in present]
        if sparse.issparse(X_raw):
            library_parts.append(np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float32))
            sub = X_raw[:, cols]
            sub = sub.toarray() if sparse.issparse(sub) else sub
        else:
            library_parts.append(X_raw.sum(axis=1).astype(np.float32))
            sub = X_raw[:, cols]
        sub = np.asarray(sub, dtype=np.float32)
        for j, gene in enumerate(present):
            gene_counts[gene].append(sub[:, j])

        var_names = np.array(adata.var_names)
        X = adata.X.toarray()
        X = torch.from_numpy(X).cuda().float()
        X /= torch.sum(X, axis=-1, keepdims=True)
        X.clamp_(max=upper_bound)
        X *= MIN_MAX_SCALE_FACTOR / upper_bound
        X = X.cpu().numpy()

        activities = []
        for name, genes, loads in zip(program_names, program_genes, program_loads):
            gene_mask = np.isin(var_names, genes)
            if not np.any(gene_mask):
                raise ValueError(
                    f"None of the program genes for program '{name}' are present in {path}")
            X_prog = X[:, gene_mask]
            present_genes = var_names[gene_mask]
            idx = [genes.index(g) for g in present_genes if g in genes]
            loads_present = np.array(loads)[idx]
            activities.append((X_prog @ loads_present * 1 / np.sqrt(X_prog.shape[1])).reshape(-1, 1))
        activity_parts.append(np.hstack(activities))
        del X

    obs = pd.concat(obs_parts, axis=0)
    obs.index = obs.index.astype(str)
    counts = {g: (np.concatenate(v) if v else None) for g, v in gene_counts.items()}
    return obs, np.vstack(activity_parts), counts, np.concatenate(library_parts)


def compute_wePAS(activity, obs, programs, out_dir, transform):
    """Score each cell by how strongly it expresses each program, weighted by the loadings.

    Genes contribute in proportion to their loading, so the score reflects a
    program's dominant genes most. Written to wePAS.h5ad for the plots and summaries.
    """
    adata = ad.AnnData(X=activity, obs=obs)
    adata.var_names = [f'new_program_{i}_activity' for i in range(activity.shape[1])]
    adata = scale_program_activity(adata, programs, program_prefix="new_program_",
                                   transform=transform)
    write_program_activity_adata(adata, out_dir, program_prefix="new_program_")


def compute_sePAS(counts, library, obs, programs_json, out_dir, weighting="positive"):
    """Co-expression score per (cell, program) -> sePAS.h5ad.

    weighting : {"positive", "allgenes"}
        positive (default) drops negative-loading genes; allgenes keeps every
        program gene with weight +1.
    """
    if weighting not in ("positive", "allgenes"):
        raise ValueError(f"weighting must be 'positive' or 'allgenes', got {weighting!r}")
    logger.info("sePAS gene weighting: %s (options: positive, allgenes)", weighting)

    with open(programs_json) as f:
        programs = json.load(f)
    pids = sorted(programs, key=int)
    n_cells = len(library)
    n_programs = len(pids)

    score = np.empty((n_cells, n_programs), dtype=np.float32)
    G = np.empty(n_programs, dtype=np.int64)
    G_total = np.empty(n_programs, dtype=np.int64)
    start = time.time()
    for j, pid in enumerate(pids):
        genes = programs[pid]["program"]
        loads = programs[pid].get("loadings", {})

        Ep = np.empty((n_cells, len(genes)), dtype=np.float32)
        for k, gene in enumerate(genes):
            counts_g = counts.get(gene)
            Ep[:, k] = counts_g / library if counts_g is not None else 0.0

        per_gene_logp = compute_exact_logp(Ep)
        if weighting == "positive":
            per_gene_logp = per_gene_logp[:, [loads.get(g, 1.0) >= 0 for g in genes]]

        g = per_gene_logp.shape[1]
        score[:, j] = (per_gene_logp.mean(axis=1).astype(np.float32) if g
                       else np.zeros(n_cells, dtype=np.float32))
        G[j] = g
        G_total[j] = len(genes)

    is_high, otsu_thr = compute_otsu_high(score, nonzero_only=True)
    for j, pid in enumerate(pids):
        logger.info(f"  P{pid:>3s}: G={G[j]:>3d}  Otsu thr={otsu_thr[j]:.3f}  "
              f"high={int(is_high[:, j].sum()):,}/{n_cells:,}")
    logger.info(f"Scored {n_programs} programs in {time.time() - start:.1f}s.")
    return _write_sepas_h5ad(pids, score, is_high, otsu_thr, G, G_total, obs, out_dir)


def _write_sepas_h5ad(pids, score, is_high, otsu_thr, G, G_total, obs, out_dir):
    """Write sePAS.h5ad with the per-program scores and the on/off call for each cell.

    The threshold separating active from inactive cells is chosen per program by
    Otsu's method rather than fixed in advance.
    """
    n_cells, n_programs = score.shape
    if len(obs) != n_cells:
        raise RuntimeError(f"obs row count {len(obs)} != {n_cells} scored cells")

    var = pd.DataFrame(
        {"otsu_thr": otsu_thr, "n_high": is_high.sum(0).astype(np.int64), "G": G, "G_total": G_total},
        index=[f"new_program_{int(p)}_synchronized_score" for p in pids],
    )
    adata = ad.AnnData(X=score, obs=obs, var=var)
    adata.obsm["synchronized_program_activity_unscaled"] = score
    adata.obsm["synchronized_program_activity_high"] = is_high

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sePAS.h5ad")
    adata.write_h5ad(out_path, compression="gzip")
    logger.info(f"  wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB; "
          f"N={n_cells:,} cells x {n_programs} programs)")
    return out_path
