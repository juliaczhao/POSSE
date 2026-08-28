"""Per-gene expression clipping values.

Derives the value at which each gene's expression is clipped, from the top
expressing cells, so that outliers do not dominate the dependence estimates.

Reads
-----
    <working_dir>/anndata_files/raw_adata_chunk_*.h5ad

Writes
------
    <working_dir>/supp_material/clip_vals.npy
    <working_dir>/supp_material/clip_zscores.npy
"""

import anndata
import numpy as np
import torch
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pipeline_steps.utils as utils
import pandas as pd


_module_logger = logging.getLogger(__name__)


def _log(*messages, logger=None, level='info'):
    """Emit a message through the given logger, or this module's logger."""
    text = " ".join(str(m) for m in messages)
    getattr(logger or _module_logger, level)(text)

def count_num_cells(files):
    """Total number of cells across the given files."""
    total = 0
    for fname in tqdm(files):
        adata = anndata.read_h5ad(fname)
        n = adata.n_obs
        total += n
    gene_names = adata.var_names.tolist()
    return total, gene_names


@torch.no_grad()
def top_k_zscores(files, means, std_devs, top_k, gene_set=None, cells=None, n_chunk_size=100000):
    """
    means, std_devs: CUDA tensors (d,)
    Returns lists length d with up to top_k z-scores and corresponding normalized values.
    """
    device = means.device
    d = int(means.shape[0])
    safe_std = torch.where(std_devs == 0, torch.ones_like(std_devs), std_devs)

    global_z = torch.full((top_k, d), float("-inf"), device=device)
    global_x = torch.zeros((top_k, d), device=device)

    for fname in tqdm(files):
        adata = anndata.read_h5ad(fname)
        if gene_set is not None:
            adata = adata[:, gene_set]
        if cells is not None:
            use = list(set(cells).intersection(set(adata.obs.index)))
            if not use:
                continue
            adata = adata[use]

        N = adata.n_obs
        for start in range(0, N, n_chunk_size):
            stop = min(N, start + n_chunk_size)
            X = adata.X[start:stop].toarray().astype(np.float32, copy=False)
            X = torch.from_numpy(X).to(device, non_blocking=True)

            rs = X.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            X = X / rs

            Z = (X - means) / safe_std

            k_here = min(top_k, Z.shape[0])
            vals, idxs = torch.topk(Z, k=k_here, dim=0, largest=True, sorted=False)
            orig = torch.gather(X, 0, idxs)

            all_z = torch.cat([global_z, vals], dim=0)
            all_x = torch.cat([global_x, orig], dim=0)

            k_merge = min(top_k, all_z.shape[0])
            merged_vals, merged_idx = torch.topk(all_z, k=k_merge, dim=0, largest=True, sorted=True)
            global_z = merged_vals
            global_x = torch.gather(all_x, 0, merged_idx)

            del X, Z, vals, idxs, orig, all_z, all_x, merged_vals, merged_idx
            torch.cuda.synchronize()

    out_z, out_x = [], []
    finite_mask = torch.isfinite(global_z)
    gz, gx = global_z, global_x
    for j in tqdm(range(d)):
        m = finite_mask[:, j]
        out_z.append(gz[m, j].detach().cpu().tolist())
        out_x.append(gx[m, j].detach().cpu().tolist())
    return out_z, out_x


def compute_clipping_indices(topk_zscores_per_gene, topk_origvals_per_gene, COOCCUR_THRESHOLD):
    """
    For each gene, compute the clipping index and corresponding UMI value using point_of_max_curvature.
    Only process genes with at least one nonzero value in topk_origvals_per_gene.
    Returns:
        clip_indices: list of (idx, umi) tuples for each gene (None if not computed)
    """
    clip_indices = []
    nonzero_gene_vals = []
    for zscores, origvals in tqdm(zip(topk_zscores_per_gene, topk_origvals_per_gene), total=len(topk_zscores_per_gene), desc="Clipping indices"):
        origvals_arr = np.asarray(origvals)
        r = np.count_nonzero(origvals_arr)
        nonzero_gene_vals.append(r)
        if r > COOCCUR_THRESHOLD:
            y = np.asarray(zscores)[:r]
            x = np.arange(1, r + 1)
            idx, _ = utils.point_of_max_curvature(x, y, normalize=True, trim=0.1)
            umi = origvals_arr[idx]
            clip_indices.append((idx, umi, zscores[idx]))
        else:
            fallback_umi = float(np.max(origvals_arr)) if r > 0 else 0.0
            clip_indices.append((None, fallback_umi, 0.))
    return clip_indices, nonzero_gene_vals

def _save_consolidated_diagnostics(nonzero_gene_vals, clip_indices, total_cells, clip_values, zscore_thresholds, output_png_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

    # Proportion of cells clipped per gene (approximate, based on top-k elbow)
    if total_cells and total_cells > 0:
        clipped_counts = []
        for (idx, _, _), r in zip(clip_indices, nonzero_gene_vals):
            if idx is None:
                clipped_counts.append(0)
            else:
                clipped_counts.append(max(0, int(r) - (int(idx) + 1)))
        proportions = np.asarray(clipped_counts, dtype=float) / float(total_cells)
    else:
        proportions = np.zeros(len(nonzero_gene_vals), dtype=float)

    axes[0].hist(proportions, bins=100, log=True)
    axes[0].set_title('Proportion clipped per gene')
    axes[0].set_xlabel('Proportion of cells (clipped)')
    axes[0].set_ylabel('Count (log)')

    axes[1].hist(clip_values, bins=100, log=True)
    axes[1].set_title('Clip values (UMI) for clipped genes')
    axes[1].set_xlabel('Clip value (UMI)')
    axes[1].set_ylabel('Count (log)')

    axes[2].hist(zscore_thresholds, bins=100)
    axes[2].set_title('Z-score threshold at elbow (clipped genes)')
    axes[2].set_xlabel('Elbow z-score')
    axes[2].set_ylabel('Count')

    fig.suptitle('Clipping diagnostics', fontsize=14)
    fig.savefig(output_png_path, dpi=200)
    plt.close(fig)

def _compute_clip_core(files, top_k_proportion, COOCCUR_THRESHOLD, logger=None, return_intermediates=False):
    total_cells, gene_names = count_num_cells(files)
    _log("TOTAL NUMBER OF CELLS: ", total_cells, logger=logger)

    top_k = int(top_k_proportion / 100. * total_cells)
    _log("TOP K CUTOFF: ", top_k, logger=logger)

    if top_k < COOCCUR_THRESHOLD:
        if total_cells < COOCCUR_THRESHOLD:
            _log(
                "Skipping clipping: fewer total cells than COOCCUR_THRESHOLD",
                f"total_cells={total_cells} threshold={COOCCUR_THRESHOLD}",
                logger=logger,
            )
            clip_vals = np.ones(len(gene_names), dtype=float)
            results = {
                'skipped': True,
                'gene_names': gene_names,
                'clip_vals': clip_vals,
                'clip_z_scores': np.zeros(len(gene_names), dtype=float),
                'nonzero_gene_vals': [0] * len(gene_names),
                'clip_indices': [(None, 1.0, 0.0)] * len(gene_names),
                'top_k': top_k,
                'total_cells': total_cells,
            }
            if return_intermediates:
                results.update({
                    'orig_means': None,
                    'orig_stddevs': None,
                    'topk_zscores_per_gene': None,
                    'topk_origvals_per_gene': None,
                })
            return results
        _log(
            f"top_k below COOCCUR_THRESHOLD; flooring top_k to {COOCCUR_THRESHOLD}",
            f"was={top_k}",
            logger=logger,
        )
        top_k = COOCCUR_THRESHOLD

    _log("Computing Gene Means", logger=logger)
    orig_means = utils.compute_running_mean(files, gene_set=None, cells=None)
    _log("Computing Gene Stddevs", logger=logger)
    orig_stddevs = utils.compute_running_stddev(files, orig_means, gene_set=None, cells=None)

    topk_zscores_per_gene, topk_origvals_per_gene = top_k_zscores(
        files, orig_means, orig_stddevs, top_k, gene_set=None, cells=None, n_chunk_size=100000
    )

    clip_indices, nonzero_gene_vals = compute_clipping_indices(
        topk_zscores_per_gene, topk_origvals_per_gene, COOCCUR_THRESHOLD
    )

    clip_vals = np.array([val for idx, val, zscore in clip_indices])
    clip_z_scores = np.array([z for idx, _, z in clip_indices])

    results = {
        'skipped': False,
        'gene_names': gene_names,
        'clip_vals': clip_vals,
        'clip_z_scores': clip_z_scores,
        'nonzero_gene_vals': nonzero_gene_vals,
        'clip_indices': clip_indices,
        'top_k': top_k,
        'total_cells': total_cells,
    }

    if return_intermediates:
        results.update({
            'orig_means': orig_means,
            'orig_stddevs': orig_stddevs,
            'topk_zscores_per_gene': topk_zscores_per_gene,
            'topk_origvals_per_gene': topk_origvals_per_gene,
        })

    return results

def clip_outliers_return(files, top_k_proportion, COOCCUR_THRESHOLD, logger=None, plots_output_dir=None):
    """Compute per-gene clipping values for an explicit list of files.

    Variant of clip_outliers that takes the chunk files directly instead of
    discovering them under a working directory.
    """
    results = _compute_clip_core(
        files,
        top_k_proportion=top_k_proportion,
        COOCCUR_THRESHOLD=COOCCUR_THRESHOLD,
        logger=logger,
        return_intermediates=False,
    )

    gene_names = results['gene_names']
    clip_vals = results['clip_vals']

    if results['skipped']:
        return pd.Series(dtype=float)

    if plots_output_dir is not None and not results['skipped']:
        os.makedirs(plots_output_dir, exist_ok=True)
        clip_values = [val for idx, val, z in results['clip_indices'] if idx is not None]
        zscore_thresholds = [z for idx, _, z in results['clip_indices'] if idx is not None]
        _save_consolidated_diagnostics(
            results['nonzero_gene_vals'], results['clip_indices'], results['total_cells'], clip_values, zscore_thresholds,
            os.path.join(plots_output_dir, 'clipping_diagnostics.png')
        )
        np.save(os.path.join(plots_output_dir, 'clip_vals.npy'), clip_vals)
        np.save(os.path.join(plots_output_dir, 'clip_zscores.npy'), results['clip_z_scores'])

    return pd.Series(clip_vals, index=gene_names)

def clip_outliers(dir_name, output_dir, top_k_proportion, COOCCUR_THRESHOLD):
    """Derive the ceiling applied to each gene's expression in later steps.

    A few very high cells would otherwise dominate a gene's dependence estimates,
    so the value at the top-k cell becomes that gene's clipping point.
    """
    files = []
    # TODO: sort these paths so the order is filesystem-independent.
    for root, dirs, file in os.walk(dir_name):
        for f in file:
            if f.startswith("raw_adata_chunk_") and f.endswith(".h5ad"):
                files.append(os.path.join(root, f))

    results = _compute_clip_core(
        files,
        top_k_proportion=top_k_proportion,
        COOCCUR_THRESHOLD=COOCCUR_THRESHOLD,
        logger=None,
        return_intermediates=True,
    )

    if results['skipped']:
        _module_logger.warning(f"Skipping clipping due to small top_k={results['top_k']} (< threshold {COOCCUR_THRESHOLD})")
        supp_material_dir = output_dir
        os.makedirs(supp_material_dir, exist_ok=True)
        np.save(os.path.join(supp_material_dir, 'clip_vals.npy'), results['clip_vals'])
        np.save(os.path.join(supp_material_dir, 'clip_zscores.npy'), results['clip_z_scores'])
        return

    supp_material_dir = output_dir
    os.makedirs(supp_material_dir, exist_ok=True)

    if results['orig_means'] is not None:
        np.save(os.path.join(supp_material_dir, 'orig_means.npy'), results['orig_means'].cpu().numpy())
    if results['orig_stddevs'] is not None:
        np.save(os.path.join(supp_material_dir, 'orig_stddevs.npy'), results['orig_stddevs'].cpu().numpy())

    if results['topk_zscores_per_gene'] is not None:
        np.save(os.path.join(supp_material_dir, 'topk_zscores_per_gene.npy'), np.array(results['topk_zscores_per_gene'], dtype=object))
    if results['topk_origvals_per_gene'] is not None:
        np.save(os.path.join(supp_material_dir, 'topk_origvals_per_gene.npy'), np.array(results['topk_origvals_per_gene'], dtype=object))

    np.save(os.path.join(supp_material_dir, 'clip_vals.npy'), results['clip_vals'])
    np.save(os.path.join(supp_material_dir, 'clip_zscores.npy'), results['clip_z_scores'])

    plots_dir = os.path.join(supp_material_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    clip_values = [val for idx, val, z in results['clip_indices'] if idx is not None]
    zscore_thresholds = [z for idx, _, z in results['clip_indices'] if idx is not None]
    _save_consolidated_diagnostics(
        results['nonzero_gene_vals'], results['clip_indices'], results['total_cells'], clip_values, zscore_thresholds,
        os.path.join(plots_dir, 'clipping_diagnostics.png')
    )
