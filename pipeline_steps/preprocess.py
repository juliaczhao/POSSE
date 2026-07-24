"""Quality-control filtering of raw data.

Removes cells by total UMI count, number of expressed genes, and mitochondrial
expression fraction. Thresholds are percentiles computed over all cells.

Reads
-----
    the raw .h5ad input (a file, or a directory of files)

Writes
------
    <output_path>/filtered_<name>.h5ad
    <output_path>/plots/  quality-control distributions
"""

import logging
import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm


logger = logging.getLogger(__name__)

CELL_CHUNK_SIZE = 100_000
MT_GENE_PREFIX = "MT-"


def load_files(raw_adata_path):
    """Return the .h5ad paths at raw_adata_path, which may be a file or a directory."""
    if os.path.isdir(raw_adata_path):
        h5ad_paths = sorted(
            os.path.join(raw_adata_path, fname)
            for fname in os.listdir(raw_adata_path)
            if fname.endswith(".h5ad")
        )
    elif raw_adata_path.endswith(".h5ad") and os.path.isfile(raw_adata_path):
        h5ad_paths = [raw_adata_path]
    else:
        raise ValueError(f"Provided raw_adata_path is neither a directory nor an h5ad file: {raw_adata_path}")

    logger.info("Found %d input file(s)", len(h5ad_paths))
    for path in h5ad_paths:
        logger.info("  %s", path)
    return h5ad_paths


def _iter_chunk_bounds(n_obs, chunk_size):
    for start in range(0, n_obs, chunk_size):
        yield start, min(start + chunk_size, n_obs)


def _compute_chunk_metrics(X_chunk, mt_gene_mask):
    if sparse.issparse(X_chunk):
        row_sums = np.asarray(X_chunk.sum(axis=1)).ravel().astype(np.float32)
        nonzero_genes = X_chunk.getnnz(axis=1).astype(np.int32)
        if mt_gene_mask.any():
            mt_umis = np.asarray(X_chunk[:, mt_gene_mask].sum(axis=1)).ravel().astype(np.float32)
        else:
            mt_umis = np.zeros(X_chunk.shape[0], dtype=np.float32)
    else:
        X_chunk = np.asarray(X_chunk, dtype=np.float32)
        row_sums = X_chunk.sum(axis=1, dtype=np.float32)
        nonzero_genes = np.count_nonzero(X_chunk > 0, axis=1).astype(np.int32)
        if mt_gene_mask.any():
            mt_umis = X_chunk[:, mt_gene_mask].sum(axis=1, dtype=np.float32)
        else:
            mt_umis = np.zeros(X_chunk.shape[0], dtype=np.float32)

    mt_ratio = mt_umis / (row_sums + 1e-10)
    return row_sums, nonzero_genes, mt_ratio.astype(np.float32)


def _collect_global_qc_metrics(files, cell_chunk_size):
    all_row_sums = []
    all_nonzero_genes = []
    all_mt_ratios = []
    total_cells = 0

    for file_path in tqdm(files, desc="Collecting QC metrics"):
        adata = anndata.read_h5ad(file_path)
        mt_gene_mask = np.asarray(adata.var_names.str.startswith(MT_GENE_PREFIX))
        for start, end in _iter_chunk_bounds(adata.n_obs, cell_chunk_size):
            X_chunk = adata.X[start:end]
            row_sums, nonzero_genes, mt_ratio = _compute_chunk_metrics(X_chunk, mt_gene_mask)
            all_row_sums.append(row_sums)
            all_nonzero_genes.append(nonzero_genes)
            all_mt_ratios.append(mt_ratio)
            total_cells += len(row_sums)
        del adata

    if total_cells == 0:
        raise ValueError("No cells found in the provided AnnData input.")

    return (
        np.concatenate(all_row_sums),
        np.concatenate(all_nonzero_genes),
        np.concatenate(all_mt_ratios),
    )


def _plot_distribution(values, plots_dir, filename, xlabel, title, loglog=False):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"{filename}_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    sorted_values = np.sort(values)
    n = len(sorted_values)
    ccdf = 1 - np.arange(1, n + 1) / n

    plt.figure(figsize=(10, 6))
    if loglog:
        plt.loglog(sorted_values, ccdf, linewidth=2)
        plt.ylabel("1 - CDF (Complementary CDF)")
    else:
        plt.plot(sorted_values, ccdf)
        plt.ylabel("1 - CDF")
    plt.xlabel(xlabel)
    plt.title(f"Complementary CDF of {title}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"{filename}_ccdf.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _write_filtered_data(file_path, output_path, cell_chunk_size, umi_threshold, nonzero_threshold, mt_threshold):
    adata = anndata.read_h5ad(file_path)
    written_files = 0
    kept_cells = 0

    mt_gene_mask = np.asarray(adata.var_names.str.startswith(MT_GENE_PREFIX))
    keep_indices = []

    for start, end in _iter_chunk_bounds(adata.n_obs, cell_chunk_size):
        X_chunk = adata.X[start:end]
        row_sums, nonzero_genes, mt_ratio = _compute_chunk_metrics(X_chunk, mt_gene_mask)
        keep_mask = (
            (row_sums <= umi_threshold)
            & (nonzero_genes > nonzero_threshold)
            & (mt_ratio <= mt_threshold)
        )
        if np.any(keep_mask):
            keep_indices.append(np.arange(start, end, dtype=np.int64)[keep_mask])
            kept_cells += int(np.sum(keep_mask))

    if not keep_indices:
        logger.warning("No cells passed filters for %s", file_path)
        return 0, 0

    keep_indices = np.concatenate(keep_indices)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Write single filtered file — chunking is handled downstream by select_cell_types
    filtered_adata = adata[keep_indices].copy()
    output_file = os.path.join(output_path, f"filtered_{base_name}.h5ad")
    filtered_adata.write(output_file)
    written_files = 1
    logger.info(f"Saved filtered data to: {output_file} with shape {filtered_adata.shape}")

    del adata, filtered_adata
    return kept_cells, written_files


def clean_raw_data(
    raw_adata_path,
    umi_threshold=30000,
    nonzero_gene_percentage=1,
    mt_ratio_percentage=90,
    output_path="filtered_data/",
    cell_chunk_size=CELL_CHUNK_SIZE,
):
    """Filter cells by UMI count, expressed genes and mitochondrial fraction, and write the result.

    This is the only quality-control step; every later step reads the cleaned file,
    so cells removed here are absent from the dependence estimates and the programs.
    """
    os.makedirs(output_path, exist_ok=True)
    plots_dir = os.path.join(output_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("Filtering parameters")
    logger.info(f"Raw data path: {raw_adata_path}")
    logger.info(f"UMI threshold: {umi_threshold}")
    logger.info(f"Nonzero gene percentage: {nonzero_gene_percentage}%")
    logger.info(f"MT ratio percentage: {mt_ratio_percentage}%")
    logger.info(f"Cell chunk size: {cell_chunk_size}")

    files = load_files(raw_adata_path)

    row_sums, nonzero_genes, mt_ratios = _collect_global_qc_metrics(files, cell_chunk_size)

    nonzero_threshold = np.percentile(nonzero_genes, nonzero_gene_percentage)
    mt_threshold = np.percentile(mt_ratios, mt_ratio_percentage)

    logger.info(f"Global nonzero gene threshold ({nonzero_gene_percentage}th percentile): {nonzero_threshold}")
    logger.info(f"Global MT ratio threshold ({mt_ratio_percentage}th percentile): {mt_threshold}")

    _plot_distribution(row_sums, plots_dir, "program_sum", "UMIs per Cell", "UMIs per Cell")
    _plot_distribution(
        nonzero_genes,
        plots_dir,
        "nonzero_genes",
        "Nonzero Genes per Cell",
        "Nonzero Genes per Cell",
    )
    _plot_distribution(
        mt_ratios,
        plots_dir,
        "mt_ratio",
        "Mitochondrial Gene Expression Ratio",
        "Mitochondrial Gene Expression Ratio per Cell",
        loglog=True,
    )

    total_cells = len(row_sums)
    fail_umi = np.sum(row_sums > umi_threshold)
    fail_nonzero = np.sum(nonzero_genes <= nonzero_threshold)
    fail_mt = np.sum(mt_ratios > mt_threshold)

    logger.info("Filtering statistics, per criterion")
    logger.info(f"Total cells before filtering: {total_cells}")
    logger.info(f"  UMI > {umi_threshold}:                  {fail_umi} cells removed ({fail_umi / total_cells:.2%})")
    logger.info(f"  Nonzero genes <= {nonzero_threshold:.0f} ({nonzero_gene_percentage}th pctl): {fail_nonzero} cells removed ({fail_nonzero / total_cells:.2%})")
    logger.info(f"  MT ratio > {mt_threshold:.4f} ({mt_ratio_percentage}th pctl):  {fail_mt} cells removed ({fail_mt / total_cells:.2%})")
    combined_fail = np.sum(
        (row_sums > umi_threshold)
        | (nonzero_genes <= nonzero_threshold)
        | (mt_ratios > mt_threshold)
    )
    logger.info(f"  Any criterion:                   {combined_fail} cells removed ({combined_fail / total_cells:.2%})")
    logger.info(f"  Cells passing all filters:       {total_cells - combined_fail} ({(total_cells - combined_fail) / total_cells:.2%})")

    total_kept_cells = 0
    total_written_files = 0
    for file_path in tqdm(files, desc="Writing filtered chunks"):
        kept_cells, written_files = _write_filtered_data(
            file_path,
            output_path,
            cell_chunk_size,
            umi_threshold,
            nonzero_threshold,
            mt_threshold,
        )
        total_kept_cells += kept_cells
        total_written_files += written_files

    logger.info(f"Finished preprocessing. Kept {total_kept_cells} cells across {total_written_files} output files.")