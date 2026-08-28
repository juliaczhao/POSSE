"""Cell selection and chunking.

Selects the requested cell types from the input data and writes them as fixed-size
chunks that the later steps stream over.

Reads
-----
    the raw .h5ad input (a file, or a directory of files)

Writes
------
    <working_dir>/anndata_files/raw_adata_chunk_<n>.h5ad
    <working_dir>/gene_names/gene_names.npy
"""

import anndata
import logging
import os
import shutil
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from collections import Counter

logger = logging.getLogger(__name__)


def sanitize_obs_for_h5_write(obs):
    """Convert any Categorical column whose subset values are all-NaN (i.e. no
    real category present in the data) to a plain string column with NaNs filled
    as ''. h5py's vlen-string writer can't serialize a Categorical that has no
    actually-used categories, even though the categories metadata is non-empty.
    Also coerces object columns containing pd.NA to strings."""
    for col in obs.columns:
        s = obs[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            non_na = s.dropna()
            if len(non_na) == 0:
                obs[col] = s.astype(str).fillna('').replace({'nan': '', 'NaN': '', '<NA>': ''}).values
        elif s.dtype == object:
            if s.isna().any():
                obs[col] = s.astype(str).replace({'nan': '', 'NaN': '', '<NA>': ''}).values
    return obs

def safe_cast_to_uint16(X):
    """
    Safely cast data to uint16 with error checking.
    Uses the fastest method: direct comparison with uint16 limits.
    """
    if sparse.issparse(X):
        max_val = X.max()
        if max_val > 65535:
            raise ValueError(f"Data contains values > 65535 (uint16 limit): max value = {max_val}")
        return X.astype(np.uint16)
    else:
        # For dense arrays, use numpy's built-in overflow checking
        if X.max() > 65535:
            raise ValueError(f"Data contains values > 65535 (uint16 limit): max value = {X.max()}")
        return X.astype(np.uint16)

CELL_CHUNK_SIZE = 100_000


def select_cell_type(dir_name, subpop_dir, cell_types, cell_type_label='cell_type'):

    """Return the subset of adata matching the requested cell types.

    Programs are found within this subset, so the selection sets the biological
    context every downstream step operates in.
    """
    file_entries = load_files(dir_name)

    all_cell_type_counts = Counter()
    total_cells = 0
    gene_names_from_data = None
    chunk_idx = 0

    for parent_dir, fname in tqdm(file_entries):
        logger.info("Processing %s", fname)

        raw_adata = anndata.read_h5ad(os.path.join(parent_dir, fname))
        if cell_type_label not in raw_adata.obs:
            raw_adata.obs[cell_type_label] = 'unannotated'
        # Mirror the chosen cell_type_label to obs['cell_type'] for downstream
        # steps that read 'cell_type' directly (report_plots, llm_summaries, report).
        if cell_type_label != 'cell_type':
            raw_adata.obs['cell_type'] = raw_adata.obs[cell_type_label].values
        if cell_types == []:
            filtered_adata = raw_adata
        else:
            cell_mask = raw_adata.obs[cell_type_label].isin(cell_types)
            filtered_adata = raw_adata[cell_mask]

        cell_type_counts = filtered_adata.obs[cell_type_label].value_counts()
        all_cell_type_counts.update(cell_type_counts.to_dict())
        total_cells += filtered_adata.shape[0]

        if len(cell_type_counts) == 0:
            continue

        # Materialize view to avoid scipy sparse int32 overflow on large views
        if filtered_adata.is_view:
            filtered_adata = filtered_adata.copy()

        if gene_names_from_data is None:
            gene_names_from_data = filtered_adata.var_names.tolist()
            logger.info(f"Total number of genes: {len(filtered_adata.var.index)}")

        logger.info("Casting data to uint16")
        filtered_adata.X = safe_cast_to_uint16(filtered_adata.X)
        logger.info("Cast to uint16 complete")

        # Sanitize obs columns so all-NaN Categoricals don't break h5py write
        filtered_adata.obs = sanitize_obs_for_h5_write(filtered_adata.obs)

        n_cells = filtered_adata.shape[0]
        for start in range(0, n_cells, CELL_CHUNK_SIZE):
            end = min(start + CELL_CHUNK_SIZE, n_cells)
            chunk = filtered_adata[start:end]
            chunk_idx += 1
            output_path = os.path.join(subpop_dir, f'raw_adata_chunk_{chunk_idx}.h5ad')
            chunk.write(output_path)
            logger.info(f"  Saved chunk {chunk_idx}: {chunk.shape[0]} cells to {output_path}")

    logger.info("Cell type counts (across all files):")
    for cell_type, count in all_cell_type_counts.items():
        logger.info(f"  {cell_type}: {count}")
    logger.info(f"Total cells: {total_cells}")
    return gene_names_from_data


def load_files(raw_adata_path):
    """Return list of (directory, filename) pairs for h5ad files at the given path."""
    if os.path.isdir(raw_adata_path):
        h5ad_filenames = sorted(f for f in os.listdir(raw_adata_path) if f.endswith('.h5ad'))
        logger.info("Found %d input file(s)", len(h5ad_filenames))
        for fname in h5ad_filenames:
            logger.info("  %s", fname)
        return [(raw_adata_path, f) for f in h5ad_filenames]
    elif os.path.isfile(raw_adata_path) and raw_adata_path.endswith('.h5ad'):
        logger.info("Using single input file: %s", raw_adata_path)
        return [(os.path.dirname(raw_adata_path), os.path.basename(raw_adata_path))]
    else:
        raise ValueError(f"raw_adata_path is not a directory or h5ad file: {raw_adata_path}")


def extract_cell_types(dir_name, selected_cell_types, output_dir=None,
                       cell_type_label='cell_type', working_dir_name=None):
    """Write the selected cells as fixed-size chunks and return the working directory.

    Every later step streams over these chunks rather than loading the whole
    dataset, so the chunk files define the cell ordering used throughout.
    """
    if working_dir_name is None:
        if len(selected_cell_types) == 0:
            working_dir_name = 'all'
        elif len(selected_cell_types) == 1:
            working_dir_name = selected_cell_types[0]
        else:
            working_dir_name = "_".join(selected_cell_types)

    if selected_cell_types:
        logger.info("Selecting cell types: %s", ", ".join(map(str, selected_cell_types)))
    else:
        logger.info("Selecting all cell types")

    working_dir = os.path.join(output_dir, working_dir_name) if output_dir is not None else working_dir_name
    os.makedirs(working_dir, exist_ok=True)

    # Clear only the subdirectories this step writes into, so stale chunks /
    # gene_names from a prior run don't pollute output. Never rmtree the
    # working_dir itself — it may contain user-owned files (raw inputs,
    # configs, prior pipeline outputs from other steps).
    anndata_files_dir = os.path.join(working_dir, "anndata_files")
    if os.path.exists(anndata_files_dir):
        shutil.rmtree(anndata_files_dir)
    os.makedirs(anndata_files_dir, exist_ok=True)

    subpop_dir = anndata_files_dir

    gene_names_from_data = select_cell_type(dir_name, subpop_dir, selected_cell_types,
                                             cell_type_label=cell_type_label)
    gene_names_dir = os.path.join(working_dir, 'gene_names')
    os.makedirs(gene_names_dir, exist_ok=True)

    np.save(os.path.join(gene_names_dir, f'gene_names.npy'), gene_names_from_data)

    return working_dir