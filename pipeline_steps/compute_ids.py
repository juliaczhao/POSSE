"""Gene-gene dependence matrices.

Computes InterDependence Scores, Pearson correlation and cooccurrence counts for
every gene pair, streaming over the cell chunks.

Reads
-----
    <working_dir>/anndata_files/raw_adata_chunk_*.h5ad
    <working_dir>/supp_material/clip_vals.npy
    <working_dir>/gene_names/gene_names.npy

Writes
------
    <working_dir>/ids_cooccur/ids_matrix.hkl
    <working_dir>/ids_cooccur/correlation_matrix.hkl
    <working_dir>/ids_cooccur/cooccurrence_matrix.hkl
"""

import logging
from pipeline_steps.optimized_ids_code import calculate_IDS_over_collection_of_anndata_files
from pipeline_steps.optimized_ids_code import calculate_cooccurrence_over_collection_of_anndata_files
from pipeline_steps.optimized_correlation_code import calculate_corr_over_collection_of_anndata_files
import os
import torch
import numpy as np
import pandas as pd
import hickle as hkl

logger = logging.getLogger(__name__)


def load_upper_tri(P, NUM_GENES):
    """Rebuild a full symmetric matrix from its stored upper triangle.

    The correlation and cooccurrence matrices are symmetric, so only the upper
    triangle is saved; this restores the square form the later steps index into.
    """
    d = NUM_GENES
    i, j = torch.triu_indices(d, d)
    P = torch.from_numpy(P).float()
    reconstructed_matrix = torch.zeros((d, d), dtype=torch.float32)

    reconstructed_matrix[i, j] = P 
    reconstructed_matrix[j, i] = P 
    return reconstructed_matrix

def compute_ids_and_cooccur(dir_name, output_dir, clip_vals_dir, working_dir=None):
    """Compute the gene-gene dependence matrices that significance testing consumes.

    The IDS values are tested for significance; the correlation supplies the sign
    of each relationship, and the cooccurrence counts decide which pairs carry
    enough evidence to test at all.
    """
    C = calculate_cooccurrence_over_collection_of_anndata_files(dir_name)
    M = calculate_IDS_over_collection_of_anndata_files(dir_name, clip_vals_dir, weight=False)
    M = M * 1e4
    M = M.astype(np.uint16)
    M = M.astype(np.float32) / 10000.
    M = torch.from_numpy(M).float()
    upper_tri_mask = torch.triu(torch.ones_like(M), diagonal=0).bool()
    M_upper = M[upper_tri_mask].cpu().numpy()
    M = load_upper_tri(M_upper, len(M))
    M.fill_diagonal_(1.0)

    IDS = M.cpu().numpy()

    S = calculate_corr_over_collection_of_anndata_files(dir_name, clip_vals_dir)
    S = torch.from_numpy(S).float()
    upper_tri_mask = torch.triu(torch.ones_like(S), diagonal=0).bool()
    S_upper = S[upper_tri_mask].cpu().numpy()
    S = load_upper_tri(S_upper, len(S))    
    S.fill_diagonal_(1.0)
    Corr = S.cpu().numpy()

    gene_names = None
    if working_dir is not None:
        gene_names_dir = os.path.join(working_dir, 'gene_names')
        gene_names_path = os.path.join(gene_names_dir, 'gene_names.npy')
        if os.path.exists(gene_names_path):
            gene_names = np.load(gene_names_path, allow_pickle=True).tolist()
            logger.info(f"Loaded {len(gene_names)} gene names")
    
    if gene_names is None:
        raise ValueError("Gene names are required to save IDS/cooccurrence/correlation matrices.")
    if len(gene_names) != len(IDS):
        raise ValueError(
            f"Gene names length ({len(gene_names)}) doesn't match matrix size ({len(IDS)})."
        )

    ids_df = pd.DataFrame(IDS, index=gene_names, columns=gene_names)
    ids_hkl_path = os.path.join(output_dir, 'ids_matrix.hkl')
    hkl.dump(ids_df, ids_hkl_path)
    logger.info(f"Saved IDS matrix as DataFrame to {ids_hkl_path}")

    corr_df = pd.DataFrame(Corr, index=gene_names, columns=gene_names)
    corr_hkl_path = os.path.join(output_dir, 'correlation_matrix.hkl')
    hkl.dump(corr_df, corr_hkl_path)
    logger.info(f"Saved correlation matrix as DataFrame to {corr_hkl_path}")

    cooccur_df = pd.DataFrame(C, index=gene_names, columns=gene_names)
    cooccur_hkl_path = os.path.join(output_dir, 'cooccurrence_matrix.hkl')
    hkl.dump(cooccur_df, cooccur_hkl_path)
    logger.info(f"Saved cooccurrence matrix as DataFrame to {cooccur_hkl_path}")
