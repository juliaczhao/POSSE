"""Symmetrization of the significant-edge adjacency matrix.

Turns the directed significance calls into an undirected graph, keeping an edge
when it is significant in either direction.

Reads
-----
    <working_dir>/adjacency_matrices/significant_asymmetric_matrix.hkl

Writes
------
    <working_dir>/adjacency_matrices/significant_symmetric_matrix.hkl
"""

import logging
import os
import hickle as hkl
import pandas as pd

logger = logging.getLogger(__name__)



def clean_adjacency_matrix(adj_df, drop_keywords=None):
    """
    Removes genes whose names contain any of the specified keywords from both rows and columns.
    Returns the cleaned DataFrame.
    """
    if drop_keywords is None:
        drop_keywords = ['RPL', 'RPS', 'MT-', 'rpl', 'rps', 'mt-']

    def should_drop(gene):
        return any(kw in gene for kw in drop_keywords)

    genes_to_drop = [gene for gene in adj_df.index if should_drop(gene)]
    return adj_df.drop(index=genes_to_drop, columns=genes_to_drop)


def symmetrize_adjacency_matrix(adj_df):
    """
    Symmetrizes the adjacency matrix: sets entry to 1 if either (i,j) or (j,i) is 1.
    Handles non-square input by first expanding to the union of all row and column genes.
    Returns a square symmetrized DataFrame.
    """
    all_genes = sorted(set(adj_df.index) | set(adj_df.columns))
    sq = adj_df.reindex(index=all_genes, columns=all_genes, fill_value=0)
    adj_sym = ((sq.values + sq.values.T) > 0).astype('int8')
    return pd.DataFrame(adj_sym, index=all_genes, columns=all_genes)


def clean_and_symmetrize(working_dir):
    """Make the significance calls undirected, as program finding requires.

    Significance is tested per direction, so an edge may be called one way only.
    Keeping it in either direction gives the undirected graph used downstream.
    """
    adj_dir = os.path.join(working_dir, 'adjacency_matrices')
    adj_path = os.path.join(adj_dir, 'significant_asymmetric_matrix.hkl')
    adj_df = hkl.load(adj_path)


    adj_df_sym = symmetrize_adjacency_matrix(adj_df)

    sym_path = os.path.join(adj_dir, "significant_symmetric_matrix.hkl")
    hkl.dump(adj_df_sym, sym_path)

    percent_nonzero = 100.0 * (adj_df_sym.values != 0).sum() / adj_df_sym.size
    logger.info("Symmetric adjacency matrix: shape %s, %d nodes, %.2f%% density",
                adj_df_sym.shape, len(adj_df_sym), percent_nonzero)
    logger.info("Saved symmetrized adjacency matrix to %s", sym_path)