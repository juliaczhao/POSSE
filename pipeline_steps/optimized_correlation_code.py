"""Pearson correlation over a collection of cell chunks.

Mirrors the InterDependence Score computation, accumulating per-gene statistics
across files and computing the pairwise matrix in blocks.
"""

import logging
import os
import torch
import anndata
from tqdm import tqdm
import numpy as np
from copy import deepcopy

MIN_MAX_SCALE_FACTOR = 8.
logger = logging.getLogger(__name__)

def cuda_transform(y):
    """Return the values unchanged, so correlation is measured on the raw scale."""
    return deepcopy(y)

def get_population_max(files, upper_bound, gene_set, cells = None):

    """Per-gene maximum across files, after normalisation and clipping."""
    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()

    maxes = None

    with torch.no_grad():
        for fname in tqdm(files):
            adata = anndata.read_h5ad(fname)
            adata = adata[:, gene_set]
            if cells is not None:
                cells_to_use = set(cells).intersection(set(adata.obs.index))
                cells_to_use = list(cells_to_use)
                if len(cells_to_use) == 0:
                    continue
                adata = adata[cells_to_use]
            X = adata.X.toarray()
            X = torch.from_numpy(X).cuda().float()
            X /= torch.sum(X, axis=-1, keepdims=True)
            X.clamp_(max=upper_bound)
                        
            if maxes is None:
                maxes = torch.max(X, dim=0, keepdims=True)[0]
            else:
                maxes = torch.max(maxes, torch.max(X, dim=0, keepdims=True)[0])

    return maxes

def get_means(X, num_expanded, n_chunk_size=10000):
    """Sum the feature columns of X in chunks."""
    n, d = X.shape

    sums = torch.zeros(d*num_expanded).float().cuda()
    for start in tqdm(range(0, n, n_chunk_size)):
        end = min(start + n_chunk_size, n)
        X_tmp = cuda_transform(X[start:end, :])
        sums = sums + torch.sum(X_tmp, dim=0)
        del X_tmp

    return sums

def get_population_means(files, upper_bound, maxes, gene_set = None, cells = None, num_expanded = 1, n_chunk_size = 5000):

    """Running per-gene mean across files."""
    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()

    means = None
    total = 0 

    with torch.no_grad():
        for idx, fname in enumerate(tqdm(files)):
            adata = anndata.read_h5ad(fname)
            adata = adata[:, gene_set]
            if cells is not None:
                cells_to_use = set(cells).intersection(set(adata.obs.index))
                cells_to_use = list(cells_to_use)
                if len(cells_to_use) == 0:
                    continue
                adata = adata[cells_to_use]
            
            X = adata.X.toarray()
            X = torch.from_numpy(X).cuda().float()
            X /= torch.sum(X, axis=-1, keepdims=True)
            X.clamp_(max=upper_bound)
            X *= MIN_MAX_SCALE_FACTOR / maxes
            X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)

            n, d = X.shape
            
            if means is None: 
                means = torch.zeros(d * num_expanded).float().cuda()
            total += n
            means += n / total * (get_means(X, num_expanded, n_chunk_size=n_chunk_size) * 1/n - means)
            del X
            torch.cuda.empty_cache()
 
    means = torch.nan_to_num(means, nan=0., posinf=0, neginf=0).cpu()
    torch.cuda.empty_cache()

    return means        
       

def get_population_stddev(files, upper_bound, maxes, means, gene_set = None, cells = None, num_expanded = 1, n_chunk_size = 5000):
    """Running per-gene inverse standard deviation across files."""
    d = len(means)
    D = torch.zeros(d, device='cuda')

    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()

    with torch.no_grad():
        means = means.cuda()
        total = 0
        for idx, fname in enumerate(tqdm(files)):
            adata = anndata.read_h5ad(fname)
            adata = adata[:, gene_set]
            if cells is not None:
                cells_to_use = set(cells).intersection(set(adata.obs.index))
                cells_to_use = list(cells_to_use)
                if len(cells_to_use) == 0:
                    continue
                adata = adata[cells_to_use]
            
            X = adata.X.toarray()
            X = torch.from_numpy(X).cuda().float()
            X /= torch.sum(X, axis=-1, keepdims=True)
            X.clamp_(max=upper_bound)
            X *= MIN_MAX_SCALE_FACTOR / maxes
            X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
            n, _ = X.shape

            for start in tqdm(range(0, n, n_chunk_size)):
                end = min(start + n_chunk_size, n)
                X_tmp1 = cuda_transform(X[start:end, :])
                X_tmp1 = (X_tmp1 - means)

                chunk = torch.mean(torch.square(X_tmp1), axis=0)

                total += len(X_tmp1)
                D += len(X_tmp1) / total * (chunk - D)
                del X_tmp1
                torch.cuda.empty_cache()
    D = D.cpu()
    D = 1/torch.sqrt(D)
    D = torch.nan_to_num(D, posinf=0, neginf=0)
    torch.cuda.empty_cache()
    return D


def calculate_corr_over_collection_of_anndata_files(directory, clip_vals_dir, gene_set = None, cells = None, num_expanded = 1, DIM_chunk_size = 10000):
    """Compute the gene-by-gene correlation matrix in blocks."""
    files = []
    # TODO: sort these paths so the order is filesystem-independent.
    for root, dirs, file in os.walk(directory):
        for f in file:
            if f.startswith("raw_adata_chunk_") and f.endswith(".h5ad"):
                files.append(os.path.join(root, f))

    logger.info("Using %d files", len(files))
    logger.debug("files: %s", files)
    clip_vals = np.load(os.path.join(clip_vals_dir, 'clip_vals.npy'))
    clip_vals = torch.from_numpy(clip_vals).cuda()
    upper_bound = clip_vals

    maxes = upper_bound

    logger.info("Computing means")
    means = get_population_means(files, upper_bound, maxes, gene_set, cells, num_expanded)
    logger.info("Computing standard deviations")
    stddevs = get_population_stddev(files, upper_bound, maxes, means, gene_set, cells)

    d = len(means) // num_expanded

    var_batches = []

    for dim_start_1 in (range(0, d, DIM_chunk_size)):
        dim_end_1 = min(dim_start_1 + DIM_chunk_size, d)
        for dim_start_2 in (range(dim_start_1, d, DIM_chunk_size)):
            dim_end_2 = min(dim_start_2 + DIM_chunk_size, d)
            var_batches.append((dim_start_1, dim_end_1, dim_start_2, dim_end_2))


    OUT = torch.zeros(d, d, device='cpu')
    for var_batch_tuple in tqdm(var_batches):
        dim_start_1, dim_end_1, dim_start_2, dim_end_2 = var_batch_tuple
        with torch.no_grad():
            torch.backends.cudnn.benchmark = True        
            torch.backends.cuda.matmul.allow_tf32 = True   
            OUT[dim_start_1:dim_end_1, dim_start_2:dim_end_2] = calculate_corr_for_batch(files, upper_bound, maxes, means, stddevs, dim_start_1, dim_end_1, dim_start_2, dim_end_2, num_expanded, cells, gene_set)


    return OUT.numpy()



def calculate_corr_for_batch(files, upper_bound, maxes, means, stddevs, dim_start_1, dim_end_1, dim_start_2, dim_end_2, num_expanded, cells = None, gene_set = None, n_chunk_size = 5000):
    """Correlation values for one block of the gene-by-gene matrix."""
    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()
    
    with torch.no_grad():    
        G = torch.zeros(((dim_end_1-dim_start_1)*num_expanded, (dim_end_2-dim_start_2)*num_expanded), device='cuda')

        means_1 = []
        means_2 = []

        D1 = []
        D2 = []

        d = len(means) // num_expanded

        for i in range(1):
            means_1.append(means[i*d + dim_start_1:i*d + dim_end_1])
            means_2.append(means[i*d + dim_start_2:i*d + dim_end_2])
            D1.append(stddevs[i*d + dim_start_1:i*d + dim_end_1])
            D2.append(stddevs[i*d + dim_start_2:i*d + dim_end_2])

        means_1 = torch.cat(means_1).cuda()
        means_2 = torch.cat(means_2).cuda()

        D1 = torch.cat(D1).cuda()
        D2 = torch.cat(D2).cuda()

        total = 0

        for idx, fname in enumerate(tqdm(files)):
            adata = anndata.read_h5ad(fname)
            adata = adata[:, gene_set]
            if cells is not None:
                cells_to_use = set(cells).intersection(set(adata.obs.index))
                cells_to_use = list(cells_to_use)
                adata = adata[cells_to_use]
            X = adata.X.toarray()
            X = torch.from_numpy(X).cuda().float()
            X /= torch.sum(X, axis=-1, keepdims=True)
            X.clamp_(max=upper_bound)
            X *= MIN_MAX_SCALE_FACTOR / maxes
            X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)

            X1 = X[:, dim_start_1:dim_end_1]
            X1 = X1.float()
            X2 = X[:, dim_start_2:dim_end_2]
            X2 = X2.float()

            n, _ = X1.shape

            for start in tqdm(range(0, n, n_chunk_size)):
                end = min(start + n_chunk_size, n)
                X_tmp1 = cuda_transform(X1[start:end, :])
                X_tmp1.sub_(means_1)

                X_tmp2 = cuda_transform(X2[start:end, :])
                X_tmp2.sub_(means_2)

                total += len(X_tmp1)
                chunk = X_tmp1.T @ X_tmp2
                chunk.div_(len(X_tmp1))
                G.add_((chunk - G).mul_(len(X_tmp1) / total))
                del chunk
                del X_tmp1
                del X_tmp2
            del X
            del X1
            del X2
            torch.cuda.empty_cache()

        G *= D1.view(-1, 1)
        G *= D2.view(1, -1)

        G = G.reshape(num_expanded, dim_end_1-dim_start_1,
                    num_expanded, dim_end_2-dim_start_2)
        G = torch.amax(G, dim=(0, 2))
        G = G.cpu()
        torch.cuda.empty_cache()

    return G