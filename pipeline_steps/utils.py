"""Helpers shared across pipeline steps."""

import json
import torch
import numpy as np
import anndata 
from copy import deepcopy
from tqdm import tqdm


def point_of_max_curvature(x, y, normalize=True, trim=0.1):
    """
    Return (idx, kappa[idx]) where kappa is curvature of y=f(x).
    - normalize=True rescales x,y to [0,1] (good for elbow picking).
    - trim ignores a small fraction of points at both ends.
    - enforce_monotone makes y nonincreasing via a running minimum.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError("Need at least 3 points to compute curvature.")

    # Optional endpoint/units invariance for elbows
    if normalize:
        xr = (x - x.min()) / (x.max() - x.min() + 1e-12)
        yr = (y - y.min()) / (y.max() - y.min() + 1e-12)
    else:
        xr, yr = x, y

    yx  = np.gradient(yr, xr, edge_order=2)
    yxx = np.gradient(yx, xr, edge_order=2)

    # Curvature for graphs y=f(x): kappa = |y''| / (1 + (y')^2)^(3/2)
    denom = (1.0 + yx*yx)**1.5
    kappa = np.abs(yxx) / np.where(denom == 0, np.nan, denom)

    n = x.size
    lo = 0
    hi = int(n - np.floor(trim * n))
    hi = max(lo + 1, hi)
    window = slice(lo, hi)
    rel = np.nanargmax(kappa[window])
    idx = lo + rel
    return idx, kappa[idx]


def mean_subroutine(X, n_chunk_size=10000):
    """Sum the transformed feature columns of X in chunks."""
    n, d = X.shape
    sums = torch.zeros(d).float().cuda()
    for start in tqdm(range(0, n, n_chunk_size)):
        end = min(start + n_chunk_size, n)
        sums = sums + torch.sum(X[start:end, :], dim=0)
    return sums

def compute_running_mean(files, gene_set = None, cells = None, n_chunk_size = 5000):

    """Running per-gene mean across files, normalised by each cell's total counts."""
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
            X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)

            n, d = X.shape
            
            if means is None: 
                means = torch.zeros(d).float().cuda()
            total += n
            means += n / total * (mean_subroutine(X, n_chunk_size=n_chunk_size) * 1/n - means)
            del X
            torch.cuda.empty_cache()
 
    means = torch.nan_to_num(means, nan=0., posinf=0, neginf=0)
    torch.cuda.empty_cache()

    return means        
       

def compute_running_stddev(files, means, gene_set = None, cells = None, n_chunk_size = 5000):
    """Running per-gene standard deviation across files, given the population means."""
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
            X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
            n, _ = X.shape

            for start in tqdm(range(0, n, n_chunk_size)):
                end = min(start + n_chunk_size, n)
                X_tmp1 = deepcopy(X[start:end, :] - means)
                chunk = torch.mean(torch.square(X_tmp1), axis=0)
                total += len(X_tmp1)
                D += len(X_tmp1) / total * (chunk - D)
                del X_tmp1
                torch.cuda.empty_cache()
    D = torch.sqrt(D)
    D = torch.nan_to_num(D, posinf=0, neginf=0)
    torch.cuda.empty_cache()
    return D


def load_programs(path):
    """Return each program's core gene list, ordered by program number.

    Peripheral genes recorded alongside each program are ignored, so activities
    and summaries are computed over core genes only.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    programs = []
    for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        program_genes = data[key].get("program", [])
        programs.append(program_genes)
    return programs
