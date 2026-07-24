"""InterDependence Score computation over a collection of cell chunks.

Accumulates per-gene means and standard deviations across files, then computes
the pairwise scores in blocks so the full gene-by-gene matrix never needs to fit
in memory at once.
"""

import logging
import os
import torch
import anndata
from tqdm import tqdm
import numpy as np

MIN_MAX_SCALE_FACTOR = 8.
logger = logging.getLogger(__name__)

def cuda_transform(y):
    """Lift each value into a Gaussian-weighted Hermite basis of six features.

    Dependence measured in this expanded space captures non-linear relationships,
    which is what distinguishes the IDS from a correlation.
    """
    B = 0.5
    exp = torch.exp(-B * y * y)
    y_ = torch.cat([exp,
                    (y) * exp,
                    (y*y)/np.sqrt(2) * exp,
                    (y*y*y)/np.sqrt(6) * exp, 
                    (y*y*y*y)/np.sqrt(24) * exp, 
                    (y*y*y*y*y)/np.sqrt(120) * exp,]
                    ,dim=-1)
    return y_

def get_population_max(files, upper_bound, gene_set=None, cells = None):

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

def get_means(X, num_expanded, n_chunk_size=10000, weight=None, labels=None):
    """Sum the expanded features of X in chunks."""
    n, d = X.shape

    sums = torch.zeros(d*num_expanded).float().cuda()
    if weight is not None:
        total_weight = 0
        categories_to_weights = weight
        weights = torch.from_numpy(np.array([categories_to_weights[label] for label in labels])).to(X.device)

    for start in tqdm(range(0, n, n_chunk_size)):
        end = min(start + n_chunk_size, n)
        X_tmp = cuda_transform(X[start:end, :])
        if weight is not None:
            weight_tmp = weights[start:end].reshape(-1, 1)
            sums = sums + torch.sum(weight_tmp * X_tmp, dim=0)
            total_weight += torch.sum(weight_tmp)
        else:
            sums = sums + torch.sum(X_tmp, dim=0)

        del X_tmp
    if weight is not None:
        return sums, total_weight
    else:    
        return sums


def get_population_means(files, upper_bound, maxes, gene_set = None, cells = None, num_expanded = 6, n_chunk_size = 5000, weight=None, label=None):

    """Running per-gene mean of the expanded features across files."""
    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()

    means = None
    total = 0 
    if weight is not None:
        total_weight = 0

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
            if weight is not None:
                adata_labels = list(adata.obs[label].astype(str))
                batch_mean, batch_weight = get_means(X, num_expanded, n_chunk_size=n_chunk_size, weight=weight, labels=adata_labels)
                total_weight += batch_weight
                means += batch_weight/total_weight * (batch_mean * 1/batch_weight - means)
            else: 
                total += n
                means += n / total * (get_means(X, num_expanded, n_chunk_size=n_chunk_size) * 1/n - means)
            del X
            torch.cuda.empty_cache()
 
    means = torch.nan_to_num(means, nan=0., posinf=0, neginf=0).cpu()
    torch.cuda.empty_cache()

    return means        
       

def get_population_stddev(files, upper_bound, maxes, means, gene_set = None, cells = None, num_expanded = 6, n_chunk_size = 5000, weight=None, label=None):
    """Running per-gene inverse standard deviation of the expanded features."""
    d = len(means)
    D = torch.zeros(d, device='cuda')

    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()


    with torch.no_grad():
        means = means.cuda()
        total = 0
        if weight is not None:
            total_weight = 0
            categories_to_weights = weight

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
            if weight is not None:
                adata_labels = list(adata.obs[label].astype(str))
                weights = torch.from_numpy(np.array([categories_to_weights[label] for label in adata_labels])).to(X.device)
            n, _ = X.shape

            for start in tqdm(range(0, n, n_chunk_size)):
                end = min(start + n_chunk_size, n)
                X_tmp1 = cuda_transform(X[start:end, :])
                X_tmp1 = (X_tmp1 - means)

                if weight is not None:
                    weight_tmp = weights[start:end].reshape(-1, 1)
                    batch_weight = torch.sum(weight_tmp)
                    chunk = torch.sum(weight_tmp * torch.square(X_tmp1), axis=0) / batch_weight
                    total_weight += torch.sum(weight_tmp)
                    D += batch_weight / total_weight * (chunk - D)
                else:
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


def calculate_cooccurrence_over_collection_of_anndata_files(directory, gene_set = None, cells = None):
    """Count the cells in which each gene pair is jointly expressed."""
    files = []
    # TODO: sort these paths so the order is filesystem-independent.
    for root, dirs, file in os.walk(directory):
        for f in file:
            if f.startswith("raw_adata_chunk_") and f.endswith(".h5ad"):
                files.append(os.path.join(root, f))

    if gene_set is None:
        fname = files[0]
        adata = anndata.read_h5ad(fname)
        gene_set = adata.var_names.tolist()

    B = torch.zeros(len(gene_set), len(gene_set), device='cuda').float()
    
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
        n, d = X.shape
        X = torch.from_numpy(X).cuda().float()
        X.gt_(0).float()
        chunk = X.T @ X
        B.add_(chunk)
        del X
        del chunk
        torch.cuda.empty_cache()

    return B.cpu().numpy()


def derive_balanced_weights(files, label, gene_set=None, cells=None):
    """
    Computes a single global weight for each observed category of `label` so that all categories 
    have total sum 1/K, and every individual cell in a category has the same weight (1/K / n_c),
    but memory-efficiently only tracking a scalar per category.

    Returns:
        category_to_weight: dict mapping category name to scalar per-cell weight for that category.
    """
    category_counts = {}
    total_samples = 0

    for fname in files:
        adata = anndata.read_h5ad(fname)
        if cells is not None:
            cells_to_use = set(cells).intersection(set(adata.obs.index))
            cells_to_use = list(cells_to_use)
            if len(cells_to_use) == 0:
                continue
            adata = adata[cells_to_use]
        label_values = adata.obs[label].astype(str).values
        for val in np.unique(label_values):
            count = np.sum(label_values == val)
            if val not in category_counts:
                category_counts[val] = 0
            category_counts[val] += count
        total_samples += len(label_values)

    categories = list(category_counts.keys())

    category_to_weight = {}
    for category in categories:
        n_c = category_counts[category]
        if n_c == 0:
            per_cell_weight = 0.0
        else:
            per_cell_weight = total_samples / n_c
        category_to_weight[category] = per_cell_weight

    return category_to_weight


def calculate_IDS_over_collection_of_anndata_files(directory, clip_vals_dir, gene_set = None, cells = None, num_expanded = 6, DIM_chunk_size = 10000, weight=False):
    """Compute the gene-by-gene IDS matrix in blocks."""
    files = []
    # TODO: sort these paths so the order is filesystem-independent.
    for root, dirs, file in os.walk(directory):
        for f in file:
            if f.startswith("raw_adata_chunk_") and f.endswith(".h5ad"):
                files.append(os.path.join(root, f))

    logger.info("Using %d files", len(files))
    logger.debug("files: %s", files)

    categories_to_weights = None

    clip_vals = np.load(os.path.join(clip_vals_dir, 'clip_vals.npy'))
    clip_vals = torch.from_numpy(clip_vals).cuda()
    upper_bound = clip_vals

    maxes = upper_bound
    
    logger.info("Computing means")
    means = get_population_means(files, upper_bound, maxes, gene_set, cells, num_expanded, weight=categories_to_weights, label=None)
    logger.info("Computing standard deviations")
    stddevs = get_population_stddev(files, upper_bound, maxes, means, gene_set, cells, weight=categories_to_weights, label=None)

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
            OUT[dim_start_1:dim_end_1, dim_start_2:dim_end_2] = calculate_IDS_for_batch(files, upper_bound, maxes, means, stddevs, dim_start_1, dim_end_1, dim_start_2, dim_end_2, num_expanded, cells, gene_set, weight=categories_to_weights, label=None)


    return OUT.numpy()



def calculate_IDS_for_batch(files, upper_bound, maxes, means, stddevs, dim_start_1, dim_end_1, dim_start_2, dim_end_2, num_expanded, cells = None, gene_set = None, n_chunk_size = 5000, weight=None, label=None):
    """IDS values for one block of the gene-by-gene matrix."""
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

        for i in range(6):
            means_1.append(means[i*d + dim_start_1:i*d + dim_end_1])
            means_2.append(means[i*d + dim_start_2:i*d + dim_end_2])
            D1.append(stddevs[i*d + dim_start_1:i*d + dim_end_1])
            D2.append(stddevs[i*d + dim_start_2:i*d + dim_end_2])

        means_1 = torch.cat(means_1).cuda()
        means_2 = torch.cat(means_2).cuda()

        D1 = torch.cat(D1).cuda()
        D2 = torch.cat(D2).cuda()

        total = 0
        if weight is not None:
            total_weight = 0    
            categories_to_weights = weight        
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

            if weight is not None:
                adata_labels = list(adata.obs[label].astype(str))
                weights = torch.from_numpy(np.array([categories_to_weights[label] for label in adata_labels])).to(X.device).float()

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

                if weight is not None:
                    weight_tmp = weights[start:end].reshape(-1, 1)
                    batch_weight = torch.sum(weight_tmp)
                    total_weight += batch_weight
                    chunk = (weight_tmp * X_tmp1).T @ X_tmp2
                    chunk.div_(batch_weight)
                    G.add_((chunk - G).mul_(batch_weight / total_weight))
                else:
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
        G.abs_()

        G = G.reshape(num_expanded, dim_end_1-dim_start_1,
                    num_expanded, dim_end_2-dim_start_2)
        G = torch.amax(G, dim=(0, 2))
        G = G.cpu()
        torch.cuda.empty_cache()

    return G