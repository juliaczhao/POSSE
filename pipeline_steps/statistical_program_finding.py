"""Significance testing of gene-gene dependence.

Fits a Beta distribution to each gene's dependence scores with the xRFM models,
converts the scores to p-values, and applies Benjamini-Hochberg correction to
decide which gene pairs form edges.

Reads
-----
    <working_dir>/ids_cooccur/ids_matrix.hkl
    <working_dir>/ids_cooccur/cooccurrence_matrix.hkl
    <working_dir>/gene_names/gene_names.npy
    the xRFM models named in pipeline_config.json

Writes
------
    <working_dir>/adjacency_matrices/significant_asymmetric_matrix.hkl
    <working_dir>/supp_material/beta_fit_all_results.json
    <working_dir>/supp_material/beta_fit_excluded_results.json
    <working_dir>/supp_material/pvals.hkl
"""

import logging
import os

# Determinism: cuBLAS picks matmul algorithms heuristically by default, which
# makes xRFM model.predict non-bitwise-reproducible across runs. Set the
# cuBLAS workspace config *before* torch is imported / initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import contextlib
import io
import tempfile
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, kstwobign
from scipy import special as sc_special
from tqdm import tqdm
import pickle
import hickle as hkl
import json


def _enable_deterministic_torch():
    """Lock down the torch/CUDA stack for bit-exact runs.

    Must be called in every process that does GPU work (main + each spawned
    worker), because spawned processes are fresh interpreters.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_enable_deterministic_torch()

logger = logging.getLogger(__name__)

GENE_CHUNK_SIZE = 10000
_CHUNKS_PER_WORKER = 4
_MIN_CHUNK = 500


_QUANTILE_POS = np.linspace(0, 1, 200)

_MIN_C, _MAX_C = 0.01, 1.0
_MIN_M, _MAX_M = 200, 20000
_MAX_K = 10.0
_MIN_AB, _MAX_AB = 0.3, 100.0
_LOG_KAPPA_MIN = np.log(_MIN_AB + _MIN_AB + 1)
_LOG_KAPPA_MAX = np.log(_MAX_AB + _MAX_AB + 1)


def _norm_c(c):
    return (c - _MIN_C) / (_MAX_C - _MIN_C)


def _norm_m(m):
    return (np.log(np.clip(m, 1, None)) - np.log(_MIN_M)) / (np.log(_MAX_M) - np.log(_MIN_M))


def _denorm_log_kappa(lk_norm):
    return lk_norm * (_LOG_KAPPA_MAX - _LOG_KAPPA_MIN) + _LOG_KAPPA_MIN


def _vectorized_quantiles(sorted_data, n_keeps, offsets, q_pos=_QUANTILE_POS):
    """Quantiles at q_pos for variable-length sorted rows with offsets."""
    nk = np.maximum(n_keeps.astype(np.float64), 1)
    float_idx = q_pos[None, :] * (nk[:, None] - 1)
    lo_local = np.floor(float_idx).astype(np.intp)
    hi_local = np.minimum(lo_local + 1, (n_keeps[:, None] - 1).astype(np.intp))
    frac = (float_idx - lo_local).astype(np.float32)
    off = offsets[:, None].astype(np.intp)
    lo = lo_local + off
    hi = hi_local + off
    rows = np.arange(len(n_keeps))[:, None]
    val_lo = sorted_data[rows, lo].astype(np.float32)
    val_hi = sorted_data[rows, hi].astype(np.float32)
    return val_lo + frac * (val_hi - val_lo)


def _ks_statistic_chunk(sorted_data_rows, n_keeps, offsets, c_vals, a_vals, b_vals):
    """Two-sided KS statistic for a chunk of rows with offsets."""
    chunk = len(n_keeps)
    max_nk = int(n_keeps.max())
    col_idx = offsets[:, None].astype(np.intp) + np.arange(max_nk, dtype=np.intp)[None, :]
    col_idx = np.clip(col_idx, 0, sorted_data_rows.shape[1] - 1)
    data = sorted_data_rows[np.arange(chunk)[:, None], col_idx]
    j = np.arange(max_nk, dtype=np.float64)[None, :]
    valid = j < n_keeps[:, None]
    nk_f = n_keeps[:, None].astype(np.float64)
    x_norm = data / c_vals[:, None]
    F_theo = beta.cdf(np.clip(x_norm, 0, 1 - 1e-15), a_vals[:, None], b_vals[:, None])
    d_plus  = np.where(valid, (j + 1) / nk_f - F_theo, -np.inf).max(axis=1)
    d_minus = np.where(valid, F_theo - j / nk_f,        -np.inf).max(axis=1)
    return np.maximum(d_plus, d_minus)


def _build_sorted_inf_matrix(M_filt):
    """Extract nonzero values per row, sort ascending, pad with inf.

    Returns (sorted_data, m_values) where sorted_data is (N, max_obs) float64
    with real values left-packed and inf padding on the right, and m_values is
    the number of nonzero observations per row.
    """
    m_values = np.count_nonzero(M_filt, axis=1).astype(np.int64)
    max_obs = int(m_values.max())
    sorted_data = np.full((len(m_values), max_obs), np.inf, dtype=np.float64)
    for i in range(len(m_values)):
        row = M_filt[i]
        vals = np.sort(row[row != 0])
        sorted_data[i, :len(vals)] = vals
    return sorted_data, m_values


_K_GRID_DEFAULT = np.array([0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _silent_predict(model, features):
    """Run a fitted model's predict while discarding its stdout chatter.

    The xRFM models print routing notices to stdout on every call; our own logging
    goes to stderr, so suppressing stdout here keeps the logs clean without hiding
    any pipeline output.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return model.predict(features)


def _fit_from_matrix(sorted_data, m_values, model_mu, model_logkappa,
                     k_lo_grid=None, k_hi_grid=None,
                     ks_threshold=0.01, n_iter=30, conv_tol=1e-6,
                     chunk_size=2000):
    """Estimate (c, k_lo, k, alpha, beta, cvm, n, ks_pval) for every row.

    Parameters
    ----------
    sorted_data : (N, max_obs) float64, real values left-packed, inf padding.
    m_values : (N,) int64, nonzero count per row.
    model_mu, model_logkappa : fitted pytabkit regressors.
    k_lo_grid, k_hi_grid : left/right trim % grids (default [0,.5,1,1.5,2]).
    ks_threshold : KS p-value accept threshold (default 0.01).

    Returns
    -------
    dict of arrays length N: c, k_lo, k, alpha, beta, cvm, n, ks_pval.
    """
    if k_lo_grid is None:
        k_lo_grid = _K_GRID_DEFAULT
    else:
        k_lo_grid = np.asarray(k_lo_grid, dtype=float)
    if k_hi_grid is None:
        k_hi_grid = _K_GRID_DEFAULT
    else:
        k_hi_grid = np.asarray(k_hi_grid, dtype=float)

    pairs = [(float(kl), float(kh)) for kl in k_lo_grid for kh in k_hi_grid]
    pairs.sort(key=lambda p: (p[0] + p[1], p[0]))
    n_pairs = len(pairs)
    pair_k_lo = np.array([p[0] for p in pairs])
    pair_k_hi = np.array([p[1] for p in pairs])

    N, max_obs = sorted_data.shape
    P = N * n_pairs

    n_trim_lo_2d = np.ceil(m_values[:, None] * pair_k_lo[None, :] / 100).astype(np.int64)
    n_trim_hi_2d = np.ceil(m_values[:, None] * pair_k_hi[None, :] / 100).astype(np.int64)
    n_keeps_2d = np.maximum(m_values[:, None] - n_trim_lo_2d - n_trim_hi_2d, 2)

    max_idx_2d = n_trim_lo_2d + n_keeps_2d - 1
    max_vals_2d = sorted_data[np.arange(N)[:, None], max_idx_2d]

    all_quantiles = np.empty((N, n_pairs, 200), dtype=np.float32)
    for pi in range(n_pairs):
        all_quantiles[:, pi, :] = _vectorized_quantiles(
            sorted_data, n_keeps_2d[:, pi], n_trim_lo_2d[:, pi])

    max_vals_flat = max_vals_2d.reshape(P)
    m_flat = np.repeat(m_values, n_pairs).astype(np.float64)
    n_trim_hi_flat = n_trim_hi_2d.reshape(P).astype(np.float64)
    ppf_prob = (m_flat - n_trim_hi_flat) / (m_flat + 1)

    features = np.empty((P, 204), dtype=np.float32)
    features[:, :200] = all_quantiles.reshape(P, 200)
    features[:, 201] = np.tile(pair_k_lo, N) / _MAX_K
    features[:, 202] = np.tile(pair_k_hi, N) / _MAX_K
    features[:, 203] = _norm_m(np.repeat(m_values, n_pairs).astype(np.float32))

    c = max_vals_flat * 1.5
    converged = np.zeros(P, dtype=bool)

    for _ in range(n_iter):
        features[:, 200] = _norm_c(c).astype(np.float32)
        mu  = _silent_predict(model_mu, features)
        lk  = _silent_predict(model_logkappa, features)
        kap = np.exp(_denorm_log_kappa(lk))
        a   = np.clip(mu * (kap - 1), 1e-6, None)
        b   = np.clip((1 - mu) * (kap - 1), 1e-6, None)
        ppf = beta.ppf(ppf_prob, a, b)
        c_new = max_vals_flat / np.clip(ppf, 1e-10, None)
        c_new = np.clip(c_new, max_vals_flat * 1.001, 1.0)
        converged |= np.abs(c_new - c) / np.clip(c, 1e-10, None) < conv_tol
        c = np.where(converged, c, c_new)
        if converged.all():
            break

    features[:, 200] = _norm_c(c).astype(np.float32)
    mu  = _silent_predict(model_mu, features)
    lk  = _silent_predict(model_logkappa, features)
    kap = np.exp(_denorm_log_kappa(lk))
    alpha_flat = np.clip(mu * (kap - 1), 1e-6, None)
    beta_flat  = np.clip((1 - mu) * (kap - 1), 1e-6, None)

    alpha_2d = alpha_flat.reshape(N, n_pairs)
    beta_2d  = beta_flat.reshape(N, n_pairs)
    c_2d     = c.reshape(N, n_pairs)
    offsets_2d = n_trim_lo_2d

    selected_pi = np.full(N, -1, dtype=np.int64)
    best_pval   = np.full(N, -1.0)
    best_pi     = np.zeros(N, dtype=np.int64)
    remaining   = np.ones(N, dtype=bool)
    ks_pval_sel = np.zeros(N)

    for pi in range(n_pairs):
        idx = np.where(remaining)[0]
        if len(idx) == 0:
            break
        nk  = n_keeps_2d[idx, pi]
        off = offsets_2d[idx, pi]
        a_p = alpha_2d[idx, pi]
        b_p = beta_2d[idx, pi]
        c_p = c_2d[idx, pi]
        D = _ks_statistic_chunk(sorted_data[idx], nk, off, c_p, a_p, b_p)
        pval = kstwobign.sf(D * np.sqrt(nk.astype(np.float64)))
        better = pval > best_pval[idx]
        best_pval[idx[better]] = pval[better]
        best_pi[idx[better]] = pi
        passes = pval > ks_threshold
        pass_idx = idx[passes]
        selected_pi[pass_idx] = pi
        ks_pval_sel[pass_idx] = pval[passes]
        remaining[pass_idx] = False

    fallback = selected_pi == -1
    selected_pi[fallback] = best_pi[fallback]
    ks_pval_sel[fallback] = best_pval[fallback]

    ri = np.arange(N)
    out_c     = c_2d[ri, selected_pi]
    out_alpha = alpha_2d[ri, selected_pi]
    out_beta  = beta_2d[ri, selected_pi]
    out_k_lo  = pair_k_lo[selected_pi]
    out_k     = pair_k_hi[selected_pi]
    out_n     = n_keeps_2d[ri, selected_pi]
    out_off   = offsets_2d[ri, selected_pi]

    max_nk_all = int(out_n.max())
    col_idx = out_off[:, None].astype(np.intp) + np.arange(max_nk_all, dtype=np.intp)[None, :]
    col_idx = np.clip(col_idx, 0, sorted_data.shape[1] - 1)
    data_all = sorted_data[np.arange(N)[:, None], col_idx]
    j = np.arange(max_nk_all, dtype=np.float64)[None, :]
    valid = j < out_n[:, None]
    x_n = data_all / out_c[:, None]
    F_t = beta.cdf(np.clip(x_n, 0, 1 - 1e-15), out_alpha[:, None], out_beta[:, None])
    F_e = (j + 1) / out_n[:, None].astype(np.float64)
    sq = np.where(valid, (F_t - F_e) ** 2, 0)
    out_cvm = sq.sum(axis=1) / out_n

    return {
        "c": out_c, "k_lo": out_k_lo, "k": out_k,
        "alpha": out_alpha, "beta": out_beta,
        "cvm": out_cvm, "n": out_n,
        "ks_pval": ks_pval_sel,
    }



def load_and_filter_ids(ids_path, cooccur_path, gene_names, COOCCUR_THRESHOLD=100):
    """Load the IDS matrix and mask the pairs that should not be tested.

    Gene pairs seen together in too few cells carry too little evidence to fit,
    and self-pairs would distort the fitted distribution, so both are zeroed
    before the beta fit.
    """
    logger.info(f"Loading IDS matrix from {ids_path}")
    logger.info(f"Loading cooccurrence matrix from {cooccur_path}")
    ids_df = hkl.load(ids_path)
    cooccur_df = hkl.load(cooccur_path)
    if list(ids_df.index) != gene_names or list(cooccur_df.index) != gene_names:
        logger.warning("Gene names do not match DataFrame indices; reindexing")
        ids_df = ids_df.reindex(index=gene_names, columns=gene_names, fill_value=0)
        cooccur_df = cooccur_df.reindex(index=gene_names, columns=gene_names, fill_value=0)
    mask = (cooccur_df.values > COOCCUR_THRESHOLD).astype(np.float32)
    np.fill_diagonal(mask, 0)
    M = ids_df.values * mask
    return M



def _load_xrfm_models(model_dir, mu_filename, logkappa_filename):
    """Load the fitted beta models used to assign a p-value to every IDS value.

    The two models predict the beta distribution a gene pair's IDS would follow
    under the null, from which the significance of the observed value is read.
    """
    mu_path = os.path.join(model_dir, mu_filename)
    lk_path = os.path.join(model_dir, logkappa_filename)
    logger.info(f"Loading mu model from {mu_path}")
    with open(mu_path, "rb") as f:
        model_mu = pickle.load(f)
    logger.info(f"Loading logkappa model from {lk_path}")
    with open(lk_path, "rb") as f:
        model_lk = pickle.load(f)
    return model_mu, model_lk



def _bh_fdr_vectorized(pvals_matrix, alpha, row_chunk=2000):
    """BH-FDR for all genes at once, chunked over rows to limit peak memory.

    Mathematically identical to calling multipletests(method='fdr_bh') per gene.

    Parameters
    ----------
    pvals_matrix : (n_genes, n_cols) float array, NaN where not observed.
    alpha        : FDR level.
    row_chunk    : genes processed per iteration (trades memory for speed).

    Returns
    -------
    rejected : (n_genes, n_cols) bool, True where BH-significant.
    """
    n_genes, n_cols = pvals_matrix.shape
    rejected = np.zeros((n_genes, n_cols), dtype=bool)

    for s in range(0, n_genes, row_chunk):
        e = min(s + row_chunk, n_genes)
        chunk = pvals_matrix[s:e]

        filled = np.where(np.isnan(chunk), 2.0, chunk.astype(np.float64))

        m_i = (filled < 1.5).sum(axis=1, keepdims=True).astype(np.float64)
        m_i = np.maximum(m_i, 1.0)

        sort_idx = np.argsort(filled, axis=1, kind='stable')
        sorted_p = np.take_along_axis(filled, sort_idx, axis=1)

        ranks = np.arange(1, n_cols + 1, dtype=np.float64)[None, :]
        bh_thresh = alpha * ranks / m_i

        in_bounds = ranks <= m_i
        below = (sorted_p <= bh_thresh) & in_bounds

        rej_sorted = np.flip(
            np.maximum.accumulate(np.flip(below.view(np.uint8), axis=1), axis=1),
            axis=1,
        ).astype(bool)

        rej = np.zeros((e - s, n_cols), dtype=bool)
        np.put_along_axis(rej, sort_idx, rej_sorted, axis=1)

        rej[np.isnan(chunk)] = False

        rejected[s:e] = rej

    return rejected


# Multi-GPU inference: rows are split into contiguous slabs, one per GPU, each run
# in a spawned worker under its own CUDA_VISIBLE_DEVICES.

def _gpu_inference_worker(gpu_id, row_start, row_end,
                          sorted_data_path, m_values, model_dir,
                          mu_filename, logkappa_filename,
                          gene_chunk_size, out_path):
    """Run _fit_from_matrix on rows [row_start, row_end) on a single GPU.

    Pins to ``gpu_id`` via CUDA_VISIBLE_DEVICES so the xRFM model (which loads
    onto ``cuda:0`` by default) maps to the chosen physical GPU. Loads its own
    model copy from disk. Mirrors the _gpu_worker pattern from
    multigpu_main.py.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _enable_deterministic_torch()

    sorted_data = np.load(sorted_data_path, mmap_mode="r")[row_start:row_end]
    m_values_slab = m_values[row_start:row_end]

    model_mu, model_lk = _load_xrfm_models(model_dir, mu_filename, logkappa_filename)

    chunk_keys = ['c', 'k_lo', 'k', 'alpha', 'beta', 'cvm', 'n', 'ks_pval']
    all_chunks = {k: [] for k in chunk_keys}
    n_rows = row_end - row_start
    logger.debug(f"[GPU {gpu_id}] processing rows {row_start}:{row_end} "
          f"({n_rows} genes) in chunks of {gene_chunk_size}")
    for start in range(0, n_rows, gene_chunk_size):
        end = min(start + gene_chunk_size, n_rows)
        res = _fit_from_matrix(np.asarray(sorted_data[start:end]),
                               m_values_slab[start:end],
                               model_mu, model_lk)
        for k in chunk_keys:
            all_chunks[k].append(res[k])
        logger.debug(f"[GPU {gpu_id}] {end}/{n_rows} done")
    out = {k: np.concatenate(v) for k, v in all_chunks.items()}

    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    logger.debug(f"[GPU {gpu_id}] saved {out_path}")


def _multi_gpu_batch_fit(sorted_data, m_values, model_dir,
                          mu_filename, logkappa_filename,
                          n_gpus, gene_chunk_size):
    """Distribute _fit_from_matrix across N GPUs.

    Splits rows into n_gpus contiguous slabs, spawns one worker per GPU
    (pinned via CUDA_VISIBLE_DEVICES), aggregates per-GPU results in slab
    order so the output matches the single-GPU code path row-for-row.

    sorted_data is written once to a tempfile (np.save) and mmap-loaded by
    each worker, avoiding bulk-pickling a multi-GB array through Process args.
    """
    chunk_keys = ['c', 'k_lo', 'k', 'alpha', 'beta', 'cvm', 'n', 'ks_pval']
    N = len(m_values)

    with tempfile.TemporaryDirectory(prefix="xrfm_infer_multigpu_") as tmp:
        data_path = os.path.join(tmp, "sorted_data.npy")
        np.save(data_path, sorted_data)

        slabs = np.array_split(np.arange(N), n_gpus)

        ctx = mp.get_context("spawn")
        processes = []
        out_paths = []
        for gpu_id, slab in enumerate(slabs):
            if len(slab) == 0:
                continue
            row_start = int(slab[0])
            row_end = int(slab[-1]) + 1
            slab_size = row_end - row_start
            worker_chunk = max(min(gene_chunk_size, slab_size // _CHUNKS_PER_WORKER),
                               min(_MIN_CHUNK, slab_size))
            out_path = os.path.join(tmp, f"results_gpu{gpu_id}.pkl")
            out_paths.append((gpu_id, out_path))
            p = ctx.Process(
                target=_gpu_inference_worker,
                args=(gpu_id, row_start, row_end, data_path, m_values,
                      model_dir, mu_filename, logkappa_filename,
                      worker_chunk, out_path),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [(i, p) for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            msgs = [f"GPU {i} (pid={p.pid}) exited with code {p.exitcode}"
                    for i, p in failed]
            raise RuntimeError("Worker(s) failed:\n  " + "\n  ".join(msgs))

        per_gpu_results = []
        for gpu_id, path in out_paths:
            with open(path, "rb") as f:
                per_gpu_results.append(pickle.load(f))

    return {k: np.concatenate([r[k] for r in per_gpu_results])
            for k in chunk_keys}



def _gpu_pval_worker(gpu_id, row_start, row_end, M_filt_path,
                     pvals_path, pvals_shape,
                     alpha_slab, beta_slab, c_slab, pval_chunk_size):
    """Compute p-values for rows [row_start, row_end) on one GPU.

    Reads M_filt slab via mmap. Writes results directly into a shared
    memmap pvals file (preallocated by the parent), so no aggregation pass
    is needed afterward.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _enable_deterministic_torch()


    M_filt = np.load(M_filt_path, mmap_mode="r")
    pvals_mm = np.memmap(pvals_path, dtype=np.float32, mode='r+', shape=pvals_shape)

    n_rows = row_end - row_start
    logger.debug(f"[GPU {gpu_id}] computing p-values for rows {row_start}:{row_end} "
          f"({n_rows} genes, chunk={pval_chunk_size})")

    for s in range(0, n_rows, pval_chunk_size):
        e = min(s + pval_chunk_size, n_rows)
        M_chunk = np.asarray(M_filt[row_start + s : row_start + e]).astype(np.float32)
        c_chunk = c_slab[s:e]
        x_scaled = np.clip(
            M_chunk / c_chunk[:, None], 0.0, 1.0 - 1e-12
        ).astype(np.float32)

        pvals_chunk = (1.0 - sc_special.betainc(
            alpha_slab[s:e, None].astype(np.float64),
            beta_slab[s:e, None].astype(np.float64),
            x_scaled.astype(np.float64),
        )).astype(np.float32)

        np.clip(pvals_chunk, np.finfo(np.float32).tiny, 1.0, out=pvals_chunk)
        pvals_chunk[M_chunk == 0] = np.nan
        pvals_mm[row_start + s : row_start + e] = pvals_chunk
        logger.debug(f"[GPU {gpu_id}] {e}/{n_rows} done")

    pvals_mm.flush()
    del pvals_mm
    logger.debug(f"[GPU {gpu_id}] p-values written")


def _multi_gpu_pval_compute(M_filt, alpha_arr, beta_arr, c_arr,
                             n_gpus, pval_chunk_size):
    """Distribute p-value computation across N GPUs.

    Preallocates pvals_matrix as a memmap so workers write their slabs in
    place — avoids round-tripping a multi-GB array through pickle/disk twice.
    Returns the fully-populated pvals_matrix as a numpy array.
    """
    N, n_cols = M_filt.shape

    with tempfile.TemporaryDirectory(prefix="xrfm_pval_multigpu_") as tmp:
        M_path = os.path.join(tmp, "M_filt.npy")
        np.save(M_path, M_filt)

        pvals_path = os.path.join(tmp, "pvals.bin")
        pvals_mm = np.memmap(pvals_path, dtype=np.float32, mode='w+', shape=(N, n_cols))
        pvals_mm[:] = np.nan
        pvals_mm.flush()
        del pvals_mm

        slabs = np.array_split(np.arange(N), n_gpus)

        ctx = mp.get_context("spawn")
        processes = []
        for gpu_id, slab in enumerate(slabs):
            if len(slab) == 0:
                continue
            row_start = int(slab[0])
            row_end = int(slab[-1]) + 1
            slab_size = row_end - row_start
            worker_chunk = max(min(pval_chunk_size, slab_size // _CHUNKS_PER_WORKER),
                               min(_MIN_CHUNK, slab_size))
            p = ctx.Process(
                target=_gpu_pval_worker,
                args=(gpu_id, row_start, row_end, M_path,
                      pvals_path, (N, n_cols),
                      alpha_arr[row_start:row_end],
                      beta_arr[row_start:row_end],
                      c_arr[row_start:row_end],
                      worker_chunk),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [(i, p) for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            msgs = [f"GPU {i} (pid={p.pid}) exited with code {p.exitcode}"
                    for i, p in failed]
            raise RuntimeError("p-value worker(s) failed:\n  " + "\n  ".join(msgs))

        pvals_matrix = np.array(
            np.memmap(pvals_path, dtype=np.float32, mode='r', shape=(N, n_cols)))

    return pvals_matrix



def run_xrfm_inference(M, gene_names, model_dir, output_dir,
                       M_THRESHOLD=100, BAD_FIT_THRESHOLD=1e-1, FDR_ALPHA=0.05,
                       mu_filename=None, logkappa_filename=None):
    """Fit Beta(alpha, beta)*c to each gene's IDS values using xRFM 20HPO.

    Parameters
    ----------
    M           : (N_genes, N_genes) IDS matrix, zeros where cooc < threshold.
    gene_names  : list of gene name strings, length N_genes.
    model_dir   : path to directory containing 1M_20_HPO_xrfm_*.pkl files.
    output_dir  : where to save results (beta_fit_all_results.json, pvals.hkl, plots/).
    M_THRESHOLD : minimum nonzero observations to include a gene.
    BAD_FIT_THRESHOLD : mse_cdf above this is flagged (but still recorded).
    FDR_ALPHA   : BH-FDR alpha for calling significant neighbors.

    Returns
    -------
    results : list of dicts (one per gene, all genes recorded).
    """

    row_nnz = np.count_nonzero(M, axis=1)
    col_nnz = np.count_nonzero(M, axis=0)
    row_mask = row_nnz >= M_THRESHOLD
    col_mask = col_nnz >= M_THRESHOLD
    M_filt = M[row_mask, :][:, col_mask]
    filtered_gene_names_rows = [gene_names[i] for i, keep in enumerate(row_mask) if keep]
    filtered_gene_names_cols = [gene_names[i] for i, keep in enumerate(col_mask) if keep]
    n_genes, n_cols = M_filt.shape
    logger.info(f"Filtered to {n_genes} row-genes, {n_cols} col-genes (>= {M_THRESHOLD} nonzero obs)")

    sorted_data, m_values = _build_sorted_inf_matrix(M_filt)

    # Batch fit in outer gene chunks: split rows across all visible GPUs, or run
    # single-GPU/CPU when only one device is available.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    chunk_keys = ['c', 'k_lo', 'k', 'alpha', 'beta', 'cvm', 'n', 'ks_pval']

    if n_gpus > 1:
        logger.info(f"Multi-GPU batch fit: distributing {n_genes} genes across "
              f"{n_gpus} GPUs (chunk_size={GENE_CHUNK_SIZE})")
        xrfm_res = _multi_gpu_batch_fit(sorted_data, m_values, model_dir,
                                         mu_filename, logkappa_filename,
                                         n_gpus, GENE_CHUNK_SIZE)
    else:
        model_mu, model_lk = _load_xrfm_models(model_dir, mu_filename, logkappa_filename)
        all_chunks = {k: [] for k in chunk_keys}
        logger.info(f"Running batch fit in chunks of {GENE_CHUNK_SIZE} genes...")
        for start in tqdm(range(0, n_genes, GENE_CHUNK_SIZE)):
            end = min(start + GENE_CHUNK_SIZE, n_genes)
            res = _fit_from_matrix(sorted_data[start:end], m_values[start:end],
                                   model_mu, model_lk)
            for k in chunk_keys:
                all_chunks[k].append(res[k])
        xrfm_res = {k: np.concatenate(v) for k, v in all_chunks.items()}

    logger.info("Computing p-values...")
    alpha_arr = xrfm_res['alpha']
    beta_arr  = xrfm_res['beta']
    c_arr     = xrfm_res['c']

    PVAL_CHUNK = 2000

    if n_gpus > 1:
        logger.info(f"Multi-GPU p-values: distributing {n_genes} genes across {n_gpus} GPUs")
        pvals_matrix = _multi_gpu_pval_compute(
            M_filt, alpha_arr, beta_arr, c_arr, n_gpus, PVAL_CHUNK)
    else:
        pvals_matrix = np.full((n_genes, n_cols), np.nan, dtype=np.float32)
        for s in tqdm(range(0, n_genes, PVAL_CHUNK)):
            e = min(s + PVAL_CHUNK, n_genes)
            M_chunk = M_filt[s:e].astype(np.float32)
            c_chunk = c_arr[s:e]
            x_scaled = np.clip(
                M_chunk / c_chunk[:, None], 0.0, 1.0 - 1e-12
            ).astype(np.float32)

            pvals_chunk = (1.0 - sc_special.betainc(
                alpha_arr[s:e, None].astype(np.float64),
                beta_arr[s:e, None].astype(np.float64),
                x_scaled.astype(np.float64),
            )).astype(np.float32)

            np.clip(pvals_chunk, np.finfo(np.float32).tiny, 1.0, out=pvals_chunk)
            pvals_chunk[M_chunk == 0] = np.nan
            pvals_matrix[s:e] = pvals_chunk

    logger.info("Applying BH-FDR (vectorized)...")
    rejected_matrix = _bh_fdr_vectorized(pvals_matrix, alpha=FDR_ALPHA)

    TOTAL = n_genes
    NO_SIG_NEIGHBORS = 0
    results = []

    for i in range(n_genes):
        sig_idx = np.where(rejected_matrix[i])[0]
        significant_hit_genes = [filtered_gene_names_cols[j] for j in sig_idx]
        num_hits = len(significant_hit_genes)
        num_obs = int(np.sum(~np.isnan(pvals_matrix[i])))

        mse_val = float(xrfm_res['cvm'][i])
        cur = {
            "gene": filtered_gene_names_rows[i],
            "num_hits": num_hits,
            "alpha": float(alpha_arr[i]),
            "beta": float(beta_arr[i]),
            "scale": float(c_arr[i]),
            "mse_cdf": mse_val,
            "significant_hit_genes": significant_hit_genes,
            "num_nonzero": num_obs,
            "ks_pval": float(xrfm_res['ks_pval'][i]),
        }
        results.append(cur)

        if num_hits > 100:
            logger.debug(f"GENE WITH HIGH NEIGHBORS: {filtered_gene_names_rows[i]}")
        if np.isfinite(mse_val) and mse_val > BAD_FIT_THRESHOLD:
            logger.debug(f"GENE WITH BAD FIT: {filtered_gene_names_rows[i]}")
        if num_hits <= 1:
            NO_SIG_NEIGHBORS += 1

    pvals_df = pd.DataFrame(
        pvals_matrix,
        index=filtered_gene_names_rows,
        columns=filtered_gene_names_cols,
        dtype=float,
    )
    pvals_df = pvals_df.dropna(axis=1, how='all')
    pvals_path = os.path.join(output_dir, "pvals.hkl")
    hkl.dump(pvals_df, pvals_path)
    logger.info(f"Raw p-values DataFrame saved to {pvals_path}")

    alpha_vals = np.array([r["alpha"]   for r in results])
    beta_vals  = np.array([r["beta"]    for r in results])
    scale_vals = np.array([r["scale"]   for r in results])
    mse_cdfs   = np.array([r["mse_cdf"] for r in results])
    num_hits_arr = np.array([r["num_hits"] for r in results])

    fig, axs = plt.subplots(1, 5, figsize=(22, 4))
    axs[0].hist(alpha_vals, bins=40, color='C0', alpha=0.7)
    axs[0].set_title("Alpha values"); axs[0].set_xlabel("alpha"); axs[0].set_ylabel("Count")
    axs[0].grid(True, ls='--', lw=0.5)

    axs[1].hist(beta_vals, bins=40, color='C1', alpha=0.7)
    axs[1].set_title("Beta values"); axs[1].set_xlabel("beta"); axs[1].set_ylabel("Count")
    axs[1].grid(True, ls='--', lw=0.5)

    axs[2].hist(scale_vals, bins=40, color='C2', alpha=0.7)
    axs[2].set_title("Scale values"); axs[2].set_xlabel("scale"); axs[2].set_ylabel("Count")
    axs[2].grid(True, ls='--', lw=0.5)

    mse_plot = mse_cdfs[np.isfinite(mse_cdfs)]
    axs[3].hist(mse_plot, bins=40, color='C3', alpha=0.7)
    axs[3].set_title("MSE (Empirical vs Beta CDF)"); axs[3].set_xlabel("MSE CDF")
    axs[3].set_ylabel("Count"); axs[3].grid(True, ls='--', lw=0.5)

    axs[4].hist(num_hits_arr, bins=40, color='C4', alpha=0.7)
    axs[4].set_title("Number of significant neighbors")
    axs[4].set_xlabel(f"Num neighbors (FDR<{FDR_ALPHA})")
    axs[4].set_ylabel("Count"); axs[4].grid(True, ls='--', lw=0.5, axis='y')
    plt.tight_layout()

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(output_dir, "plots", "gene_param_distributions.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gene parameter distributions plot saved to {plot_path}")

    logger.info(f"Total genes: {TOTAL}")
    logger.info(f"Genes with no significant neighbors: {NO_SIG_NEIGHBORS}")

    return results



def create_adjacency_matrix(results, output_dir, MSE_THRESHOLD=1e-3):
    """Record the significant gene pairs as the graph that program finding searches."""
    low_mse_genes = set(
        r["gene"] for r in results
        if ("mse_cdf" in r) and (r["mse_cdf"] is not None) and (r["mse_cdf"] < MSE_THRESHOLD)
    )
    genes_with_hits = set()
    gene_to_hits = {}
    for r in results:
        gene = r.get("gene")
        if gene in low_mse_genes:
            hits = set(r.get("significant_hit_genes", []))
            gene_to_hits[gene] = hits
            genes_with_hits.update(hits)

    row_genes = sorted(low_mse_genes)
    col_genes = sorted(low_mse_genes.union(genes_with_hits))
    adj_matrix = pd.DataFrame(
        0,
        index=row_genes,
        columns=col_genes,
        dtype=int
    )
    for gene_i in low_mse_genes:
        hits = gene_to_hits.get(gene_i, set())
        for gene_j in hits:
            if gene_j in adj_matrix.columns:
                adj_matrix.at[gene_i, gene_j] = 1

    adj_matrix_path = os.path.join(output_dir, "significant_asymmetric_matrix.hkl")
    hkl.dump(adj_matrix, adj_matrix_path)




def statistical_program_finding(working_dir, COOCCUR_THRESHOLD=100, MSE_THRESHOLD=1e-2,
                                 P_VAL=0.05, MAX_SIGNIFICANT_NEIGHBORS=1000,
                                 model_dir=None,
                                 mu_filename=None, logkappa_filename=None):

    """Test every gene pair for significant dependence and save the resulting edges.

    Takes the IDS matrix from the dependence step, predicts each pair's null beta
    distribution, applies Benjamini-Hochberg correction across all tested pairs,
    and writes the directed significant-edge matrix that symmetrization and
    program finding consume.
    """
    adjacency_matrix_dir = os.path.join(working_dir, 'adjacency_matrices')
    if not os.path.exists(adjacency_matrix_dir):
        os.makedirs(adjacency_matrix_dir)

    supp_materials_dir = os.path.join(working_dir, 'supp_material')
    if not os.path.exists(supp_materials_dir):
        os.makedirs(supp_materials_dir)

    ids_cooccur_dir = os.path.join(working_dir, 'ids_cooccur')

    ids_path = os.path.join(ids_cooccur_dir, 'ids_matrix.hkl')
    cooccur_path = os.path.join(ids_cooccur_dir, 'cooccurrence_matrix.hkl')

    gene_names_dir  = os.path.join(working_dir, 'gene_names')
    gene_names_path = os.path.join(gene_names_dir, 'gene_names.npy')
    gene_names = np.load(gene_names_path, allow_pickle=True).tolist()

    M = load_and_filter_ids(ids_path, cooccur_path, gene_names,
                             COOCCUR_THRESHOLD=COOCCUR_THRESHOLD)

    pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(pipeline_root, "pipeline_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        spf_cfg = cfg.get("statistical_program_finding", {})
    else:
        spf_cfg = {}

    if model_dir is None:
        model_dir_cfg = spf_cfg.get("model_dir", "beta_fit_model/")
        if os.path.isabs(model_dir_cfg):
            model_dir = model_dir_cfg
        else:
            model_dir = os.path.join(pipeline_root, model_dir_cfg)

    if mu_filename is None:
        mu_filename = spf_cfg["model_mu_filename"]
    if logkappa_filename is None:
        logkappa_filename = spf_cfg["model_logkappa_filename"]

    logger.info(f"Using model directory: {model_dir}")
    logger.info(f"Using mu model: {mu_filename}")
    logger.info(f"Using logkappa model: {logkappa_filename}")

    results = run_xrfm_inference(M, gene_names, model_dir, supp_materials_dir,
                                 M_THRESHOLD=100, BAD_FIT_THRESHOLD=1e-1, FDR_ALPHA=P_VAL,
                                 mu_filename=mu_filename, logkappa_filename=logkappa_filename)

    # Split results: filtered (pass MSE, have hits, and not over the
    # significant-neighbor cap) go into the main JSON and adjacency matrix;
    # excluded go into a separate JSON for inspection. The cap removes
    # promiscuous "hub" genes whose row would otherwise dominate downstream
    # program finding.
    def _keep(r):
        return (
            r['mse_cdf'] < MSE_THRESHOLD
            and r['num_hits'] > 1
            and r['num_hits'] <= MAX_SIGNIFICANT_NEIGHBORS
        )
    results_ok       = [r for r in results if _keep(r)]
    results_excluded = [r for r in results if not _keep(r)]

    n_passed_mse = sum(1 for r in results
                       if r.get('mse_cdf') is not None and r['mse_cdf'] < MSE_THRESHOLD)
    logger.info("Genes: %d total, %d passed MSE < %g, %d kept with 2-%d neighbors",
                len(results), n_passed_mse, MSE_THRESHOLD, len(results_ok),
                MAX_SIGNIFICANT_NEIGHBORS)

    n_over_cap = sum(1 for r in results if r['num_hits'] > MAX_SIGNIFICANT_NEIGHBORS)
    logger.info(f"Excluded {n_over_cap} genes with > {MAX_SIGNIFICANT_NEIGHBORS} "
          f"significant neighbors (max_significant_neighbors cap)")

    ok_path = os.path.join(supp_materials_dir, "beta_fit_all_results.json")
    with open(ok_path, "w") as f:
        json.dump(results_ok, f, indent=2)
    logger.info(f"Filtered results ({len(results_ok)} genes) saved to {ok_path}")

    excl_path = os.path.join(supp_materials_dir, "beta_fit_excluded_results.json")
    with open(excl_path, "w") as f:
        json.dump(results_excluded, f, indent=2)
    logger.info(f"Excluded results ({len(results_excluded)} genes) saved to {excl_path}")

    create_adjacency_matrix(results_ok, adjacency_matrix_dir, MSE_THRESHOLD=MSE_THRESHOLD)
