"""Gene programs from the graph of significant gene-gene edges.

Three stages:
    1. Seed-anchored clique peeling over each gene's adjacency neighbourhood.
    2. Deduplicate cliques, cluster them by edge density with Louvain, and take a
       core, its cliques, and a penumbra from each community.
    3. Grow each core within its community, then split, merge and evaluate.

Reads
-----
    <working_dir>/adjacency_matrices/significant_symmetric_matrix.hkl
        Symmetric binary gene-gene adjacency (int8, zero diagonal).
    <working_dir>/supp_material/beta_fit_all_results.json
        Per-gene significant neighbours, used to seed stage 1.

Writes
------
    <working_dir>/<program_path_dir>/<program_path_name>   final programs
    <working_dir>/<program_path_dir>/neighborhood_cliques.json
    <working_dir>/<program_path_dir>/programs.json
    <working_dir>/<program_path_dir>/metrics.json
"""

import json
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import random
from collections import defaultdict
from itertools import combinations

import community as community_louvain
import hickle as hkl
import igraph as ig
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

try:
    import cudf
    import cugraph
    _CUGRAPH_AVAILABLE = True
except ImportError:
    _CUGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

_GROWTH_DENSITY_FLOOR    = 0.9
_LOUVAIN_RESOLUTION_S3   = 2.0
_CLUSTER_LOW_THRESHOLD   = 0.5
_SPLIT_THRESHOLD         = 0.5
_MERGE_THRESHOLD         = 0.8


def _compute_edge_density_matrix(gene_lists, adj, gene_to_idx):
    """NxN edge-density matrix between lists of genes.

    Builds a binary (n_cliques x n_genes) membership matrix M and computes
    edge_sums = M @ adj @ M^T; possible-pairs come from the clique sizes.
    """
    n = len(gene_lists)
    n_genes = adj.shape[0]

    idx_lists = [
        np.array([gene_to_idx[g] for g in genes if g in gene_to_idx], dtype=int)
        for genes in gene_lists
    ]
    sizes = np.array([len(il) for il in idx_lists], dtype=np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Edge density matrix: %d cliques on %s", n, device)

    rows, cols = [], []
    for i, il in enumerate(idx_lists):
        rows.extend([i] * len(il))
        cols.extend(il.tolist())
    M = torch.zeros(n, n_genes, dtype=torch.float32, device=device)
    if rows:
        M[rows, cols] = 1.0

    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

    edge_sums = (M @ adj_t @ M.T).cpu().numpy().astype(np.float64)

    del adj_t, M
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    possible_matrix = np.outer(sizes, sizes)
    valid = possible_matrix > 0
    dm = np.full((n, n), np.nan, dtype=np.float64)
    dm[valid] = edge_sums[valid] / possible_matrix[valid]

    for i in range(n):
        li = int(sizes[i])
        if li < 2:
            dm[i, i] = np.nan
            continue
        il = idx_lists[i]
        diag_sum = adj[il, il].sum()
        num_edges = (edge_sums[i, i] - diag_sum) / 2.0
        possible = li * (li - 1) / 2.0
        dm[i, i] = num_edges / possible if possible > 0 else np.nan

    zero_mask = sizes == 0
    dm[zero_mask, :] = np.nan
    dm[:, zero_mask] = np.nan

    return dm


def _build_neighbors_from_beta_fits(beta_fits_path):
    """Derive per-gene neighborhoods from beta_fit_all_results.json.

    Reads only genes that passed MSE + num_hits thresholding (the filtered
    'all_results' file, NOT the 'excluded' file). Uses 'significant_hit_genes'
    as the directed neighbor list for each source gene.
    """
    with open(beta_fits_path) as f:
        records = json.load(f)
    neighbors = {}
    for r in records:
        gene = r.get("gene")
        hits = r.get("significant_hit_genes", [])
        neighbors[gene] = hits
    return neighbors


def _peel_cliques(adj_matrix, gene_list_idx, gene_names, seed_gene, min_clique_size):
    """Extract seed-anchored maximal cliques from a gene neighborhood subgraph."""
    idxs = [gene_list_idx[g] for g in gene_names]
    sub_adj = adj_matrix[np.ix_(idxs, idxs)]
    n = len(gene_names)

    upper = np.triu(sub_adj, k=1)
    rows, cols = np.where(upper == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    G = ig.Graph(n=n, edges=edges, directed=False)

    name_to_v = {g: i for i, g in enumerate(gene_names)}
    v_to_name = {i: g for g, i in name_to_v.items()}
    seed_v = name_to_v[seed_gene]
    active = set(range(n))

    cliques = []
    while True:
        if len(active) < min_clique_size or seed_v not in active:
            break

        sub_vids = sorted(active)
        sub = G.subgraph(sub_vids)
        orig_to_sub = {o: i for i, o in enumerate(sub_vids)}
        seed_sub = orig_to_sub[seed_v]

        nbrs_in_sub = sub.neighbors(seed_sub)
        seed_closed = sorted(set(nbrs_in_sub) | {seed_sub})
        if len(seed_closed) < min_clique_size:
            break

        nbr_sub = sub.subgraph(seed_closed)
        largest = nbr_sub.largest_cliques()
        if not largest:
            break
        max_size = len(largest[0])
        if max_size < min_clique_size:
            break

        candidates = []
        for c in largest:
            gene_names_c = sorted(v_to_name[sub_vids[seed_closed[v]]] for v in c)
            if seed_gene in gene_names_c:
                candidates.append(gene_names_c)

        if not candidates:
            break
        # Deterministic tie-break: lex-smallest sorted clique
        candidates.sort()
        clique = candidates[0]
        cliques.append(clique)

        for g in clique:
            if g != seed_gene:
                active.discard(name_to_v[g])

    return cliques


def _process_gene_cliques(args):
    """Worker function for parallel Stage 1 clique peeling."""
    gene, nbrs, adj_genes_set, gene_list, gene_list_idx, shm_name, adj_shape, min_clique_size = args

    shm = shared_memory.SharedMemory(name=shm_name)
    adj_matrix = np.ndarray(adj_shape, dtype=np.int8, buffer=shm.buf)

    neighborhood = [g for g in nbrs if g in adj_genes_set]
    if gene in adj_genes_set and gene not in neighborhood:
        neighborhood.append(gene)

    if len(neighborhood) < min_clique_size:
        shm.close()
        return gene, {"neighborhood_size": len(neighborhood), "cliques": []}

    cliques = _peel_cliques(adj_matrix, gene_list_idx, neighborhood,
                            seed_gene=gene, min_clique_size=min_clique_size)
    shm.close()
    return gene, {"neighborhood_size": len(neighborhood), "cliques": cliques}


def _find_neighborhood_cliques(adj_df, neighbors, min_clique_size):
    """Stage 1: For each gene, peel seed-anchored cliques from its neighborhood.

    Parallelized across genes using shared memory for the adjacency matrix.
    """
    adj_matrix = adj_df.values.astype(np.int8)
    gene_list = list(adj_df.index)
    gene_list_idx = {g: i for i, g in enumerate(gene_list)}
    adj_genes_set = set(gene_list)
    adj_shape = adj_matrix.shape

    n_workers = min(mp.cpu_count(), 16)

    shm = shared_memory.SharedMemory(create=True, size=adj_matrix.nbytes)
    shared_arr = np.ndarray(adj_shape, dtype=np.int8, buffer=shm.buf)
    np.copyto(shared_arr, adj_matrix)

    work_items = [
        (gene, nbrs, adj_genes_set, gene_list, gene_list_idx, shm.name, adj_shape, min_clique_size)
        for gene, nbrs in neighbors.items()
    ]

    collected = {}
    with mp.Pool(n_workers) as pool:
        for gene, result in tqdm(
            pool.imap_unordered(_process_gene_cliques, work_items),
            total=len(work_items),
            desc=f"Stage 1: neighborhood cliques ({n_workers} workers)"
        ):
            collected[gene] = result

    shm.close()
    shm.unlink()

    # Preserve original neighbor iteration order for downstream determinism
    results = {gene: collected[gene] for gene in neighbors}

    genes_with_cliques = sum(1 for v in results.values() if v["cliques"])
    total_cliques      = sum(len(v["cliques"]) for v in results.values())
    skipped = sum(1 for v in results.values() if not v["cliques"] and v["neighborhood_size"] < min_clique_size)
    logger.info("Stage 1: %d genes, %d skipped (<min_size), %d with cliques, %d total cliques",
                len(results), skipped, genes_with_cliques, total_cliques)
    return results


def _deduplicate_cliques(clique_results):
    """Exact dedup then remove subsumed cliques via inverted gene index."""
    seen, unique = set(), []
    for entry in clique_results.values():
        for clique in entry["cliques"]:
            key = tuple(sorted(clique))
            if key not in seen:
                seen.add(key)
                unique.append(list(key))

    unique.sort(key=len, reverse=True)

    gene_to_cliques = defaultdict(set)
    for i, c in enumerate(unique):
        for g in c:
            gene_to_cliques[g].add(i)

    keep_mask = [True] * len(unique)
    for i, c in enumerate(unique):
        common = gene_to_cliques[c[0]].copy()
        for g in c[1:]:
            common &= gene_to_cliques[g]
            if not common:
                break
        this_len = len(c)
        for j in common:
            if j < i and len(unique[j]) > this_len:
                keep_mask[i] = False
                break

    keep = [c for i, c in enumerate(unique) if keep_mask[i]]
    logger.info("Stage 2 dedup: %d unique → %d after removing subsumed", len(unique), len(keep))
    return keep


_LOUVAIN_BACKEND = "gpu"


def _build_nx_from_density(density_matrix):
    """Build a weighted NetworkX graph from an NxN density matrix."""
    n = density_matrix.shape[0]
    upper = np.triu(density_matrix, k=1)
    mask = np.isfinite(upper) & (upper > 0)
    rows, cols = np.where(mask)
    weights = upper[rows, cols]
    logger.info("Building networkx graph: %d nodes, %d edges", n, len(rows))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(zip(rows.tolist(), cols.tolist(), weights.tolist()))
    return G


def _build_cugraph_from_density(density_matrix):
    """Build a cugraph Graph from an NxN density matrix."""
    upper = np.triu(density_matrix, k=1)
    mask = np.isfinite(upper) & (upper > 0)
    rows, cols = np.where(mask)
    weights = upper[rows, cols]
    logger.info("Building cugraph: %d nodes, %d edges", density_matrix.shape[0], len(rows))
    gdf = cudf.DataFrame({
        'src': rows.astype(np.int32),
        'dst': cols.astype(np.int32),
        # float64: cugraph sums edge weights in a nondeterministic parallel
        # order, and float32 rounding is large enough to tip near-tie modularity
        # gains. float64 keeps the partition reproducible.
        'weight': weights.astype(np.float64),
    })
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='weight')
    return G


def _louvain_partition(density_matrix, resolution):
    """Run Louvain on the density graph using the configured backend.

    Returns a dict mapping every vertex id (0..n-1) to a community id.
    """
    n = density_matrix.shape[0]
    backend = _LOUVAIN_BACKEND
    if backend == "gpu" and not _CUGRAPH_AVAILABLE:
        logger.warning("cugraph unavailable; falling back to CPU Louvain")
        backend = "cpu"

    if backend == "cpu":
        G = _build_nx_from_density(density_matrix)
        random.seed(SEED)
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=SEED
        )
        if len(partition) < n:
            next_comm = (int(max(partition.values())) + 1) if partition else 0
            for v in range(n):
                if v not in partition:
                    partition[v] = next_comm
                    next_comm += 1
        logger.info("CPU Louvain: %d communities", len(set(partition.values())))
        return partition

    G = _build_cugraph_from_density(density_matrix)
    parts, modularity = cugraph.louvain(G, resolution=resolution)
    pdf = parts.to_pandas()
    partition = dict(zip(pdf['vertex'].values, pdf['partition'].values))
    if len(partition) < n:
        next_comm = (int(max(partition.values())) + 1) if partition else 0
        n_singletons = 0
        for v in range(n):
            if v not in partition:
                partition[v] = next_comm
                next_comm += 1
                n_singletons += 1
        logger.info("GPU Louvain: %d communities (+%d singletons from isolated nodes), modularity=%.4f",
                    len(set(partition.values())), n_singletons, modularity)
    else:
        logger.info("GPU Louvain: %d communities, modularity=%.4f",
                    len(set(partition.values())), modularity)
    return partition


def _remove_bridges_from_density(density_matrix):
    """Zero out any bridge edges in the density graph.

    A bridge is an edge whose removal disconnects its connected component.
    In the clique-similarity graph, a single false-positive edge between two
    otherwise-separate clique clusters acts as a bridge and would cause
    Louvain to merge them into one community.

    Returns a copy of density_matrix with bridge edges set to 0.
    """
    n = density_matrix.shape[0]
    upper = np.triu(density_matrix, k=1)
    mask = np.isfinite(upper) & (upper > 0)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return density_matrix

    edges = list(zip(rows.tolist(), cols.tolist()))
    G = ig.Graph(n=n, edges=edges, directed=False)
    bridge_eids = G.bridges()
    if not bridge_eids:
        return density_matrix

    bridge_endpoints = [G.es[eid].tuple for eid in bridge_eids]
    dm = density_matrix.copy()
    for u, v in bridge_endpoints:
        dm[u, v] = 0.0
        dm[v, u] = 0.0
    logger.info("Removed %d bridge edges before Louvain", len(bridge_endpoints))
    return dm


def _louvain_on_density(density_matrix, resolution):
    """Stage 2 Louvain. Bridges are removed first so a single FP edge can't
    merge otherwise-separate clique clusters into one community.
    """
    dm = _remove_bridges_from_density(density_matrix)
    return _louvain_partition(dm, resolution)


def _largest_clique_lex_max(G_ig, vid_to_gene):
    """Return the lex-MAX max-size clique as sorted gene list (for _extract_programs)."""
    largest = G_ig.largest_cliques()
    if not largest:
        return []
    candidates = [tuple(sorted(vid_to_gene[v] for v in c)) for c in largest]
    return list(max(candidates))


def _compute_clique_periphery(clique_genes, candidate_pool, adj, gene_to_idx):
    """Genes in candidate_pool with edges to at least 50% of clique_genes.

    Threshold is inclusive (≥). Genes missing from the adjacency matrix are
    skipped. This is a frozen, one-shot computation: callers store the result
    and propagate it through later stages without recomputation.
    """
    if not clique_genes:
        return []
    half        = 0.5 * len(clique_genes)
    clique_idxs = [gene_to_idx[g] for g in clique_genes if g in gene_to_idx]
    if not clique_idxs:
        return []
    out = []
    for g in candidate_pool:
        gi = gene_to_idx.get(g)
        if gi is None:
            continue
        if adj[gi, clique_idxs].sum() >= half:
            out.append(g)
    return sorted(out)


def _extract_programs(cliques, partition, adj_df, min_clique_size):
    """For each Louvain community: union genes → peel core + remaining cliques.

    Each peeled clique (core included) is annotated with a frozen periphery:
    the genes in the community's penumbra with edges to ≥50% of the clique's
    genes. Peripheries are immutable from this point on — Stage 3 only unions
    them on merge or duplicates them on split.
    """
    communities = defaultdict(list)
    for clique_idx, comm_id in partition.items():
        communities[comm_id].append(clique_idx)

    adj        = adj_df.values
    gene_to_idx = {g: i for i, g in enumerate(adj_df.index)}
    adj_genes  = set(adj_df.index)

    def _density(genes):
        if len(genes) < 2:
            return float('nan')
        idxs = np.array([gene_to_idx[g] for g in genes])
        sub  = adj[np.ix_(idxs, idxs)]
        return float(np.triu(sub, k=1).sum() / (len(genes) * (len(genes) - 1) / 2))

    programs = []
    for comm_id in sorted(communities):
        gene_union = sorted(set(g for ci in communities[comm_id]
                                for g in cliques[ci] if g in adj_genes))
        if len(gene_union) < 2:
            continue

        idxs    = [gene_to_idx[g] for g in gene_union]
        sub_adj = adj[np.ix_(idxs, idxs)]
        upper   = np.triu(sub_adj, k=1)
        rows, cols = np.where(upper == 1)
        edges   = list(zip(rows.tolist(), cols.tolist()))
        sub_G   = ig.Graph(n=len(gene_union), edges=edges, directed=False)

        vid_to_gene = {i: g for i, g in enumerate(gene_union)}
        gene_to_vid = {g: i for i, g in enumerate(gene_union)}

        core_genes = _largest_clique_lex_max(sub_G, vid_to_gene)

        peeled_cliques = []
        remaining_set = set(gene_union) - set(core_genes)
        while len(remaining_set) >= min_clique_size:
            remaining_vids = sorted(gene_to_vid[g] for g in remaining_set)
            rem_G = sub_G.subgraph(remaining_vids)
            rem_vid_to_gene = {i: vid_to_gene[remaining_vids[i]] for i in range(len(remaining_vids))}
            clique = _largest_clique_lex_max(rem_G, rem_vid_to_gene)
            if len(clique) < min_clique_size:
                break
            peeled_cliques.append(sorted(clique))
            remaining_set -= set(clique)
        remaining = sorted(remaining_set)

        core_periphery = _compute_clique_periphery(core_genes, remaining, adj, gene_to_idx)
        clique_peripheries = [
            _compute_clique_periphery(pc, remaining, adj, gene_to_idx)
            for pc in peeled_cliques
        ]
        program_periphery = sorted(set(core_periphery).union(*clique_peripheries))

        programs.append({
            "community_id":         int(comm_id),
            "core_genes":           core_genes,
            "core_size":            len(core_genes),
            "core_density":         _density(core_genes),
            "core_periphery":       core_periphery,
            "cliques":              [{"genes": pc, "size": len(pc), "density": _density(pc),
                                      "periphery": cper}
                                     for pc, cper in zip(peeled_cliques, clique_peripheries)],
            "num_cliques":          len(peeled_cliques),
            "num_clique_genes":     sum(len(pc) for pc in peeled_cliques),
            "penumbra_genes":       remaining,
            "penumbra_size":        len(remaining),
            "periphery_genes":      program_periphery,
            "periphery_size":       len(program_periphery),
            "total_size":           len(gene_union),
            "full_density":         _density(gene_union),
            "num_input_cliques":    len(communities[comm_id]),
        })

    programs.sort(key=lambda p: p["total_size"], reverse=True)
    return programs


def _cluster_and_extract_programs(clique_results, adj_df, resolution, min_clique_size):
    """Stage 2: Deduplicate → edge-density Louvain → extract core/cliques programs."""
    cliques = _deduplicate_cliques(clique_results)
    if not cliques:
        logger.error("Stage 2: No cliques found. Check Stage 1 output.")
        return []

    adj         = adj_df.values
    gene_to_idx = {g: i for i, g in enumerate(adj_df.index)}

    logger.info("Stage 2: Computing edge density for %d cliques ...", len(cliques))
    dm = _compute_edge_density_matrix(cliques, adj, gene_to_idx)

    logger.info("Stage 2: Running Louvain (resolution=%.2f) ...", resolution)
    partition    = _louvain_on_density(dm, resolution)
    n_communities = len(set(partition.values()))
    logger.info("Stage 2: %d communities", n_communities)

    programs = _extract_programs(cliques, partition, adj_df, min_clique_size)
    logger.info("Stage 2: Extracted %d programs", len(programs))
    return programs


def _grow_cliques(cliques, adj_df, growth_threshold, gene_pools=None):
    """Two-round growth for each seed clique.

    If gene_pools is provided, each clique can only recruit from its
    corresponding gene pool (penumbra of its community).  Otherwise
    falls back to the full adjacency matrix (original behaviour).

    Returns one grown gene list per input clique, in the same order.
    Peripheries are tracked outside this function (frozen at Stage 2D) and
    follow each grown clique by index.
    """
    A         = adj_df
    all_genes = list(A.index)

    round1 = []
    for i, clique in enumerate(cliques):
        clique_set = set(clique)
        if gene_pools is not None:
            candidate_genes = [g for g in gene_pools[i] if g not in clique_set]
        else:
            candidate_genes = [g for g in all_genes if g not in clique_set]

        if not candidate_genes:
            round1.append(sorted(clique_set))
            continue

        sub        = A.loc[candidate_genes, clique]
        connectivity = np.sum(sub.values, axis=1) / len(clique)

        eligible = sorted(
            [(g, c) for g, c in zip(candidate_genes, connectivity) if c > growth_threshold],
            key=lambda x: -x[1]
        )
        selected = []
        for gene, _ in eligible:
            test = selected + [gene]
            n_t  = len(test)
            if n_t > 1:
                sub_A            = A.loc[test, test].values
                upper            = np.triu_indices(n_t, k=1)
                internal_density = sub_A[upper].sum() / (n_t * (n_t - 1) / 2)
            else:
                internal_density = 0
            if internal_density >= _GROWTH_DENSITY_FLOOR or n_t <= 1:
                selected.append(gene)
            else:
                break
        round1.append(sorted(clique_set | set(selected)))

    round2 = []
    for i, grown_clique in enumerate(round1):
        expanded = set(grown_clique)
        if gene_pools is not None:
            candidate_genes = [g for g in gene_pools[i] if g not in expanded]
        else:
            candidate_genes = [g for g in all_genes if g not in expanded]

        if not candidate_genes:
            round2.append(sorted(expanded))
            continue

        sub      = A.loc[candidate_genes, grown_clique]
        connectivity = np.sum(sub.values, axis=1) / len(grown_clique)
        eligible = [g for g, c in zip(candidate_genes, connectivity) if c > growth_threshold]
        round2.append(sorted(expanded | set(eligible)))

    logger.info("Stage 3 grow: after round 2, program sizes: %s",
                sorted([len(p) for p in round2], reverse=True)[:10])
    return round2


def _louvain_cluster_with_pruning(density_matrix, cliques, peripheries, adj_df, resolution):
    """Louvain on grown programs + iterative pruning of low-density clusters.

    Each cluster's kept programs are unioned (genes + peripheries).
    Pruned-out programs keep their own (genes, periphery) as singletons.
    Returns a list of (gene_list, periphery_set) tuples.
    """
    partition = _louvain_partition(density_matrix, resolution)

    cluster_to_indices = defaultdict(list)
    for idx, cid in partition.items():
        cluster_to_indices[cid].append(idx)

    adj          = adj_df.values
    gene_to_idx  = {g: i for i, g in enumerate(adj_df.index)}
    clusters     = []

    for cid in tqdm(sorted(cluster_to_indices), desc="Stage 3: Louvain pruning"):
        indices    = cluster_to_indices[cid]
        keep_idxs  = set(indices)

        while len(keep_idxs) > 1:
            genes = sorted(set(g for idx in keep_idxs
                               for g in cliques[idx] if g in gene_to_idx))
            if len(genes) < 2:
                cluster_density = 0.0
            else:
                g_idxs  = [gene_to_idx[g] for g in genes]
                sub_adj = adj[np.ix_(g_idxs, g_idxs)]
                ne      = np.triu(sub_adj, k=1).sum()
                np_     = len(genes) * (len(genes) - 1) / 2
                cluster_density = ne / np_ if np_ > 0 else 0.0
            if cluster_density >= _CLUSTER_LOW_THRESHOLD:
                break

            clique_conn = {}
            for idx in keep_idxs:
                others    = [j for j in keep_idxs if j != idx]
                conn_vals = [density_matrix[idx, j] for j in others
                             if not np.isnan(density_matrix[idx, j])]
                clique_conn[idx] = np.mean(conn_vals) if conn_vals else 0.0

            to_remove = min(clique_conn, key=lambda idx: (clique_conn[idx], idx))
            keep_idxs.remove(to_remove)

        if len(keep_idxs) == 1:
            i = next(iter(keep_idxs))
            clusters.append((sorted(cliques[i]), set(peripheries[i])))
        elif len(keep_idxs) > 1:
            merged_genes  = sorted(set(g for idx in keep_idxs for g in cliques[idx]))
            merged_peri   = set().union(*(peripheries[idx] for idx in keep_idxs))
            clusters.append((merged_genes, merged_peri))

        for idx in set(indices) - keep_idxs:
            clusters.append((sorted(cliques[idx]), set(peripheries[idx])))

    return clusters


def _find_maximal_cliques_greedy(genes, adj_df):
    """Greedy peel: repeatedly extract the largest maximal clique."""
    if not genes:
        return []
    genes = list(genes)
    adj_subset = adj_df.loc[genes, genes].values
    n = len(genes)

    upper = np.triu(adj_subset, k=1)
    rows, cols = np.where(upper == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    G = ig.Graph(n=n, edges=edges, directed=False)

    vid_to_gene = {i: g for i, g in enumerate(genes)}
    gene_to_vid = {g: i for i, g in enumerate(genes)}

    cliques = []
    remaining = set(range(n))
    while remaining:
        sub_vids = sorted(remaining)
        sub = G.subgraph(sub_vids)
        if sub.ecount() == 0:
            cliques.extend([vid_to_gene[v]] for v in sub_vids)
            break
        largest = sub.largest_cliques()
        if not largest:
            cliques.extend([vid_to_gene[v]] for v in sub_vids)
            break
        # Deterministic tie-break: lex-smallest gene tuple among the largest cliques
        candidates = [sorted(vid_to_gene[sub_vids[v]] for v in c) for c in largest]
        candidates.sort()
        max_clique = candidates[0]
        cliques.append(max_clique)
        remaining -= {gene_to_vid[g] for g in max_clique}
    return cliques


def _visualize_maximal_cliques_in_programs(programs, adj_df, out_path):
    """Block-diagonal heatmap of the adjacency matrix grouped by program.

    For each program, genes are ordered by greedy-peeled maximal cliques
    (largest first, descending), then any remaining program genes are
    appended in sorted order — so every gene in the program is drawn,
    not only those that fell into a peeled clique. Red lines divide
    programs; tick labels show the program index.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    valid = set(adj_df.index)
    all_genes, program_boundaries, program_labels = [], [], []
    for pi, prog in enumerate(programs):
        prog_in = [g for g in prog if g in valid]
        if prog_in:
            cliques = _find_maximal_cliques_greedy(prog_in, adj_df)
            used = set()
            for clique in cliques:
                all_genes.extend(clique)
                program_labels.extend([pi] * len(clique))
                used.update(clique)
            leftover = sorted(g for g in prog_in if g not in used)
            all_genes.extend(leftover)
            program_labels.extend([pi] * len(leftover))
        program_boundaries.append(len(all_genes))

    if not all_genes:
        logger.warning("No genes to plot for maximal-cliques heatmap; skipping.")
        return

    adj_subset = adj_df.loc[all_genes, all_genes].values
    N = max(1, len(all_genes) // 40)
    tick_labels = [str(l) if (i % N == 0) else "" for i, l in enumerate(program_labels)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    sns.heatmap(
        adj_subset, cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels,
        ax=ax, cbar=False,
    )
    for boundary in program_boundaries[:-1]:
        plt.axhline(boundary - 0.5, color="red", linewidth=0.3)
        plt.axvline(boundary - 0.5, color="red", linewidth=0.3)
    plt.title(f"Adjacency Matrix Heatmap with Program Boundaries\n"
              f"{len(programs)} programs, {len(all_genes)} genes")
    plt.xlabel("Program index (for each gene, by order)")
    plt.ylabel("Program index (for each gene, by order)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    logger.info("Saved maximal-cliques heatmap: %s", out_path)


def _split_low_connectivity_cliques(programs, adj_df, threshold):
    """Split off low-connectivity sub-cliques as their own programs.

    Periphery rule (option 3): every resulting piece — the kept remainder
    AND each split-off sub-clique — inherits the parent program's full
    periphery (duplicated).
    """
    new_programs = []
    for genes, peri in programs:
        if len(genes) < 5:
            new_programs.append((genes, set(peri)))
            continue
        cliques = _find_maximal_cliques_greedy(genes, adj_df)
        if len(cliques) <= 1:
            new_programs.append((genes, set(peri)))
            continue
        keep, splits = [], []
        for clique in cliques:
            rest = [g for g in genes if g not in clique]
            if not rest:
                keep.append(clique)
                continue
            sub  = adj_df.loc[clique, rest].values
            conn = sub.sum() / (len(clique) * len(rest))
            if conn < threshold and len(clique) > 5:
                splits.append(clique)
            else:
                keep.append(clique)
        if keep:
            new_programs.append((sorted(set(g for c in keep for g in c)), set(peri)))
        for c in splits:
            new_programs.append((sorted(c), set(peri)))
    logger.info("Stage 3 split: %d → %d programs", len(programs), len(new_programs))
    return new_programs


def _merge_high_connectivity(programs, adj_df, threshold):
    """Greedy edge-density merge. Genes and peripheries unioned on merge."""
    working = {i: (list(g), set(p)) for i, (g, p) in enumerate(programs)}
    while True:
        items = list(working.items())
        n     = len(items)
        if n <= 1:
            break
        best_conn, best_pair = threshold, None
        for i in range(n):
            gi = items[i][1][0]
            for j in range(i + 1, n):
                gj   = items[j][1][0]
                sub  = adj_df.loc[gi, gj].values
                conn = sub.sum() / (len(gi) * len(gj)) if gi and gj else 0.0
                if conn > best_conn:
                    best_conn, best_pair = conn, (i, j)
        if best_pair is None:
            break
        ki, kj   = items[best_pair[0]][0], items[best_pair[1]][0]
        gi, pi   = working[ki]
        gj, pj   = working[kj]
        merged_g = sorted(set(gi + gj))
        merged_p = pi | pj
        logger.info("  Edge-density merge: %d + %d → %d", len(gi), len(gj), len(merged_g))
        del working[ki], working[kj]
        working[max(working, default=0) + 1] = (merged_g, merged_p)
    return list(working.values())


def _merge_overlapping(programs, threshold):
    """Greedy containment merge. Genes and peripheries unioned on merge."""
    working = {i: (set(g), set(p)) for i, (g, p) in enumerate(programs)}
    while True:
        best_pair, best_overlap = None, threshold
        keys = list(working.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj   = keys[i], keys[j]
                gi, _    = working[ki]
                gj, _    = working[kj]
                inter    = len(gi & gj)
                min_size = min(len(gi), len(gj))
                if min_size == 0:
                    continue
                overlap = inter / min_size
                if overlap > best_overlap:
                    best_overlap, best_pair = overlap, (ki, kj)
        if best_pair is None:
            break
        ki, kj  = best_pair
        gi, pi  = working[ki]
        gj, pj  = working[kj]
        merged_g = gi | gj
        merged_p = pi | pj
        logger.info("  Containment merge: %d + %d → %d (overlap %.3f)",
                    len(gi), len(gj), len(merged_g), best_overlap)
        working[ki] = (merged_g, merged_p)
        del working[kj]
    logger.info("Stage 3 containment merge: %d → %d programs", len(programs), len(working))
    return [(sorted(g), p) for (g, p) in working.values()]


def _compute_evaluation_metrics(programs, adj, gene_to_idx, total_genes):
    gene_slots   = sum(len(p) for p in programs)
    unique_genes = set(g for p in programs for g in p)
    n_unique     = len(unique_genes)
    overlap_ratio = gene_slots / n_unique if n_unique > 0 else float('inf')

    max_jaccard, max_pair = 0.0, (-1, -1)
    sets = [set(p) for p in programs]
    for i, j in combinations(range(len(programs)), 2):
        inter = len(sets[i] & sets[j])
        union = len(sets[i] | sets[j])
        if union > 0:
            jacc = inter / union
            if jacc > max_jaccard:
                max_jaccard, max_pair = jacc, (i, j)

    densities = []
    for prog in programs:
        idxs = np.array([gene_to_idx[g] for g in prog if g in gene_to_idx], dtype=int)
        n = len(idxs)
        if n >= 2:
            sub = adj[np.ix_(idxs, idxs)]
            densities.append(float(np.triu(sub, k=1).sum() / (n * (n - 1) / 2)))
    median_density = float(np.median(densities)) if densities else 0.0

    return {
        "program_count":           len(programs),
        "gene_slots":              gene_slots,
        "unique_genes":            n_unique,
        "overlap_ratio":           round(overlap_ratio, 3),
        "max_pairwise_jaccard":    round(max_jaccard, 4),
        "max_jaccard_pair":        list(max_pair),
        "median_internal_density": round(median_density, 4),
        "gene_coverage":           round(n_unique / total_genes, 4) if total_genes > 0 else 0.0,
        "program_sizes":           sorted([len(p) for p in programs], reverse=True),
    }


def _grow_and_merge(stage2_programs, adj_df, growth_threshold, containment_merge_threshold):
    """Stage 3: Grow → Louvain → Split → Merge (edge-density) → Merge (containment)."""
    adj         = adj_df.values
    gene_to_idx = {g: i for i, g in enumerate(adj_df.index)}

    input_cliques     = []
    input_peripheries = []
    clique_penumbras  = []
    for p in stage2_programs:
        penumbra = p.get("penumbra_genes", [])
        input_cliques.append(p["core_genes"])
        input_peripheries.append(set(p.get("core_periphery", [])))
        clique_penumbras.append(penumbra)
        for pc in p["cliques"]:
            input_cliques.append(pc["genes"])
            input_peripheries.append(set(pc.get("periphery", [])))
            clique_penumbras.append(penumbra)
    logger.info("Stage 3: %d input cliques from Stage 2", len(input_cliques))

    logger.info("Step 1: Grow cliques (growth_threshold=%.3f, community-constrained)", growth_threshold)
    grown_genes = _grow_cliques(input_cliques, adj_df, growth_threshold,
                                gene_pools=clique_penumbras)
    grown_pairs = sorted(
        zip(grown_genes, input_peripheries),
        key=lambda gp: (-len(gp[0]), sorted(gp[0]))
    )
    grown_only_genes = [g for g, _ in grown_pairs]
    grown_peris      = [p for _, p in grown_pairs]

    logger.info("Step 2: Louvain clustering (resolution=%.1f) + pruning", _LOUVAIN_RESOLUTION_S3)
    dm        = _compute_edge_density_matrix(grown_only_genes, adj, gene_to_idx)
    clustered = sorted(
        _louvain_cluster_with_pruning(dm, grown_only_genes, grown_peris,
                                       adj_df, _LOUVAIN_RESOLUTION_S3),
        key=lambda gp: (-len(gp[0]), sorted(gp[0]))
    )
    logger.info("After Louvain: %d programs", len(clustered))

    logger.info("Step 3: Split low-connectivity cliques (threshold=%.2f)", _SPLIT_THRESHOLD)
    split = sorted(
        _split_low_connectivity_cliques(clustered, adj_df, _SPLIT_THRESHOLD),
        key=lambda gp: (-len(gp[0]), sorted(gp[0]))
    )

    logger.info("Step 4: Merge high-connectivity (threshold=%.2f)", _MERGE_THRESHOLD)
    merged = sorted(_merge_high_connectivity(split, adj_df, _MERGE_THRESHOLD),
                    key=lambda gp: (-len(gp[0]), sorted(gp[0])))

    logger.info("Step 4b: Containment merge (threshold=%.2f)", containment_merge_threshold)
    final = sorted(_merge_overlapping(merged, containment_merge_threshold),
                   key=lambda gp: (-len(gp[0]), sorted(gp[0])))

    final_genes = [g for g, _ in final]
    metrics = _compute_evaluation_metrics(final_genes, adj, gene_to_idx, len(adj_df.index))
    logger.info("Stage 3 complete: %d programs, overlap_ratio=%.3f, median_density=%.4f",
                len(final), metrics["overlap_ratio"], metrics["median_internal_density"])
    return final, metrics


def run_program_finder(
    working_dir,
    MIN_CLIQUE_SIZE=5,
    GROWTH_THRESHOLD=0.9,
    CLUSTER_START_RESOLUTION=1.0,
    MIN_PROGRAM_SIZE=5,
    PROGRAM_PATH_DIR="clique_based_programs",
    PROGRAM_PATH_NAME="programs_after_grow_merge.json",
    CONTAINMENT_MERGE_THRESHOLD=0.7,
    LOUVAIN_BACKEND="cpu",
    beta_fits_path=None,
):
    """
    Run the 3-stage neighborhood clique program finding pipeline.

    Parameters:
      MIN_CLIQUE_SIZE              — minimum clique size (all stages)
      GROWTH_THRESHOLD             — growth connectivity threshold (Stage 3)
      CLUSTER_START_RESOLUTION     — Louvain resolution for Stage 2 community detection
      CONTAINMENT_MERGE_THRESHOLD  — overlap threshold for containment merge (Stage 3)
      MIN_PROGRAM_SIZE             — minimum program size in final output
      PROGRAM_PATH_DIR / NAME      — output location for final programs JSON
      LOUVAIN_BACKEND              — "gpu" (cugraph) or "cpu" (python-louvain).
    """
    global _LOUVAIN_BACKEND
    backend = (LOUVAIN_BACKEND or "cpu").lower()
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"LOUVAIN_BACKEND must be 'cpu' or 'gpu', got: {LOUVAIN_BACKEND!r}")
    _LOUVAIN_BACKEND = backend
    logger.info("Louvain backend: %s (options: cpu, gpu)", backend)
    adj_path = os.path.join(working_dir, 'adjacency_matrices', 'significant_symmetric_matrix.hkl')
    out_dir  = os.path.join(working_dir, PROGRAM_PATH_DIR) if PROGRAM_PATH_DIR else working_dir
    os.makedirs(out_dir, exist_ok=True)

    if beta_fits_path is None:
        beta_fits_path = os.path.join(working_dir, 'supp_material', 'beta_fit_all_results.json')

    logger.info("Loading adjacency matrix from %s", adj_path)
    try:
        adj_df = hkl.load(adj_path)
    except (ValueError, NotImplementedError):
        import pandas as pd
        adj_dir = os.path.dirname(adj_path)
        vals = np.load(os.path.join(adj_dir, 'sym_adj_values.npy'))
        genes = np.loadtxt(os.path.join(adj_dir, 'sym_adj_genes.txt'), dtype=str)
        adj_df = pd.DataFrame(vals, index=genes, columns=genes)
        logger.info("Loaded from numpy fallback")
    logger.info("Adjacency matrix shape: %s", adj_df.shape)

    logger.info("STAGE 1: Building neighbors + finding neighborhood cliques")
    logger.info("Loading neighbors from beta fits: %s", beta_fits_path)
    neighbors     = _build_neighbors_from_beta_fits(beta_fits_path)
    clique_results = _find_neighborhood_cliques(adj_df, neighbors, MIN_CLIQUE_SIZE)
    cliques_path  = os.path.join(out_dir, "neighborhood_cliques.json")
    with open(cliques_path, "w") as f:
        json.dump(clique_results, f)
    logger.info("Stage 1 saved: %s", cliques_path)

    logger.info("STAGE 2: Cluster cliques → extract programs (resolution=%.2f)",
                CLUSTER_START_RESOLUTION)
    stage2_programs = _cluster_and_extract_programs(
        clique_results, adj_df,
        resolution=CLUSTER_START_RESOLUTION,
        min_clique_size=MIN_CLIQUE_SIZE,
    )
    if not stage2_programs:
        logger.warning("No programs found in Stage 2. Aborting.")
        return
    programs2_path = os.path.join(out_dir, "programs.json")
    with open(programs2_path, "w") as f:
        json.dump(stage2_programs, f, indent=2)
    logger.info("Stage 2 saved: %s (%d programs)", programs2_path, len(stage2_programs))

    logger.info("STAGE 3: Grow → cluster → split → merge")
    final_programs, _ = _grow_and_merge(
        stage2_programs, adj_df,
        growth_threshold=GROWTH_THRESHOLD,
        containment_merge_threshold=CONTAINMENT_MERGE_THRESHOLD,
    )

    pre_filter_count = len(final_programs)
    final_programs = [(g, p) for (g, p) in final_programs if len(g) >= MIN_PROGRAM_SIZE]
    if pre_filter_count != len(final_programs):
        logger.info("Min program size filter: %d → %d programs (removed %d with < %d genes)",
                    pre_filter_count, len(final_programs),
                    pre_filter_count - len(final_programs), MIN_PROGRAM_SIZE)

    adj = adj_df.values
    gene_to_idx = {g: i for i, g in enumerate(adj_df.index)}
    final_genes = [g for g, _ in final_programs]
    metrics = _compute_evaluation_metrics(final_genes, adj, gene_to_idx, len(adj_df.index))

    output = {}
    for i, (prog, peri) in enumerate(final_programs):
        prog_sorted = sorted(prog)
        peri_out    = sorted(set(peri) - set(prog_sorted))
        output[str(i)] = {"program": prog_sorted, "periphery": peri_out}
    final_path = os.path.join(out_dir, PROGRAM_PATH_NAME)
    with open(final_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Final programs saved: %s (%d programs)", final_path, len(final_programs))

    plot_path = os.path.join(
        working_dir, "supp_material", "program_finding_plots",
        "maximal_cliques_in_programs_final.png",
    )
    try:
        _visualize_maximal_cliques_in_programs(
            [sorted(g) for g, _ in final_programs], adj_df, plot_path
        )
    except Exception as e:
        logger.warning("Failed to render maximal-cliques heatmap: %s", e)

    metrics_out = {
        "parameters": {
            "min_clique_size":              MIN_CLIQUE_SIZE,
            "stage2_resolution":            CLUSTER_START_RESOLUTION,
            "growth_threshold":             GROWTH_THRESHOLD,
            "containment_merge_threshold":  CONTAINMENT_MERGE_THRESHOLD,
            "growth_density_floor":         _GROWTH_DENSITY_FLOOR,
            "louvain_resolution_stage3":    _LOUVAIN_RESOLUTION_S3,
            "cluster_low_threshold":        _CLUSTER_LOW_THRESHOLD,
            "split_threshold":              _SPLIT_THRESHOLD,
            "merge_threshold":              _MERGE_THRESHOLD,
        },
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved.")

    logger.info(f"Program finding complete: {len(final_programs)} programs")
    logger.info(f"  Overlap ratio:      {metrics['overlap_ratio']:.3f}  (target < 1.5)")
    logger.info(f"  Max Jaccard:        {metrics['max_pairwise_jaccard']:.4f} (target < 0.5)")
    logger.info(f"  Median density:     {metrics['median_internal_density']:.4f} (target > 0.6)")
    logger.info(f"  Gene coverage:      {metrics['gene_coverage']:.4f}")
    logger.info(f"  Program sizes:      {metrics['program_sizes']}")
