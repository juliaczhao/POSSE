"""Find deterministic positive quasi-clique programs in the signed gene graph."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import hickle
import numpy as np
import pandas as pd
from scipy import sparse


logger = logging.getLogger(__name__)
_REVIEW_ROUNDS = 5


@dataclass(frozen=True)
class Config:
    gamma: float = 0.4
    min_clique_size: int = 5
    seed: int = 0

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or self.min_clique_size < 2:
            raise ValueError("Require 0 < gamma <= 1 and min_clique_size >= 2")


@dataclass(frozen=True)
class Program:
    quasi_clique: tuple[int, ...]
    core: tuple[int, ...]
    periphery: tuple[int, ...] = ()

    @property
    def genes(self):
        return tuple(sorted(self.quasi_clique + self.periphery))


def covered_genes(programs):
    return {gene for program in programs for gene in program.genes}


def load_graph(directory):
    directory = Path(directory)
    matrix_path = directory / "adjacency_matrices/significant_symmetric_matrix.hkl"
    matrix = hickle.load(matrix_path)
    genes = np.load(directory / "gene_names/gene_names.npy", allow_pickle=False)
    all_genes = genes
    if isinstance(matrix, pd.DataFrame):
        if not matrix.index.equals(matrix.columns) or not matrix.index.is_unique:
            raise ValueError("Matrix row and column labels must match and be unique")
        lookup = {gene: i for i, gene in enumerate(genes)}
        genes = genes[[lookup[gene] for gene in matrix.index]]
        matrix = matrix.to_numpy()
    a = sparse.csr_matrix(matrix)
    a.eliminate_zeros()
    if a.shape != (len(genes), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError("Gene names must identify each matrix row uniquely")
    if np.any(a.data != 1) or np.any(a.diagonal()) or (a != a.T).nnz:
        raise ValueError("Adjacency must be binary, symmetric, and have a zero diagonal")
    correlation = hickle.load(matrix_path.resolve().parent.parent / "ids_cooccur/correlation_matrix.hkl")
    return sign_adjacency(a, genes, correlation, all_genes), genes


def sign_adjacency(a, genes, correlation, correlation_genes=None):
    if isinstance(correlation, pd.DataFrame):
        rows, columns = correlation.index, correlation.columns
        values = correlation.to_numpy()
    else:
        rows = columns = pd.Index(correlation_genes)
        values = np.asarray(correlation)
    if not rows.is_unique or not columns.is_unique or values.shape != (len(rows), len(columns)):
        raise ValueError("Correlation axes must have unique gene labels matching their matrix dimensions")
    row_index, column_index = rows.get_indexer(genes), columns.get_indexer(genes)
    if np.any(row_index < 0) or np.any(column_index < 0):
        missing = np.asarray(genes)[(row_index < 0) | (column_index < 0)]
        raise ValueError(f"Adjacency genes missing from correlation matrix: {missing.tolist()}")
    edges = sparse.coo_matrix(a)
    correlations = values[row_index[edges.row], column_index[edges.col]]
    if not np.isfinite(correlations).all():
        raise ValueError("Correlations for significant edges must be finite")
    signed = sparse.csr_matrix((np.sign(correlations).astype(np.int8), (edges.row, edges.col)), shape=a.shape)
    signed.eliminate_zeros()
    if (signed != signed.T).nnz:
        raise ValueError("Aligned correlation signs must be symmetric on significant edges")
    return signed


def graph_summary(a):
    return {"genes": a.shape[0], "positive_edges": int(np.count_nonzero(a.data > 0) // 2),
            "negative_edges": int(np.count_nonzero(a.data < 0) // 2),
            "negative_edges_allowed_within_programs": 0}


def track(items, label, progress):
    start = last = perf_counter()
    if progress:
        print(f"{label}: 0/{len(items)}", flush=True)
    for i, item in enumerate(items, 1):
        yield item
        now = perf_counter()
        if progress and (now - last >= 2 or i == len(items)):
            elapsed = now - start
            print(f"{label}: {i}/{len(items)} | {elapsed:.1f}s | "
                  f"ETA {elapsed * (len(items) - i) / i:.1f}s", flush=True)
            last = now


def maximum_clique(g):
    """Exact clique search with bitsets and a greedy coloring bound."""
    g = np.asarray(g) > 0
    order = np.lexsort((np.arange(len(g)), -g.sum(1)))
    b = g[np.ix_(order, order)]
    neighbors = [int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little") for row in b]
    best, pool = [], (1 << len(b)) - 1
    while pool:
        v = max((i for i in range(len(b)) if pool & (1 << i)),
                key=lambda i: (neighbors[i] & pool).bit_count())
        best.append(v)
        pool &= neighbors[v]

    def color_sort(pool):
        vertices, bounds, color = [], [], 0
        while pool:
            color += 1
            available = pool
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                vertices.append(v)
                bounds.append(color)
                pool ^= bit
                available &= ~bit & ~neighbors[v]
        return vertices, bounds

    def search(clique, pool):
        nonlocal best
        vertices, bounds = color_sort(pool)
        for i in range(len(vertices) - 1, -1, -1):
            if len(clique) + bounds[i] <= len(best):
                return
            v = vertices[i]
            extension, remaining = clique + [v], pool & neighbors[v]
            if remaining:
                search(extension, remaining)
            elif len(extension) > len(best):
                best = extension
            pool &= ~(1 << v)

    search([], (1 << len(b)) - 1)
    return tuple(sorted(map(int, order[best])))


# 1. Find maximal clique seeds without removing genes from the graph.
def find_seeds(g, config, rank, progress=False):
    positive = g > 0
    degree = positive.sum(1)
    roots = np.lexsort((rank, -degree))
    roots = roots[degree[roots] >= config.min_clique_size - 1]
    covered = np.zeros(len(g), bool)
    seeds = []
    for root in track(roots, "1. Clique seeds", progress):
        if covered[root]:
            continue
        neighbors = np.flatnonzero(positive[root])
        neighbors = neighbors[np.argsort(rank[neighbors])]
        clique = neighbors[list(maximum_clique(g[np.ix_(neighbors, neighbors)]))]
        seed = tuple(sorted([int(root), *map(int, clique)]))
        if len(seed) >= config.min_clique_size:
            seeds.append(seed)
            covered[list(seed)] = True
    return seeds


def required_degree(size, gamma):
    return int(np.ceil(gamma * (size - 1) - 1e-12))


def is_quasiclique(g, nodes, gamma):
    if not len(nodes):
        return False
    block = g[np.ix_(nodes, nodes)]
    return bool(not (block < 0).any() and (block > 0).sum(1).min() >= required_degree(len(nodes), gamma))


# 2. Grow clique seeds while checking every member's degree.
def grow_quasiclique(g, nodes, gamma, rank, preferred=()):
    nodes = list(nodes)
    selected = np.zeros(len(g), bool)
    selected[nodes] = True
    priority = np.zeros(len(g), bool)
    priority[list(preferred)] = True
    links = (g[:, nodes] > 0).sum(1, dtype=np.int32)
    conflicts = (g[:, nodes] < 0).any(1)
    degree = links[nodes].copy()
    while True:
        required = required_degree(len(nodes) + 1, gamma)
        candidates = np.flatnonzero(~selected & ~conflicts & (links >= required))
        weak = np.asarray(nodes)[degree < required]
        candidates = candidates[(g[np.ix_(candidates, weak)] > 0).all(1)]
        if not len(candidates):
            return tuple(sorted(nodes))
        v = int(candidates[np.lexsort((rank[candidates], -links[candidates],
                                      ~priority[candidates]))[0]])
        degree = np.r_[degree + (g[nodes, v] > 0), links[v]]
        nodes.append(v)
        selected[v] = True
        links += g[:, v] > 0
        conflicts |= g[:, v] < 0


def remove_subsets(family):
    kept, sets = [], []
    for nodes in sorted({tuple(sorted(q)) for q in family}, key=lambda q: (-len(q), q)):
        members = set(nodes)
        if not any(members <= other for other in sets):
            kept.append(nodes)
            sets.append(members)
    return kept


# 3. Cluster proposals: exclusive connections + a cost for splitting shared genes.
def clustering_weights(a, quasi, gamma):
    rows = np.concatenate(quasi)
    cols = np.repeat(np.arange(len(quasi)), [len(q) for q in quasi])
    membership = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(a.shape[0], len(quasi)))
    connections = a @ membership
    overlap = (membership.T @ membership).toarray()
    shared_links = (membership.multiply(connections).T @ membership).toarray()
    def shared_edges_for_proposal(item):
        i, nodes = item
        shared = membership[list(nodes)].toarray()
        return i, (shared * (a[list(nodes)][:, list(nodes)] @ shared)).sum(0)

    shared_edges = np.zeros_like(overlap)
    for i, values in map(shared_edges_for_proposal, enumerate(quasi)):
        shared_edges[i] = values
    exclusive_edges = (membership.T @ connections).toarray() - shared_links - shared_links.T + shared_edges
    sizes = np.array([len(q) for q in quasi])
    exclusive_sizes = sizes[:, None] - overlap
    pairs = exclusive_sizes * exclusive_sizes.T
    density = np.divide(exclusive_edges, pairs, out=np.zeros_like(exclusive_edges), where=pairs > 0)
    smaller = np.minimum.outer(sizes, sizes)
    weights = overlap + (smaller - overlap) * density - gamma * smaller
    np.fill_diagonal(weights, 0)
    return weights


def clustering_loss(weights, labels):
    same = labels[:, None] == labels[None, :]
    costs = np.where(same, np.maximum(-weights, 0), np.maximum(weights, 0))
    return float(np.triu(costs, 1).sum())


def cluster_quasicliques(weights, progress=False):
    """Greedily take the best improving cluster merge or proposal relocation."""
    labels = np.arange(len(weights))
    before = clustering_loss(weights, labels)
    improvement, moves, merges = 0.0, 0, 0
    start = last = perf_counter()
    if progress:
        print(f"3. Clustering: {len(weights)} proposals | initial loss {before:.2f}", flush=True)
    n = len(weights)
    affinity = np.empty_like(weights)
    affinity[:] = weights
    cluster_weights = np.empty_like(weights)
    cluster_weights[:] = weights
    k = n
    while len(weights) > 1:
        merge_gain = np.triu(cluster_weights[:k, :k], 1)
        merge_gain[np.tril_indices(k)] = -np.inf
        current = affinity[np.arange(n), labels]
        existing_gain = affinity[:, :k] - current[:, None]
        merge = np.unravel_index(np.argmax(merge_gain), merge_gain.shape)
        existing_move = np.unravel_index(
            np.argmax(existing_gain), existing_gain.shape
        )
        singleton_row = int(np.argmax(-current))
        if -current[singleton_row] > existing_gain[existing_move] or (
            -current[singleton_row] == existing_gain[existing_move]
            and (singleton_row, k) < existing_move
        ):
            move = (singleton_row, k)
            move_value = -current[singleton_row]
        else:
            move = existing_move
            move_value = existing_gain[existing_move]
        gain = max(merge_gain[merge], move_value)
        if gain <= 1e-9:
            break
        if merge_gain[merge] >= move_value:
            left, right = merge
            labels[labels == right] = left
            labels[labels > right] -= 1
            affinity[:, left] += affinity[:, right]
            cluster_weights[left, :k] += cluster_weights[right, :k]
            cluster_weights[:k, left] += cluster_weights[:k, right]
            if right < k - 1:
                affinity[:, right : k - 1] = affinity[:, right + 1 : k]
                cluster_weights[right : k - 1, :k] = cluster_weights[right + 1 : k, :k]
                cluster_weights[:k, right : k - 1] = cluster_weights[:k, right + 1 : k]
            k -= 1
            merges += 1
        else:
            proposal, target = move
            source = labels[proposal]
            proposal_affinity = affinity[proposal, :k].copy()
            cluster_weights[source, :k] -= proposal_affinity
            cluster_weights[:k, source] -= proposal_affinity
            affinity[:, source] -= weights[:, proposal]
            if target == k:
                cluster_weights[k, :k] = proposal_affinity
                cluster_weights[:k, k] = proposal_affinity
                cluster_weights[k, k] = 0
                affinity[:, k] = weights[:, proposal]
                k += 1
            else:
                cluster_weights[target, :k] += proposal_affinity
                cluster_weights[:k, target] += proposal_affinity
                affinity[:, target] += weights[:, proposal]
            labels[proposal] = target
            if not np.any(labels == source):
                labels[labels > source] -= 1
                if source < k - 1:
                    affinity[:, source : k - 1] = affinity[:, source + 1 : k]
                    cluster_weights[source : k - 1, :k] = cluster_weights[source + 1 : k, :k]
                    cluster_weights[:k, source : k - 1] = cluster_weights[:k, source + 1 : k]
                k -= 1
            moves += 1
        improvement += gain
        now = perf_counter()
        if progress and now - last >= 2:
            print(f"3. Clustering: {labels.max() + 1} clusters | loss {before - improvement:.2f} | "
                  f"{merges} merges, {moves} moves | {now - start:.1f}s", flush=True)
            last = now
    return labels, {"loss_before": before, "loss_after": clustering_loss(weights, labels),
                    "merges": merges, "relocations": moves,
                    "stopping_rule": "No improving cluster merge or single-proposal relocation"}


# 4. Repair unions while protecting a clique anchor or required program genes.
def reinforce_quasiclique(g, nodes, gamma, rank, preferred=(), required=()):
    nodes = np.array(nodes, int)
    anchor = (required if required else
              nodes[list(maximum_clique(g[np.ix_(nodes, nodes)]))])
    protected = np.isin(nodes, anchor)
    priority = np.isin(nodes, preferred)
    block = g[np.ix_(nodes, nodes)]
    degree = (block > 0).sum(1, dtype=np.int32)
    conflicts = (block < 0).sum(1, dtype=np.int32)
    if required and ((block[np.ix_(protected, protected)] < 0).any()
                     or degree[protected].min() < required_degree(int(protected.sum()), gamma)):
        return None
    while conflicts.any() or degree.min() < required_degree(len(nodes), gamma):
        candidates = np.flatnonzero(~protected)
        if conflicts.any():
            candidates = candidates[conflicts[candidates] > 0]
        if not len(candidates):
            return None
        weak = protected & (degree < required_degree(len(nodes), gamma))
        support = ((g[np.ix_(nodes[candidates], nodes[weak])] > 0).sum(1)
                   if required else np.zeros(len(candidates)))
        drop = candidates[np.lexsort((rank[nodes[candidates]], degree[candidates],
                                      support, -conflicts[candidates], priority[candidates]))[0]]
        degree -= g[nodes, nodes[drop]] > 0
        conflicts -= g[nodes, nodes[drop]] < 0
        nodes = np.delete(nodes, drop)
        degree = np.delete(degree, drop)
        conflicts = np.delete(conflicts, drop)
        protected = np.delete(protected, drop)
        priority = np.delete(priority, drop)
    return tuple(map(int, nodes))


# 5. Attach a compatible periphery from the entire graph, then recompute the core.
def attach_periphery(g, quasi, gamma, rank, preferred=()):
    full = grow_quasiclique(g, quasi, gamma, rank, preferred)
    core = tuple(quasi[v] for v in maximum_clique(g[np.ix_(quasi, quasi)]))
    return Program(quasi, core, tuple(sorted(set(full) - set(quasi))))


def family_summary(family):
    sets = [set(nodes) for nodes in family]
    if not sets:
        return {"sets": 0, "unique_genes": 0, "gene_memberships": 0,
                "duplicate_memberships": 0, "strict_subset_pairs": 0,
                "near_subset_pairs": 0}
    union = set().union(*sets)
    memberships = sum(map(len, sets))
    if len(sets) < 2:
        strict_subset_pairs = near_subset_pairs = 0
    else:
        rows = np.fromiter((gene for nodes in sets for gene in nodes), int)
        columns = np.repeat(np.arange(len(sets)), [len(nodes) for nodes in sets])
        incidence = sparse.csc_matrix(
            (np.ones(len(rows), np.int32), (rows, columns)),
            shape=(max(union) + 1, len(sets)),
        )
        overlap = (incidence.T @ incidence).toarray()
        sizes = np.asarray([len(nodes) for nodes in sets])
        smaller = np.minimum.outer(sizes, sizes)
        upper = np.triu(np.ones_like(overlap, dtype=bool), 1)
        distinct = overlap < np.maximum.outer(sizes, sizes)
        strict_subset_pairs = int(
            np.count_nonzero(upper & distinct & (overlap == smaller))
        )
        near_subset_pairs = int(
            np.count_nonzero(upper & distinct & (overlap >= 0.9 * smaller))
        )
    return {"sets": len(sets), "unique_genes": len(union), "gene_memberships": memberships,
            "duplicate_memberships": memberships - len(union),
            "strict_subset_pairs": strict_subset_pairs,
            "near_subset_pairs": near_subset_pairs}


def remove_program_subsets(programs):
    by_genes = {}
    for p in programs:
        by_genes.setdefault(p.genes, p)
    return [by_genes[nodes] for nodes in remove_subsets(by_genes)]


# 6. Recover missing genes from proposals and positive clique seeds.
def recover_seed(g, seed, proposal, missing, gamma, rank):
    """Grow missing members around a shared clique anchor before adding periphery."""
    pool = np.array(sorted(set(seed) | (set(proposal) & missing)), int)
    local = grow_quasiclique(g[np.ix_(pool, pool)], np.searchsorted(pool, seed),
                            gamma, rank[pool])
    quasi = tuple(map(int, pool[list(local)]))
    return attach_periphery(g, quasi, gamma, rank, sorted(missing))


def recover_programs(g, programs, proposals, gamma, rank, min_clique_size=5, progress=False,
                     seeds=()):
    start = last = perf_counter()
    working = list(programs)
    represented = covered_genes(working)
    previous = represented.copy()
    candidates = list(map(set, proposals))
    seed_sets = list(map(set, seeds))
    missing = set().union(*candidates) - represented
    events = []
    pending = set(range(len(candidates)))
    while missing:
        coverage = len(represented)
        for i in track(range(len(working)), "6. Periphery recovery", progress):
            if not missing:
                break
            original = working[i]
            candidate = attach_periphery(g, original.quasi_clique, gamma, rank, sorted(missing))
            added = set(candidate.genes) - represented
            others = covered_genes(p for j, p in enumerate(working) if j != i)
            lost = set(original.genes) - set(candidate.genes) - others
            if added and not lost:
                working[i] = candidate
                represented.update(candidate.genes)
                missing -= represented
                events.append({"kind": "periphery", "input_program_id": i,
                               "added_genes": sorted(added),
                               "replaced_periphery_genes": sorted(set(original.genes) - set(candidate.genes))})
        while missing:
            eligible = {}
            for i in sorted(pending):
                novel = np.array(sorted(candidates[i] & missing), int)
                core = novel[list(maximum_clique(g[np.ix_(novel, novel)]))] if len(novel) >= min_clique_size else ()
                if len(core) >= min_clique_size:
                    eligible[i] = core.tolist()
                else:
                    # Missing sets only shrink; a rejected proposal cannot become eligible later.
                    pending.remove(i)
            if eligible:
                index = max(eligible, key=lambda i: (len(candidates[i] & missing), -len(candidates[i]), -i))
                candidate = attach_periphery(g, tuple(sorted(candidates[index])), gamma, rank, sorted(missing))
                event = {"kind": "proposal", "proposal_id": index, "novel_clique": eligible[index]}
            else:
                eligible = [i for i, nodes in enumerate(seed_sets)
                            if len(nodes) >= min_clique_size and 2 * len(nodes & missing) > len(nodes)]
                if not eligible:
                    break
                index = max(eligible, key=lambda i: (len(seed_sets[i] & missing), -len(seed_sets[i]), -i))
                source = min((i for i, nodes in enumerate(candidates) if seed_sets[index] <= nodes),
                             key=lambda i: (len(candidates[i]), i))
                candidate = recover_seed(g, seeds[index], proposals[source], missing, gamma, rank)
                event = {"kind": "seed", "seed_id": index, "proposal_id": source,
                         "seed_clique": list(seeds[index]),
                         "reused_anchor_genes": sorted(seed_sets[index] - missing)}
            added = set(candidate.genes) - represented
            working.append(candidate)
            represented.update(candidate.genes)
            missing -= represented
            events.append({**event, "added_genes": sorted(added), "program_size": len(candidate.genes)})
            now = perf_counter()
            if progress and (now - last >= 2 or not missing):
                print(f"6. Context recovery: {len(working)} programs | {len(missing)} proposal genes left | "
                      f"{now - start:.1f}s", flush=True)
                last = now
        if len(represented) == coverage:
            break
    result = remove_program_subsets(working)
    report = {"events": events, "added_genes": sorted(represented - previous),
              "unresolved_genes": sorted(missing), "min_clique_size": min_clique_size,
              "restored_programs": sum(e["kind"] in ("proposal", "seed") for e in events),
              "seed_restorations": sum(e["kind"] == "seed" for e in events),
              "periphery_changes": sum(e["kind"] == "periphery" for e in events),
              "seconds": perf_counter() - start,
              "stopping_rule": "No coverage-increasing periphery growth, missing clique, or source seed with a missing majority"}
    if progress:
        print(f"6. Recovery: {len(report['added_genes'])} genes | {report['restored_programs']} contexts | "
              f"{report['seed_restorations']} from seeds | {report['periphery_changes']} periphery changes | "
              f"{report['seconds']:.2f}s", flush=True)
    return result, report


# 7. Review complete programs with fresh weights and coverage-preserving merges.
def review_programs(adjacency, programs, config=Config(), preserve_coverage=True, progress=True,
                    recovery_proposals=(), recovery_seeds=()):
    start = last = perf_counter()
    a = sparse.csr_matrix(adjacency, dtype=np.int32)
    g = a.toarray().astype(np.int8)
    if any(not is_quasiclique(g, p.quasi_clique, config.gamma)
           or not is_quasiclique(g, p.genes, config.gamma) for p in programs):
        raise ValueError("Review inputs must satisfy gamma with no negative edges; rerun extraction on the signed graph")
    if any(not is_quasiclique(g, q, config.gamma) for q in recovery_proposals):
        raise ValueError("Recovery proposals must satisfy gamma with no negative edges; rerun extraction on the signed graph")
    if any((g[np.ix_(seed, seed)] > 0).sum() != len(seed) * (len(seed) - 1) for seed in recovery_seeds):
        raise ValueError("Recovery seeds must be positive cliques; rerun extraction on the signed graph")
    rank = np.random.default_rng(config.seed).permutation(len(g))
    original_programs = programs
    programs, recovery = recover_programs(g, programs, recovery_proposals, config.gamma, rank,
                                          config.min_clique_size, progress, recovery_seeds)
    family = [p.genes for p in programs]
    weights = clustering_weights(a, family, config.gamma) if family else np.empty((0, 0))
    working = dict(enumerate(programs))
    sources = {i: [i] for i in working}
    blocked, events = {}, []
    if progress:
        print(f"Review: {len(programs)} programs | preserve coverage: {preserve_coverage}", flush=True)
    while len(working) > 1:
        ids = list(working)
        membership = np.zeros((len(family), len(ids)))
        for j, index in enumerate(ids):
            membership[sources[index], j] = 1
        gains = membership.T @ weights @ membership
        gains[np.tril_indices(len(ids))] = -np.inf
        for left, right in blocked:
            gains[ids.index(left), ids.index(right)] = -np.inf
        pair = np.unravel_index(np.argmax(gains), gains.shape)
        if gains[pair] <= 1e-9:
            break
        left, right = ids[pair[0]], ids[pair[1]]
        source = sorted(sources[left] + sources[right])
        union = tuple(sorted(set(working[left].genes) | set(working[right].genes)))
        negative_edges = int((g[np.ix_(union, union)] < 0).sum() // 2)
        others = covered_genes(p for i, p in working.items() if i not in (left, right))
        needed = sorted(set(union) - others) if preserve_coverage else ()
        quasi = reinforce_quasiclique(g, union, config.gamma, rank, needed)
        preferred = sorted(set(union) - set(quasi) - others) if preserve_coverage else ()
        candidate = attach_periphery(g, quasi, config.gamma, rank, preferred)
        lost = (set(working[left].genes) | set(working[right].genes)) - set(candidate.genes) - others
        repair_method = "clique_anchor"
        if preserve_coverage and lost:
            repaired = reinforce_quasiclique(g, union, config.gamma, rank, required=needed)
            if repaired is not None:
                alternative = attach_periphery(g, repaired, config.gamma, rank)
                if len(alternative.core) >= config.min_clique_size:
                    candidate = alternative
                    lost = set(union) - set(candidate.genes) - others
                    repair_method = "required_genes"
        if preserve_coverage and lost:
            blocked[left, right] = {"reason": "coverage", "lost_genes": sorted(lost)}
        else:
            events.append({"source_program_ids": source, "loss_reduction": float(gains[pair]),
                           "repair_method": repair_method,
                           "union_size": len(union), "retained_size": len(candidate.genes),
                           "negative_edges_in_union": negative_edges,
                           "pruned_genes": sorted(set(union) - set(candidate.genes)),
                           "lost_genes": sorted(lost)})
            working[left], sources[left] = candidate, source
            del working[right], sources[right]
            blocked.clear()
        now = perf_counter()
        if progress and now - last >= 2:
            print(f"Review: {len(working)} programs | {len(events)} merges | "
                  f"{len(blocked)} blocked | {now - start:.1f}s", flush=True)
            last = now
    labels = np.empty(len(family), int)
    unions, reinforced = [], []
    for label, index in enumerate(working):
        labels[sources[index]] = label
        unions.append(tuple(sorted(set().union(*(set(family[i]) for i in sources[index])))))
        reinforced.append(working[index].quasi_clique)
    result = remove_program_subsets(working.values())
    retained, previous = covered_genes(result), covered_genes(original_programs)
    report = {"config": asdict(config), "graph": graph_summary(a),
              "seconds": {"review": perf_counter() - start},
              "recovery": recovery, "recovery_proposals": recovery_proposals, "recovery_seeds": recovery_seeds,
              "proposals": family, "proposal_labels": labels.tolist(), "cluster_unions": unions,
              "reinforced_quasicliques": reinforced,
              "program_sources": [next(sources[i] for i, p in working.items() if p.genes == q.genes)
                                  for q in result],
              "optimization": {"loss_before": clustering_loss(weights, np.arange(len(family))),
                               "loss_after": clustering_loss(weights, labels),
                               "merges": len(events), "relocations": 0,
                               "stopping_rule": "No improving admissible cluster merge after reinforcement"},
              "preserve_coverage": preserve_coverage, "merge_events": events,
              "blocked_merges": [{"source_program_ids": sorted(sources[left] + sources[right]),
                                  **reason} for (left, right), reason in blocked.items()],
              "summary": {"input_programs": family_summary([p.genes for p in original_programs]),
                          "after_recovery": family_summary(family),
                          "programs": family_summary([p.genes for p in result])},
              "removed_genes": sorted(previous - retained), "added_genes": sorted(retained - previous),
              "unchanged": {p.genes for p in original_programs} == {p.genes for p in result}}
    if progress:
        print(f"Review: {len(original_programs)} -> {len(result)} programs | {len(events)} merges | "
              f"{len(report['removed_genes'])} genes lost | {report['seconds']['review']:.2f}s", flush=True)
    return result, report


def find_programs(adjacency, config=Config(), progress=True):
    a = sparse.csr_matrix(adjacency, dtype=np.int32)
    g = a.toarray().astype(np.int8)
    rank = np.random.default_rng(config.seed).permutation(len(g))
    times, start = {}, perf_counter()
    seeds = find_seeds(g, config, rank, progress)
    times["seeds"] = perf_counter() - start
    start = perf_counter()
    seed_items = list(track(seeds, "2. Quasi-clique growth", progress))
    grown = [grow_quasiclique(g, seed, config.gamma, rank) for seed in seed_items]
    quasi = remove_subsets(grown)
    times["growth"] = perf_counter() - start
    start = perf_counter()
    weights = clustering_weights(a, quasi, config.gamma) if quasi else np.empty((0, 0))
    labels, optimization = cluster_quasicliques(weights, progress)
    times["clustering"] = perf_counter() - start
    start = perf_counter()
    clusters = []
    for label in np.unique(labels):
        members = [quasi[i] for i in np.flatnonzero(labels == label)]
        clusters.append(tuple(map(int, np.unique(np.concatenate(members)))))
    reinforced = [reinforce_quasiclique(g, q, config.gamma, rank)
                  for q in track(clusters, "4. Quasi-clique reinforcement", progress)]
    final_quasi = remove_subsets(reinforced)
    times["reinforcement"] = perf_counter() - start
    start = perf_counter()
    programs = [attach_periphery(g, q, config.gamma, rank)
                for q in track(final_quasi, "5. Periphery and cores", progress)]
    programs = remove_program_subsets(programs)
    times["periphery_and_cores"] = perf_counter() - start
    proposed_genes = set().union(*map(set, quasi))
    retained_genes = covered_genes(programs)
    report = {"config": asdict(config), "graph": graph_summary(a), "seconds": times, "optimization": optimization,
              "seeds": seeds, "proposals": quasi, "proposal_labels": labels.tolist(),
              "cluster_unions": clusters, "reinforced_quasicliques": reinforced,
              "summary": {"seeds": family_summary(seeds), "grown": family_summary(grown),
                          "proposals": family_summary(quasi),
                          "quasicliques": family_summary([p.quasi_clique for p in programs]),
                          "programs": family_summary([p.genes for p in programs])},
              "removed_genes": sorted(proposed_genes - retained_genes)}
    if progress:
        print(f"Saved membership: {len(programs)} quasi-cliques | "
              f"{report['summary']['programs']['unique_genes']} genes | "
              f"{len(report['removed_genes'])} proposal genes not retained | "
              f"loss {optimization['loss_before']:.2f} -> {optimization['loss_after']:.2f}", flush=True)
    return programs, report



def _write_programs(path, genes, programs):
    records = [
        {"program_id": program_id, "genes": genes[list(program.genes)].tolist()}
        for program_id, program in enumerate(programs)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(records, indent=2) + "\n")
    temporary_path.replace(path)
    return records


def _remove_legacy_outputs(output_dir):
    for name in (
        "metrics.json",
        "neighborhood_cliques.json",
        "programs_after_grow_merge.json",
    ):
        (output_dir / name).unlink(missing_ok=True)


def run_program_finder(working_dir, gamma=0.4, min_clique_size=5, seed=0):
    """Extract quasi-clique programs and write one final programs.json file."""
    config = Config(gamma=gamma, min_clique_size=min_clique_size, seed=seed)
    adjacency, genes = load_graph(working_dir)
    summary = graph_summary(adjacency)
    logger.info(
        "Signed graph: %d genes, %d positive edges, %d negative edges",
        summary["genes"],
        summary["positive_edges"],
        summary["negative_edges"],
    )
    logger.info("Program finding uses one CPU worker")

    programs, extraction = find_programs(adjacency, config, progress=False)
    proposals = extraction["proposals"]
    seeds = extraction["seeds"]
    logger.info(
        "Extraction: %d seeds, %d proposals, %d programs in %.2f seconds (%s)",
        len(seeds),
        len(proposals),
        len(programs),
        sum(extraction["seconds"].values()),
        ", ".join(
            f"{name}={seconds:.2f}s"
            for name, seconds in extraction["seconds"].items()
        ),
    )
    rounds_completed = 0
    for round_number in range(1, _REVIEW_ROUNDS + 1):
        programs, review = review_programs(
            adjacency,
            programs,
            config,
            preserve_coverage=True,
            progress=False,
            recovery_proposals=proposals,
            recovery_seeds=seeds,
        )
        rounds_completed = round_number
        logger.info(
            "Review round %d: %d programs, %d merges, %d recovered genes in %.2f seconds",
            round_number,
            len(programs),
            review["optimization"]["merges"],
            len(review["recovery"]["added_genes"]),
            review["seconds"]["review"],
        )
        if review["unchanged"]:
            break

    output_dir = Path(working_dir) / "clique_based_programs"
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_outputs(output_dir)
    output_path = output_dir / "programs.json"
    records = _write_programs(output_path, genes, programs)
    logger.info(
        "Program finding complete: %d programs, %d genes, %d review rounds; wrote %s",
        len(records),
        len(covered_genes(programs)),
        rounds_completed,
        output_path,
    )
    return records
