"""Generate figures for the interactive program report.

Generates the UMAP, Leiden clustering and per-program activity plots used by the
program report.

Reads
-----
    <working_dir>/program_activity/wePAS.h5ad
    <working_dir>/program_activity/sePAS.h5ad

Writes
------
    <working_dir>/report_plots/global_plots/
    <working_dir>/report_plots/program_plots/
    <working_dir>/report_plots/cluster_plots/
"""

import base64
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from time import perf_counter


_PLOT_WORKER_FLAG = "--render-plot-manifest"
_IS_PLOT_WORKER = (
    __name__ == "__main__"
    and len(sys.argv) == 3
    and sys.argv[1] == _PLOT_WORKER_FLAG
)

if _IS_PLOT_WORKER:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if not _IS_PLOT_WORKER:
    import anndata as ad
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as PathEffects
    import rapids_singlecell as rsc
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    from tqdm import tqdm

logger = logging.getLogger(__name__)


def _available_cpu_count():
    """Return the CPUs available to this process, respecting scheduler affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _plot_worker_count(task_count):
    requested = os.environ.get("POSSE_PLOT_WORKERS")
    limit = int(requested) if requested else 64
    return min(task_count, _available_cpu_count(), max(1, limit))


def _activity_colormap(only_nonnegative):
    if only_nonnegative:
        name = "gray_red"
        colors = ["#E0E0E0", "#FF9999", "#CC0000"]
    else:
        name = "blue_gray_red"
        colors = ["#2166ac", "#67a9cf", "#E0E0E0", "#FF6666", "#CC0000"]
    return mcolors.LinearSegmentedColormap.from_list(name, colors, N=256)


def _render_sepas_umap(
    sepas_activity,
    embedding,
    output_dir,
    program_index,
    point_size,
    alpha,
):
    activity = sepas_activity[:, program_index]
    activity_min = np.nanmin(activity)
    activity_max = np.nanmax(activity)
    cmap = _activity_colormap(only_nonnegative=activity_min >= 0)
    if activity_min == activity_max:
        activity_min -= 1e-6
        activity_max += 1e-6

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=activity,
        cmap=cmap,
        vmin=activity_min,
        vmax=activity_max,
        s=point_size,
        alpha=alpha,
        edgecolor="none",
        rasterized=True,
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"Program {program_index} sePAS activity")
    plt.colorbar(scatter, label="sePAS activity")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"umap_sepas_activity_{program_index}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def _render_cluster_violin(
    program_activity,
    cluster_labels,
    cluster_palette,
    output_dir,
    cluster_index,
):
    cluster_values = program_activity[cluster_labels == cluster_index]
    n_cells, n_programs = cluster_values.shape
    frame = pd.DataFrame(
        {
            "Program": np.repeat(np.arange(n_programs), n_cells),
            "Activity": cluster_values.T.reshape(-1),
        }
    )
    unique_clusters = sorted(int(value) for value in np.unique(cluster_labels))
    palette_index = unique_clusters.index(cluster_index)
    color = (
        cluster_palette[palette_index]
        if palette_index < len(cluster_palette)
        else "gray"
    )

    plt.figure(figsize=(2.2 * n_programs, 3))
    sns.violinplot(
        x="Program",
        y="Activity",
        data=frame,
        color=color,
        inner="quartile",
        linewidth=1,
        cut=0,
    )
    plt.xlabel("Program")
    plt.ylabel("Activity")
    plt.title(f"Program Activity Distributions for Cluster {cluster_index}")
    plt.xticks(
        ticks=np.arange(n_programs),
        labels=[str(index) for index in range(n_programs)],
        rotation=45,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f"program_activity_violins_cluster_{cluster_index}.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def _render_plot_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    program_activity = np.load(manifest["activity_path"], mmap_mode="r")
    sepas_activity = np.load(manifest["sepas_activity_path"], mmap_mode="r")
    embedding = np.load(manifest["embedding_path"], mmap_mode="r")
    cluster_labels = np.load(manifest["cluster_labels_path"], mmap_mode="r")
    cluster_palette = np.load(manifest["cluster_palette_path"])

    for task_type, index in manifest["tasks"]:
        if task_type == "sepas_umap":
            _render_sepas_umap(
                sepas_activity,
                embedding,
                manifest["sepas_umap_dir"],
                index,
                manifest["point_size"],
                manifest["alpha"],
            )
        elif task_type == "cluster_violin":
            _render_cluster_violin(
                program_activity,
                cluster_labels,
                cluster_palette,
                manifest["cluster_violin_dir"],
                index,
            )
        else:
            raise ValueError(f"Unknown plot task: {task_type}")


def _render_plot_batches(
    X,
    sepas_activity,
    embedding,
    cluster_labels,
    cluster_palette,
    report_plot_dir,
    program_specific_dir,
    cluster_specific_dir,
    point_size,
    alpha,
):
    """Render independent per-program and per-cluster figures in CPU batches."""
    sepas_umap_dir = os.path.join(program_specific_dir, "sepas_activity_umaps")
    cluster_violin_dir = os.path.join(cluster_specific_dir, "cluster_activities")
    for directory in (sepas_umap_dir, cluster_violin_dir):
        os.makedirs(directory, exist_ok=True)

    n_programs = X.shape[1]
    if sepas_activity.shape != X.shape:
        raise ValueError(
            f"sePAS activity has shape {sepas_activity.shape}; expected {X.shape}"
        )
    unique_clusters = sorted(int(value) for value in np.unique(cluster_labels))
    tasks = (
        [("sepas_umap", index) for index in range(n_programs)]
        + [("cluster_violin", index) for index in unique_clusters]
    )
    worker_count = _plot_worker_count(len(tasks))
    logger.info(
        "Batched plot rendering: %d tasks across %d workers (%d CPUs available)",
        len(tasks),
        worker_count,
        _available_cpu_count(),
    )

    started = perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="posse_plot_batches_", dir=report_plot_dir
    ) as temp_dir:
        activity_path = os.path.join(temp_dir, "activity.npy")
        sepas_activity_path = os.path.join(temp_dir, "sepas_activity.npy")
        embedding_path = os.path.join(temp_dir, "embedding.npy")
        labels_path = os.path.join(temp_dir, "cluster_labels.npy")
        palette_path = os.path.join(temp_dir, "cluster_palette.npy")
        np.save(activity_path, np.asarray(X))
        np.save(sepas_activity_path, np.asarray(sepas_activity))
        np.save(embedding_path, np.asarray(embedding))
        np.save(labels_path, np.asarray(cluster_labels))
        np.save(palette_path, np.asarray(cluster_palette))

        batches = [[] for _ in range(worker_count)]
        for task_index, task in enumerate(tasks):
            batches[task_index % worker_count].append(task)

        processes = []
        log_handles = []
        for worker_index, batch in enumerate(batches):
            manifest = {
                "activity_path": activity_path,
                "sepas_activity_path": sepas_activity_path,
                "embedding_path": embedding_path,
                "cluster_labels_path": labels_path,
                "cluster_palette_path": palette_path,
                "sepas_umap_dir": sepas_umap_dir,
                "cluster_violin_dir": cluster_violin_dir,
                "point_size": point_size,
                "alpha": alpha,
                "tasks": batch,
            }
            manifest_path = os.path.join(temp_dir, f"worker_{worker_index}.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            log_path = os.path.join(temp_dir, f"worker_{worker_index}.log")
            log_handle = open(log_path, "w+", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "MPLBACKEND": "Agg",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    _PLOT_WORKER_FLAG,
                    manifest_path,
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append((worker_index, process, log_path))
            log_handles.append(log_handle)

        failures = []
        for worker_index, process, log_path in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_index, return_code, log_path))
        for handle in log_handles:
            handle.close()

        if failures:
            messages = []
            for worker_index, return_code, log_path in failures:
                with open(log_path, "r", encoding="utf-8") as handle:
                    messages.append(
                        f"plot worker {worker_index} exited with {return_code}:\n{handle.read()}"
                    )
            raise RuntimeError("\n\n".join(messages))

    logger.info(
        "Batched plot rendering completed in %.2f seconds", perf_counter() - started
    )


def load_program_activity(working_dir):
    """Load wePAS, the per-cell activity matrix the report figures are built from.

    The figures use the loading-weighted activity, scaled across programs so the values
    are comparable. The co-expression score (sePAS) is a continuous-vs-sparse mismatch
    for a shared embedding, so it is not used here.

    Returns (adata, X, path), where X is the scaled cells-by-programs matrix to plot and
    path is the file the UMAP embedding and cluster labels are written back to.
    """
    path = os.path.join(working_dir, "program_activity", "wePAS.h5ad")
    adata = ad.read_h5ad(path)
    obsm = adata.obsm
    X = (
        obsm["program_activity_scaled"]
        if "program_activity_scaled" in obsm
        else obsm["program_activity_unscaled"]
    )
    return adata, np.asarray(X), path


def load_sepas_activity(working_dir):
    """Load continuous cells-by-programs sePAS activity."""
    path = os.path.join(working_dir, "program_activity", "sePAS.h5ad")
    adata = ad.read_h5ad(path)
    activity = adata.obsm.get("synchronized_program_activity_unscaled", adata.X)
    if hasattr(activity, "toarray"):
        activity = activity.toarray()
    return np.asarray(activity)


def global_program_activity_violins(X, working_dir):
    """Violin plot of every program's activity across all cells."""
    save_fname = os.path.join(working_dir, "all_programs_violin_plot.png")

    n_programs = X.shape[1]
    n_cols = 5
    n_rows = math.ceil(n_programs / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4), squeeze=False
    )
    axes = axes.flatten()

    for i in tqdm(range(n_programs)):
        program_name = "Program " + str(i)
        ax = axes[i]
        sns.violinplot(y=X[:, i], ax=ax, inner="quartiles", color="indianred")
        ax.set_title(program_name)
        ax.set_ylabel("Program activity")
        ax.set_xlabel("")
        ax.set_xticks([])

    for j in range(n_programs, n_rows * n_cols):
        fig.delaxes(axes[j])

    fig.suptitle("Violin plots of program activities", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def program_correlations(
    X,
    working_dir,
    image_name="program_correlation_heatmap.png",
    title="Program Correlation Heatmap",
    label_prefix="program_",
    tick_fontsize=9,
    show_axis_labels=True,
    show_colorbar_label=True,
    colorbar_shrink=1.0,
    x_tick_rotation=45,
):
    """Heatmap of how similarly programs behave across cells.

    Correlating the per-cell activities shows which programs are co-active, which
    the report uses to group related programs.
    """
    logger.info("COMPUTING PROGRAM-PROGRAM CORRELATIONS")
    save_fname = os.path.join(working_dir, image_name)

    corr = np.corrcoef(X, rowvar=False)
    if not np.all(np.isfinite(corr)):
        logger.warning("Replacing non-finite program correlations with zero")
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)

    dist = np.clip(1 - (corr + 1) / 2, 0, 1)

    if corr.shape[0] > 1:
        condensed_dist = squareform(dist, checks=False)
        linkage_matrix = linkage(condensed_dist, method="average")
        ordered_indices = leaves_list(linkage_matrix)
    else:
        ordered_indices = np.arange(corr.shape[0])

    corr_ordered = corr[ordered_indices, :][:, ordered_indices]
    program_labels = [f"{label_prefix}{i}" for i in ordered_indices]

    plt.figure(figsize=(18, 15))
    ax = sns.heatmap(
        corr_ordered,
        xticklabels=program_labels,
        yticklabels=program_labels,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={
            "label": "Correlation" if show_colorbar_label else "",
            "shrink": colorbar_shrink,
        },
    )
    plt.title(title)
    plt.xlabel("Programs" if show_axis_labels else "")
    plt.ylabel("Programs" if show_axis_labels else "")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=x_tick_rotation,
        ha="center" if x_tick_rotation == 90 else "right",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(save_fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def leiden_cluster_and_plot(
    X,
    embedding,
    global_dir,
    leiden_resolution,
    neighbor_adata=None,
):
    """Cluster cells by their program activities rather than by expression.

    Cells sharing an activity profile group together, so the clusters correspond
    to combinations of active programs.
    """
    save_fname = os.path.join(global_dir, "global_umap_leiden.png")

    if neighbor_adata is None:
        temp_adata = ad.AnnData(X)
        logger.info("COMPUTING NEAREST NEIGHBORS FOR LEIDEN")
        start = perf_counter()
        rsc.pp.neighbors(temp_adata, use_rep="X")
        logger.info(
            "Leiden neighbor graph completed in %.2f seconds", perf_counter() - start
        )
    else:
        temp_adata = neighbor_adata
        logger.info("REUSING UMAP NEAREST-NEIGHBOR GRAPH FOR LEIDEN")

    logger.info("COMPUTING LEIDEN CLUSTERING")
    start = perf_counter()
    rsc.tl.leiden(temp_adata, resolution=leiden_resolution, random_state=0)
    logger.info("Leiden clustering completed in %.2f seconds", perf_counter() - start)
    cluster_labels = temp_adata.obs["leiden"].astype(int).values

    n_clusters = cluster_labels.max() + 1

    if n_clusters <= 20:
        palette = sns.color_palette("tab20", n_clusters)
    elif n_clusters <= 40:
        palette = sns.color_palette("tab20", 20) + sns.color_palette(
            "tab20b", n_clusters - 20
        )
    elif n_clusters <= 60:
        palette = (
            sns.color_palette("tab20", 20)
            + sns.color_palette("tab20b", 20)
            + sns.color_palette("tab20c", n_clusters - 40)
        )
    else:
        palette = sns.color_palette("hsv", n_clusters)

    cluster_colors = [palette[i] for i in cluster_labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        embedding[:, 0], embedding[:, 1], c=cluster_colors, s=5, alpha=0.7, linewidths=0
    )

    for k in range(n_clusters):
        mask = cluster_labels == k
        if np.sum(mask) == 0:
            continue
        x_med = np.median(embedding[mask, 0])
        y_med = np.median(embedding[mask, 1])
        circle = plt.Circle((x_med, y_med), radius=0.7, color="dimgray", zorder=10)
        ax.add_patch(circle)
        ax.text(
            x_med,
            y_med,
            str(k + 1),
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Visualization - Leiden Clusters")

    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        mpatches.Patch(color=palette[i], label=f"Cluster {i + 1}")
        for i in range(n_clusters)
    ]
    ncol = int(np.ceil(n_clusters / 20))
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="Clusters",
        fontsize=9,
        ncol=ncol,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return cluster_labels, palette


def umap_cells_by_program_activity(X, working_dir):
    """Embed cells by their program activities, so proximity reflects a shared activity profile."""
    save_fname = os.path.join(working_dir, "global_umap.png")

    temp_adata = ad.AnnData(X)

    logger.info("COMPUTING NEAREST NEIGHBORS")
    start = perf_counter()
    rsc.pp.neighbors(temp_adata, use_rep="X")
    logger.info(
        "Nearest-neighbor graph completed in %.2f seconds", perf_counter() - start
    )

    logger.info("COMPUTING UMAP")
    start = perf_counter()
    rsc.tl.umap(temp_adata, random_state=0)
    logger.info("UMAP completed in %.2f seconds", perf_counter() - start)

    embedding = temp_adata.obsm["X_umap"]

    plt.figure(figsize=(6, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.1)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of cells by program activity")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return embedding, temp_adata


def _cell_type_color_dict(adata, cell_type_column):
    """Return a dict mapping cell type -> color, matching the palette logic
    used by ``plot_umap_colored_by_cell_type`` so that other plots can share
    the same colormap. Cell types with <=100 cells are gray.
    """
    from collections import Counter

    cell_types = adata.obs[cell_type_column].astype(str).values
    unique_cell_types = np.unique(cell_types)
    ct_counts = Counter(cell_types)
    large_cell_types = [ct for ct in unique_cell_types if ct_counts[ct] > 100]
    small_cell_types = [ct for ct in unique_cell_types if ct_counts[ct] <= 100]

    n_large = len(large_cell_types)
    if n_large <= 20:
        palette = sns.color_palette("tab20", n_large)
    elif n_large <= 40:
        palette = sns.color_palette("tab20", 20) + sns.color_palette(
            "tab20b", n_large - 20
        )
    elif n_large <= 60:
        palette = (
            sns.color_palette("tab20", 20)
            + sns.color_palette("tab20b", 20)
            + sns.color_palette("tab20c", n_large - 40)
        )
    else:
        palette = sns.color_palette("hsv", n_large)

    color_dict = {}
    for i, ct in enumerate(large_cell_types):
        color_dict[ct] = palette[i]
    for ct in small_cell_types:
        color_dict[ct] = "lightgray"
    return color_dict


def plot_compact_interactive_cell_type_umap(
    adata,
    embedding,
    working_dir,
    cell_type_column="cell_type",
    hover_width=480,
    hover_height=400,
):
    """Save a legend-free UMAP and compact hover lookup for the HTML report."""
    if cell_type_column not in adata.obs.columns:
        logger.warning(
            "Skipping interactive cell-type UMAP: obs[%r] is unavailable",
            cell_type_column,
        )
        return

    cell_types = adata.obs[cell_type_column].astype(str).to_numpy()
    categories = sorted(np.unique(cell_types).tolist())
    if len(categories) >= np.iinfo(np.uint16).max:
        raise ValueError("Interactive UMAP supports fewer than 65,535 cell types")
    category_to_index = {name: index for index, name in enumerate(categories)}
    color_dict = _cell_type_color_dict(adata, cell_type_column)
    colors = [mcolors.to_hex(color_dict[name]) for name in categories]

    x = np.asarray(embedding[:, 0], dtype=np.float64)
    y = np.asarray(embedding[:, 1], dtype=np.float64)
    x_pad = max(1e-9, 0.02 * (x.max() - x.min()))
    y_pad = max(1e-9, 0.02 * (y.max() - y.min()))
    x_min, x_max = x.min() - x_pad, x.max() + x_pad
    y_min, y_max = y.min() - y_pad, y.max() + y_pad

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in categories:
        mask = cell_types == name
        ax.scatter(
            x[mask],
            y[mask],
            s=3,
            color=color_dict[name],
            alpha=0.7,
            edgecolor="none",
            rasterized=True,
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        os.path.join(working_dir, "umap_by_cell_type_compact.png"),
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)

    x_bin = np.clip(
        ((x - x_min) / (x_max - x_min) * (hover_width - 1)).astype(np.int32),
        0,
        hover_width - 1,
    )
    y_bin = np.clip(
        ((y_max - y) / (y_max - y_min) * (hover_height - 1)).astype(np.int32),
        0,
        hover_height - 1,
    )
    flat_bin = y_bin * hover_width + x_bin
    if len(adata) >= np.iinfo(np.uint32).max:
        raise ValueError("Interactive UMAP supports fewer than 2^32 cells")
    x_grid = (x - x_min) / (x_max - x_min) * (hover_width - 1)
    y_grid = (y_max - y) / (y_max - y_min) * (hover_height - 1)
    distance = (x_grid - x_bin - 0.5) ** 2 + (y_grid - y_bin - 0.5) ** 2
    cell_index = np.arange(len(adata), dtype=np.uint32)
    order = np.lexsort((cell_index, distance, flat_bin))
    sorted_bins = flat_bin[order]
    first_in_bin = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
    representatives = order[first_in_bin]
    representative_bins = flat_bin[representatives]

    missing_cell = np.iinfo(np.uint32).max
    hover_cells = np.full(
        hover_width * hover_height, missing_cell, dtype="<u4"
    )
    hover_cells[representative_bins] = representatives.astype(np.uint32)
    hover_labels = np.full(
        hover_width * hover_height, np.iinfo(np.uint16).max, dtype="<u2"
    )
    category_indices = np.fromiter(
        (category_to_index[name] for name in cell_types),
        dtype=np.uint16,
        count=len(cell_types),
    )
    hover_labels[representative_bins] = category_indices[representatives]

    hover_data = {
        "width": hover_width,
        "height": hover_height,
        "missing": int(np.iinfo(np.uint16).max),
        "missingCell": int(missing_cell),
        "cellTypes": categories,
        "colors": colors,
        "labels": base64.b64encode(hover_labels.tobytes()).decode("ascii"),
        "cellIndices": base64.b64encode(hover_cells.tobytes()).decode("ascii"),
    }
    with open(
        os.path.join(working_dir, "umap_by_cell_type_hover.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(hover_data, handle, separators=(",", ":"))


def plot_cell_type_counts_bar(adata, working_dir, cell_type_column="cell_type"):
    """
    Bar plot of the number of cells in each cell type from adata.obs['cell_type'].
    Bar colors match the UMAP-by-cell-type palette. Saved as
    cell_type_counts_bar.png in working_dir.
    """
    if cell_type_column not in adata.obs.columns:
        logger.warning(
            "adata.obs has no %r column; skipping cell-type bar plot.",
            cell_type_column,
        )
        return

    cell_types = adata.obs[cell_type_column].astype(str).values
    from collections import Counter

    counts = Counter(cell_types)
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [c for c, _ in ordered]
    values = [v for _, v in ordered]

    color_dict = _cell_type_color_dict(adata, cell_type_column)
    bar_colors = [color_dict[c] for c in labels]

    fig_width = max(6, min(0.4 * len(labels) + 2, 14))
    plt.figure(figsize=(fig_width, 4.5))
    ax = plt.gca()
    bars = ax.bar(
        range(len(labels)), values, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell counts per cell type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    save_path = os.path.join(working_dir, "cell_type_counts_bar.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_umap_colored_by_cell_type(
    adata,
    embedding,
    working_dir,
    cell_type_column="cell_type",
):
    """
    Make a UMAP plot colored by cell types from adata.obs['cell_type'].
    - adata: AnnData object with .obs['cell_type']
    - embedding: (n_cells, 2) numpy array (UMAP coordinates)
    - working_dir: directory to save the plot (should be report_plots/global_plots or similar)
    Adds a number for each cell type, placed directly over a representative cell of that type,
    with a gray circle and white number.
    Only cell types with >100 samples are colored; others are shown in gray.
    """

    if cell_type_column not in adata.obs.columns:
        logger.warning(
            "adata.obs has no %r column; skipping cell-type UMAP.",
            cell_type_column,
        )
        return
    cell_types = adata.obs[cell_type_column].astype(str).values
    unique_cell_types = np.unique(cell_types)
    n_types = len(unique_cell_types)

    color_dict = _cell_type_color_dict(adata, cell_type_column)

    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    for ct in unique_cell_types:
        idx = cell_types == ct
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=5,
            alpha=0.7,
            label=ct,
            color=color_dict[ct],
            edgecolor="none",
        )

    label_positions = []
    for i, ct in enumerate(unique_cell_types):
        idx = cell_types == ct
        points = embedding[idx]
        if points.shape[0] == 0:
            label_positions.append((np.nan, np.nan))
            continue

        centroid = np.median(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        best_idx = np.argmin(dists)
        x_label, y_label = points[best_idx]
        label_positions.append((x_label, y_label))

        circle = plt.Circle(
            (x_label, y_label), radius=0.7, color="dimgray", zorder=10, alpha=0.85
        )
        ax.add_patch(circle)
        txt = ax.text(
            x_label,
            y_label,
            str(i + 1),
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
        )
        txt.set_path_effects(
            [
                PathEffects.Stroke(linewidth=2, foreground="black", alpha=0.7),
                PathEffects.Normal(),
            ]
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of cells colored by cell type (>100 colored, ≤100 gray)")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    ncol = (n_types - 1) // 20 + 1
    legend_labels = [f"{i + 1}: {ct}" for i, ct in enumerate(unique_cell_types)]
    handles = []
    for i, ct in enumerate(unique_cell_types):
        markerfacecolor = color_dict[ct]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=markerfacecolor,
                markersize=7,
                label=legend_labels[i],
            )
        )
    plt.legend(
        handles=handles,
        markerscale=2,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title="Cell Type",
        ncol=ncol,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.5,
        frameon=True,
    )
    save_path = os.path.join(working_dir, "umap_by_cell_type.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_all_plots(
    working_dir,
    leiden_resolution,
    cell_type_column="cell_type",
):
    """Generate every figure the report step assembles into the final document."""
    UMAP_POINT_SIZE = 3
    UMAP_ALPHA = 0.7

    logger.info("Report analysis backend: rapids-singlecell")

    supp_materials_dir = os.path.join(working_dir, "supp_material")

    program_adata, X, program_activity_path = load_program_activity(working_dir)
    if cell_type_column not in program_adata.obs.columns:
        raise ValueError(
            f"wePAS.h5ad has no configured global cell-type label {cell_type_column!r}"
        )

    report_plot_dir = os.path.join(working_dir, "report_plots")
    if not os.path.exists(report_plot_dir):
        os.makedirs(report_plot_dir)

    global_figures_dir = os.path.join(report_plot_dir, "global_plots")
    if not os.path.exists(global_figures_dir):
        os.makedirs(global_figures_dir)

    program_specific_dir = os.path.join(report_plot_dir, "program_plots")
    if not os.path.exists(program_specific_dir):
        os.makedirs(program_specific_dir)

    cluster_specific_dir = os.path.join(report_plot_dir, "cluster_plots")
    if not os.path.exists(cluster_specific_dir):
        os.makedirs(cluster_specific_dir)

    supplemental_figures_dir = os.path.join(working_dir, "supplemental_figures")
    if not os.path.exists(supplemental_figures_dir):
        os.makedirs(supplemental_figures_dir)

    embedding, neighbor_adata = umap_cells_by_program_activity(
        X,
        global_figures_dir,
    )
    program_adata.obsm["X_umap"] = embedding
    embedding_path = os.path.join(supp_materials_dir, "umap_embedding.npy")
    np.save(embedding_path, embedding)
    program_adata.write(program_activity_path)
    logger.info(
        f"Saved UMAP embedding to adata.obsm['X_umap'] and updated {program_activity_path}"
    )

    logger.info("Report plot cell-type column: %s", cell_type_column)
    plot_cell_type_counts_bar(
        program_adata,
        global_figures_dir,
        cell_type_column=cell_type_column,
    )
    plot_umap_colored_by_cell_type(
        program_adata,
        embedding,
        global_figures_dir,
        cell_type_column=cell_type_column,
    )
    plot_compact_interactive_cell_type_umap(
        program_adata,
        embedding,
        global_figures_dir,
        cell_type_column=cell_type_column,
    )

    global_program_activity_violins(X, global_figures_dir)
    program_correlations(X, global_figures_dir)
    sepas_activity = load_sepas_activity(working_dir)
    program_correlations(
        sepas_activity,
        supplemental_figures_dir,
        image_name="sepas_program_correlation_heatmap.png",
        title="Program-Program Correlation from sePAS",
        label_prefix="P",
        tick_fontsize=12,
        show_axis_labels=False,
        show_colorbar_label=False,
        colorbar_shrink=0.42,
        x_tick_rotation=90,
    )
    cluster_labels, cluster_palette = leiden_cluster_and_plot(
        X,
        embedding,
        global_figures_dir,
        leiden_resolution,
        neighbor_adata=neighbor_adata,
    )

    program_adata.obs["leiden"] = pd.Categorical(cluster_labels.astype(str))
    program_adata.write(program_activity_path)

    cluster_labels_path = os.path.join(supp_materials_dir, "cluster_labels.npy")
    np.save(cluster_labels_path, cluster_labels)

    _render_plot_batches(
        X,
        sepas_activity,
        embedding,
        cluster_labels,
        cluster_palette,
        report_plot_dir,
        program_specific_dir,
        cluster_specific_dir,
        UMAP_POINT_SIZE,
        UMAP_ALPHA,
    )


if __name__ == "__main__":
    if not _IS_PLOT_WORKER:
        raise SystemExit(
            f"usage: {sys.executable} {__file__} "
            f"{_PLOT_WORKER_FLAG} MANIFEST.json"
        )
    _render_plot_manifest(sys.argv[2])
