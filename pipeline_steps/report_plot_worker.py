"""CPU worker for batched report-figure rendering."""

import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _activity_colormap(only_nonnegative):
    if only_nonnegative:
        return mcolors.LinearSegmentedColormap.from_list(
            "gray_red", ["#E0E0E0", "#FF9999", "#CC0000"], N=256
        )
    return mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red",
        ["#2166ac", "#67a9cf", "#E0E0E0", "#FF6666", "#CC0000"],
        N=256,
    )


def _render_program_umap(X, embedding, output_dir, program_index, point_size, alpha):
    activity = X[:, program_index]
    activity_min = np.nanmin(activity)
    activity_max = np.nanmax(activity)
    only_nonnegative = activity_min >= 0
    cmap = _activity_colormap(only_nonnegative)

    absmax = max(abs(activity_min), abs(activity_max))
    if np.isnan(absmax) or absmax == 0:
        vmin, vmax, vcenter = activity_min, activity_max, 0
    elif only_nonnegative:
        vmin, vmax, vcenter = activity_min, activity_max, activity_min
    elif activity_max <= 0:
        vmin, vmax, vcenter = activity_min, activity_max, activity_max
    else:
        vmin, vmax, vcenter = -absmax, absmax, 0

    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6
    if only_nonnegative or not (vmin < vcenter < vmax):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=activity,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolor="none",
        norm=norm,
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"Program {program_index} activity")
    colorbar = plt.colorbar(scatter, label="Program activity")
    if activity_min != activity_max:
        colorbar.ax.set_ylim(activity_min, activity_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"umap_program_activity_{program_index}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def _render_program_violin(X, labels, palette, output_dir, program_index):
    unique_clusters = np.unique(labels)
    cluster_id_map = {cluster: i for i, cluster in enumerate(sorted(unique_clusters))}
    mapped_labels = np.array([cluster_id_map[cluster] for cluster in labels])
    n_clusters = len(unique_clusters)
    colors = (
        palette
        if len(palette) >= n_clusters
        else sns.color_palette("tab20", n_clusters)
    )
    cluster_names = [f"Cluster {cluster_id_map[cluster] + 1}" for cluster in labels]
    order = [f"Cluster {i + 1}" for i in range(n_clusters)]

    plt.figure(figsize=(max(8, min(n_clusters, 10) * 0.8), 5))
    axis = sns.violinplot(
        x="Cluster",
        y="Program Activity",
        data={
            "Program Activity": X[:, program_index],
            "Cluster": cluster_names,
            "ClusterIdx": mapped_labels,
        },
        order=order,
        palette=colors,
        hue="ClusterIdx",
        hue_order=list(range(n_clusters)),
        dodge=False,
        cut=0,
        inner="quartile",
    )
    if axis.legend_:
        axis.legend_.remove()
    axis.set_title(f"Program {program_index} Activity by Cluster")
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Program Activity")
    if n_clusters > 6:
        plt.setp(
            axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"program_{program_index}_activity_violin.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def _render_cluster_violin(X, labels, palette, output_dir, cluster_index):
    cluster_values = X[labels == cluster_index]
    n_cells, n_programs = cluster_values.shape
    frame = pd.DataFrame(
        {
            "Program": np.repeat(np.arange(n_programs), n_cells),
            "Activity": cluster_values.T.reshape(-1),
        }
    )
    unique_clusters = sorted(int(value) for value in np.unique(labels))
    palette_index = unique_clusters.index(cluster_index)
    color = palette[palette_index] if palette_index < len(palette) else "gray"

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


def render_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    X = np.load(manifest["activity_path"], mmap_mode="r")
    embedding = np.load(manifest["embedding_path"], mmap_mode="r")
    labels = np.load(manifest["cluster_labels_path"], mmap_mode="r")
    palette = np.load(manifest["cluster_palette_path"])

    for task_type, index in manifest["tasks"]:
        if task_type == "program_umap":
            _render_program_umap(
                X,
                embedding,
                manifest["program_umap_dir"],
                index,
                manifest["point_size"],
                manifest["alpha"],
            )
        elif task_type == "program_violin":
            _render_program_violin(
                X, labels, palette, manifest["program_violin_dir"], index
            )
        elif task_type == "cluster_violin":
            _render_cluster_violin(
                X, labels, palette, manifest["cluster_violin_dir"], index
            )
        else:
            raise ValueError(f"Unknown plot task: {task_type}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m pipeline_steps.report_plot_worker MANIFEST.json"
        )
    render_manifest(sys.argv[1])
