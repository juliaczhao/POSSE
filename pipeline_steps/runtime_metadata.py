"""Runtime provenance and final result summaries."""

import json
import logging
import os
import platform
import subprocess
import sys

import h5py
import numpy as np

from pipeline_steps.file_order import list_cell_chunks
from pipeline_steps.utils import load_programs

logger = logging.getLogger(__name__)


def _command_output(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_metadata(repository_root):
    commit = _command_output(["git", "-C", repository_root, "rev-parse", "HEAD"])
    branch = _command_output(["git", "-C", repository_root, "branch", "--show-current"])
    status = _command_output(["git", "-C", repository_root, "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if commit is None else status is not None,
    }


def nvidia_driver_versions():
    output = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if output is None:
        return []
    return sorted({line.strip() for line in output.splitlines() if line.strip()})


def cuda_toolkit_version():
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc"
    output = _command_output([nvcc, "--version"])
    return output.splitlines()[-1].strip() if output else None


def platform_description():
    return platform.platform()


def _h5ad_cell_count(path):
    with h5py.File(path, "r") as handle:
        observations = handle["obs"]
        index_name = observations.attrs.get("_index", "_index")
        if isinstance(index_name, bytes):
            index_name = index_name.decode("utf-8")
        return int(observations[index_name].shape[0])


def summarize_results(working_dir):
    chunk_paths = list_cell_chunks(os.path.join(working_dir, "anndata_files"))
    starting_cells = sum(_h5ad_cell_count(path) for path in chunk_paths)

    gene_names_path = os.path.join(working_dir, "gene_names", "gene_names.npy")
    starting_genes = int(len(np.load(gene_names_path, allow_pickle=True)))

    programs_path = os.path.join(working_dir, "clique_based_programs", "programs.json")
    programs = load_programs(programs_path)
    sizes = [len(program) for program in programs]
    unique_genes = {gene for program in programs for gene in program}
    smallest = min(sizes) if sizes else 0
    largest = max(sizes) if sizes else 0

    return {
        "starting_cells": starting_cells,
        "starting_genes": starting_genes,
        "program_count": len(programs),
        "program_size_min": smallest,
        "program_size_max": largest,
        "program_size_range": [smallest, largest],
        "total_program_gene_memberships": sum(sizes),
        "unique_program_genes": len(unique_genes),
    }


def write_runtime_summary(
    output_dir,
    working_dir,
    args,
    repository_root,
    started_at,
    finished_at,
    elapsed_times,
):
    import torch

    devices = []
    if torch.cuda.is_available():
        devices = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_seconds": (finished_at - started_at).total_seconds(),
        "step_seconds": elapsed_times,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cuda_toolkit_version": cuda_toolkit_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
        "nvidia_driver_versions": nvidia_driver_versions(),
        "git": git_metadata(repository_root),
        "working_dir": os.path.abspath(working_dir),
        "results": summarize_results(working_dir),
    }

    execution_path = os.path.join(working_dir, "supp_material", "xrfm_execution.json")
    if os.path.exists(execution_path):
        with open(execution_path, "r", encoding="utf-8") as handle:
            summary["xrfm_execution"] = json.load(handle)

    path = os.path.join(output_dir, "runtime_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    logger.info("Runtime summary written to %s", path)
    logger.info("Result statistics: %s", summary["results"])
