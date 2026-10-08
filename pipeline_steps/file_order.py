"""Deterministic ordering for AnnData inputs and generated cell chunks."""

import os
import re

_CELL_CHUNK_PATTERN = re.compile(r"^raw_adata_chunk_(\d+)\.h5ad$")


def sort_h5ad_paths(paths):
    """Sort generated chunks numerically and other AnnData paths lexicographically."""

    def sort_key(path):
        filename = os.path.basename(path)
        match = _CELL_CHUNK_PATTERN.match(filename)
        if match:
            return 0, int(match.group(1)), path
        return 1, filename, path

    return sorted(paths, key=sort_key)


def list_cell_chunks(directory):
    """Return generated ``raw_adata_chunk_<n>.h5ad`` files in numeric order."""
    paths = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for filename in filenames:
            if _CELL_CHUNK_PATTERN.match(filename):
                paths.append(os.path.join(root, filename))
    return sort_h5ad_paths(paths)
