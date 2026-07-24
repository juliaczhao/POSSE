#!/usr/bin/env bash
# Create the pipeline environment from requirements.lock.txt and verify it.
#
# Usage: ./install.sh [venv_path]        (default: ~/posse)
set -euo pipefail

VENV="${1:-$HOME/posse}"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.lock.txt"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv venv --python 3.11 "$VENV"

# --no-deps is required: torch and the RAPIDS cu12 wheels declare incompatible
# nvidia-* pins, so any resolver refuses the set.
uv pip install --no-deps --python "$VENV/bin/python" -r "$REQ"

"$VENV/bin/python" - <<'PY'
import importlib

modules = [
    "anndata", "community", "cudf", "cugraph", "cuml", "cuvs", "diptest", "docx",
    "hickle", "ids", "igraph", "kermac", "kneed", "matplotlib", "networkx", "numpy",
    "openai", "pandas", "pytabkit", "rapids_singlecell", "scanpy", "scipy", "seaborn",
    "skimage", "sklearn", "statsmodels", "torch", "tqdm", "xrfm",
]
failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")

if failed:
    raise SystemExit("environment incomplete:\n  " + "\n  ".join(failed))

import torch
print(f"environment OK (torch {torch.__version__}, CUDA {torch.version.cuda})")
PY

echo "Activate with: source $VENV/bin/activate"
