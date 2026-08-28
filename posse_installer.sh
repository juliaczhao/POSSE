#!/usr/bin/env bash
# Install POSSE into a Python 3.11 environment.
#
# Usage: ./posse_installer.sh [environment_path]  (default: ~/posse)
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [environment_path]  (default: ~/posse)" >&2
    exit 2
fi

environment_path=${1:-"${HOME}/posse"}
repository_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
requirements_path="${repository_dir}/posse_requirements.txt"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [[ -e "${environment_path}" ]]; then
    echo "refusing to replace existing path: ${environment_path}" >&2
    echo "Choose another path or move the existing environment first." >&2
    exit 1
fi

uv venv --python 3.11 "${environment_path}"
uv pip sync \
    --python "${environment_path}/bin/python" \
    --require-hashes \
    --index-strategy unsafe-best-match \
    "${requirements_path}"
uv pip check --python "${environment_path}/bin/python"

PYTHONPATH="${repository_dir}" "${environment_path}/bin/python" - <<'PY'
import run_pipeline_manual
import torch

print(f"POSSE imports successfully (torch {torch.__version__}, CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("Warning: no NVIDIA GPU is currently visible; POSSE requires one to run the pipeline.")
PY

echo "Activate with: source ${environment_path}/bin/activate"
