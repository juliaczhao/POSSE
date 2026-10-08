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
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    "${requirements_path}"
uv pip check --python "${environment_path}/bin/python"

PYTHONPATH="${repository_dir}" "${environment_path}/bin/python" - <<'PY'
import cupy
import openai
import pydantic
import rapids_singlecell
import run_pipeline_manual
import torch
from pipeline_steps.llm_program_summaries import _program_annotation_model
from pipeline_steps.statistical_ids_significance_testing import _configure_cuda_home

_program_annotation_model()
cuda_home = _configure_cuda_home()
if cuda_home is None:
    raise RuntimeError(
        "CUDA toolkit headers were not found; set CUDA_HOME or add nvcc to PATH"
    )

print(
    "POSSE environment imports successfully "
    f"(torch {torch.__version__}, CUDA {torch.version.cuda}, "
    f"rapids-singlecell {rapids_singlecell.__version__}, "
    f"openai {openai.__version__}, pydantic {pydantic.__version__}, "
    f"CUDA_HOME {cuda_home})"
)
if torch.cuda.is_available():
    print(f"Visible NVIDIA GPUs: {torch.cuda.device_count()}")
    print(f"CuPy-visible devices: {cupy.cuda.runtime.getDeviceCount()}")
else:
    print("Warning: no NVIDIA GPU is currently visible; POSSE requires one to run the pipeline.")
PY

echo "Activate with: source ${environment_path}/bin/activate"
