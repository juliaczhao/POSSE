# POSSE

**Programs of Strongly-connected Significant Edges**

POSSE identifies gene programs from large-scale single-cell RNA-seq data. It builds on
[InterDependence Scores (IDS)](https://www.pnas.org/doi/10.1073/pnas.2509860122), a measure of
non-linear dependence between variables that scales to hundreds of millions of samples.

From gene-gene IDS values, significance testing identifies strongly-connected significant edges
(SSE). Rooted in graph theory, those edges are grouped into cliques, which are then grown and
merged into the final programs (posses). Each program is characterised by a per-cell program
activity score and an LLM coherence annotation.

Citation and paper link coming soon.

This is a minimal research codebase. A pip-installable version is coming soon.

## Installation

### Install the environment

Installation uses [uv](https://docs.astral.sh/uv/getting-started/installation/); install it first.
`install.sh` uses uv to build the environment and fetches Python 3.11 automatically if it is not
already available.

Clone the repository and install the pinned environment:

```bash
git clone https://github.com/juliaczhao/POSSE.git
cd POSSE
./install.sh
```

`install.sh` creates a Python 3.11 virtual environment (default: `~/posse`), installs the pinned
dependencies from `requirements.lock.txt`, and verifies that every import resolves. Activate it
with:

```bash
source ~/posse/bin/activate
```

### Download model files

The xRFM models used for speeding up significance testing are too large for GitHub.
[Download them here](https://drive.google.com/drive/folders/1FzsUYkHDrGApRNiHraavJIdNbMiy7ddB?usp=drive_link)
and place both files in `beta_fit_model/` in this repository:

- `1M_20_HPO_xrfm_mu.pkl`
- `1M_20_HPO_xrfm_log_kappa.pkl`

See [`beta_fit_model/instructions.md`](beta_fit_model/instructions.md) for checksums to verify the
downloads and a description of each file.

### Hardware requirements

POSSE runs on Linux with an NVIDIA GPU. The pinned dependencies are CUDA 12 builds, so a
CUDA 12.x driver and toolkit are required, and the pipeline needs a GPU with **80 GB** of memory.

## Usage

### Input data

The pipeline expects a single `.h5ad` file containing **raw counts**, restricted to protein-coding
genes. No quality-control filtering is required beforehand; the pipeline performs its own.

If cell types are already annotated, place them in `.obs['cell_type']`. Otherwise a placeholder is
inserted and the pipeline runs on all cells together.

### Configuration

All settings live in `pipeline_config.json`, provided in the repository. The fields you need to set
for your own data are:

| Parameter | Block | Description |
| :-- | :-- | :--------------------------------------------------------------------------- |
| `raw_data_path` | `global` | The raw-counts input: either a single `.h5ad` file or a directory of chunked `.h5ad` files (both are accepted). |
| `output_dir` | `global` | Directory where every output of the pipeline is written. |
| `cell_type_label` | `global` | The `.obs` column holding the cell-type labels (default `cell_type`). |
| `cell_types` | `global` | List of cell types to run on; `[]` runs on all cells. |
| `filter_raw_data` | `preprocess` | Run the built-in QC filter (UMI, gene, and mitochondrial cutoffs). Recommended for large or high-mitochondrial datasets. |
| `louvain_backend` | `program_finding` | Community-detection backend: `cpu` (fully reproducible) or `gpu` (faster, reproducible only to floating point). |
| `provide_celltype` | `llm_program_summaries` | Pass the dataset's cell types to the language model during annotation. On grounds interpretations in the data's cell types; off keeps them gene-only. |

All other parameters are the empirical-sweep defaults used for the published results. They are best
left fixed, though you are welcome to change them.

### Running the pipeline

Either edit `pipeline_config.json` directly — `run_pipeline_manual.py` reads it by default:

```bash
python run_pipeline_manual.py
```

or, if you have several datasets to run, keep a separate config for each and pass it as an
argument:

```bash
python run_pipeline_manual.py my_dataset_config.json
```

`run_pipeline_manual.py` runs Steps 1–10 in order, each reading the previous step's output from
`output_dir`. To run only part of the pipeline, comment out the steps you don't need in that file.

### LLM annotation

Step 9 names and describes each program with a language model. It runs only when a real API key
is set in `pipeline_config.json` under `llm_program_summaries.api_key` (OpenAI by default; any
provider can be substituted).

Without a key, the pipeline still completes: Step 9 writes a summaries file containing the gene
lists only, and Step 10 produces the report with those gene lists and the plots but no titles or
descriptions. To add annotations afterwards, fill the `Title` and `Summary` columns of
`llm_summaries/program_summaries_with_genes.csv` — by hand, or by pasting a program's genes into
any LLM interface (the prompts are in `prompt_templates/`) — and re-run Step 10.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). Free for research, teaching and other
noncommercial purposes, including use by universities, nonprofits and government
institutions. Commercial use is not permitted.
