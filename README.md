# POSSE

**Programs of Strongly-connected Significant Edges**

POSSE identifies gene programs from large-scale single-cell RNA-seq data. It builds on
[InterDependence Scores (IDS)](https://www.pnas.org/doi/10.1073/pnas.2509860122), a measure of
non-linear dependence between variables that scales to hundreds of millions of samples.

From gene-gene IDS values, significance testing identifies strongly-connected significant edges
(SSE). Positive clique seeds are grown and consolidated into signed positive quasi-cliques, which
define the final programs (posses). Each program is characterised by a per-cell program activity
score and an LLM coherence annotation.

Citation and paper link coming soon.

## Installation

### Install the environment

Installation uses [uv](https://docs.astral.sh/uv/getting-started/installation/); install it first.

Clone the repository and run the hash-pinned installer:

```bash
git clone https://github.com/juliaczhao/POSSE.git
cd POSSE
./posse_installer.sh
source ~/posse/bin/activate
```

By default, the installer creates `~/posse`. To use another location, pass it as the only
argument:

```bash
./posse_installer.sh /path/to/posse
```

Use the installer to create the Python 3.11 environment from the pinned POSSE dependencies.

### Download model files

The xRFM models used for speeding up significance testing are too large for GitHub.
[Download them here](https://drive.google.com/drive/folders/1FzsUYkHDrGApRNiHraavJIdNbMiy7ddB?usp=drive_link)
and place both files in `beta_fit_model/` in this repository:

- `1M_20_HPO_xrfm_mu.pkl`
- `1M_20_HPO_xrfm_log_kappa.pkl`

See [`beta_fit_model/instructions.md`](beta_fit_model/instructions.md) for file checksums.

### Hardware requirements

POSSE runs on Linux with one or more 80 GB NVIDIA GPUs and has been tested on A100, H100 and H200
hardware. It assigns fixed xRFM batches to the visible GPUs.

## Usage

### Input data

The input may be one `.h5ad` file or a directory of `.h5ad` files. Files must contain raw counts
restricted to protein-coding genes. The built-in cell filter is enabled by default.

To select annotated populations, set `global.cell_type_label` to the relevant `.obs` column and
list the desired labels in `global.cell_types`. An empty list runs all cells.

### Configuration

POSSE does not require hyperparameter tuning. The configuration file is for input data, plotting,
and LLM annotation preferences. The fields you need to set for your own data are:

| Parameter | Block | Description |
| :-- | :-- | :--------------------------------------------------------------------------- |
| `raw_data_path` | `global` | The raw-counts input: either a single `.h5ad` file or a directory of chunked `.h5ad` files (both are accepted). |
| `output_dir` | `global` | Directory where every output of the pipeline is written. |
| `cell_type_label` | `global` | The `.obs` column holding the cell-type labels (default `cell_type`). This single setting is used for selection, LLM context, cell-type plots, AUROC, and the HTML report. |
| `cell_types` | `global` | List of cell types to run on; `[]` runs on all cells. |
| `filter_raw_data` | `preprocess` | Apply the UMI, expressed-gene and mitochondrial filters. Defaults to `true` and usually removes a few percent of cells (approximately 2–5% in the tested CRC and HLCA datasets). |
| `model_dir` | `statistical_program_finding` | Directory containing the two xRFM model files. |
| `api_key` | `llm_program_summaries` | OpenAI API key for optional program annotation. Leave empty to skip annotation. |
| `model` | `llm_program_summaries` | OpenAI model used for optional program names and descriptions (default `gpt-5.6-sol`). |
| `reasoning_effort` | `llm_program_summaries` | Reasoning effort for optional annotation (default `high`). |

All other parameters are the empirical-sweep defaults used for the published results. They are best
left fixed, though you are welcome to change them.

Program finding has three reproducibility controls: `gamma`, `min_clique_size`, and `seed`. Step 6
writes the final memberships to `clique_based_programs/programs.json`; it does not emit intermediate
program JSON files. The deterministic graph search uses one CPU worker.

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

Step 9 optionally names and describes each program. Set `llm_program_summaries.api_key` in the run
configuration and keep that file private. The configured cell labels are included in the request.
With no API key, the pipeline skips annotation and still writes the complete HTML report to
`reports/program_analysis_report.html`.

The annotation prompt receives one unsigned gene list per program. The report presents that same
gene list directly. Each program page pairs a full-size, hoverable author-annotated cell-type UMAP
with a same-size UMAP of continuous sePAS activity. The global visualizations page uses the same
hoverable cell-type UMAP and places its full legend in a matching-width box below the plot.
The sePAS program–program correlation heatmap is also interactive: hovering a matrix cell shows
the corresponding program pair with the same short titles used throughout the report.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). Free for research, teaching and other
noncommercial purposes, including use by universities, nonprofits and government
institutions. Commercial use is not permitted.
