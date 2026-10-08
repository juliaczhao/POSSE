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
hardware.

## Usage

### Input data

The input may be one `.h5ad` file or a directory of `.h5ad` files. Files must contain raw counts
restricted to protein-coding genes.

To select annotated populations, set `global.cell_type_column` to the relevant `.obs` column and
list the desired labels in `global.cell_types_to_run`. An empty list runs all cells. Reports use
`report.report_cell_type_column`, which defaults to `cell_type_column` when omitted.

### Configuration

POSSE does not require hyperparameter tuning. The configuration file is for input data, plotting,
and LLM annotation preferences. The fields you need to set for your own data are:

| Parameter | Block | Description |
| :-- | :-- | :--------------------------------------------------------------------------- |
| `raw_data_path` | `global` | The raw-counts input: either a single `.h5ad` file or a directory of chunked `.h5ad` files (both are accepted). |
| `output_dir` | `global` | Directory where every output of the pipeline is written. |
| `cell_type_column`, `cell_types_to_run` | `global` | The `.obs` annotation column and labels to select for the run; `[]` runs all cells. |
| `filter_raw_data` | `preprocess` | Apply the UMI, expressed-gene and mitochondrial filters. Defaults to `true` and usually removes a few percent of cells (approximately 2–5% in the tested CRC and HLCA datasets). |
| `api_key`, `model`, `reasoning_effort` | `llm_program_summaries` | Optional program-annotation credentials and model settings; leave `api_key` empty to skip annotation. |
| `report_cell_type_column` | `report` | The `.obs` annotation column used in plots, LLM annotation, and the HTML report; defaults to `cell_type_column`. |

All other settings use validated defaults and should generally remain unchanged.

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

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). Free for research, teaching and other
noncommercial purposes, including use by universities, nonprofits and government
institutions. Commercial use is not permitted.
