# Model files

Significance testing (Step 4) assigns every gene pair a p-value from a null Beta distribution
fitted to its dependence scores. Fitting that distribution for each of ~20,000 genes with a
conventional numerical optimizer is slow, so we instead trained two
[xRFM](https://github.com/dmbeaglehole/xRFM) regressors to predict its parameters directly.

Place both model files in this directory (`beta_fit_model/`). Each is roughly 2.7 GB, so they are
distributed separately from the repository.

## Download

[Download the two files here](https://drive.google.com/drive/folders/1FzsUYkHDrGApRNiHraavJIdNbMiy7ddB?usp=drive_link)
and save them directly into `beta_fit_model/`:

- `1M_20_HPO_xrfm_mu.pkl`
- `1M_20_HPO_xrfm_log_kappa.pkl`

Set `statistical_program_finding.model_dir` in `pipeline_config.json` to the directory containing
these two canonically named files. The pipeline constructs each path by joining that directory
with the filename shown above. Absolute paths are accepted. The checked-in `beta_fit_model/`
default is resolved against the repository root and converted to an absolute path before Step 4.
To store an explicit path, run `realpath beta_fit_model` from the repository root and copy the
printed directory—not either individual model filename—into `model_dir`.

## What each file is

For a gene pair, the two models together predict the null Beta distribution that the pair's
InterDependence Score (IDS) would follow:

- `1M_20_HPO_xrfm_mu.pkl` — predicts **μ**, the mean of that Beta distribution.
- `1M_20_HPO_xrfm_log_kappa.pkl` — predicts **log κ**, its (log) concentration.

The pipeline converts (μ, κ) into the Beta shape parameters α = μ(κ − 1) and β = (1 − μ)(κ − 1) and
reads each pair's p-value off the resulting distribution.

## Verify the downloads

Confirm both files match the expected checksums before running:

```bash
md5sum 1M_20_HPO_xrfm_mu.pkl 1M_20_HPO_xrfm_log_kappa.pkl
```

| File | Size (bytes) | MD5 |
| --- | --- | --- |
| `1M_20_HPO_xrfm_mu.pkl` | 2851696442 | `f5df620e38b5def535c86ead52e0e56c` |
| `1M_20_HPO_xrfm_log_kappa.pkl` | 2850701229 | `52cf0523795cfe71dc6f4606a2858926` |
