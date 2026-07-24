"""Run the POSSE pipeline end to end.

Reads pipeline_config.json (or a config path passed as the first argument) and runs
Steps 1-10 in order, writing every output under `output_dir`. Each step reads the
previous step's output, so commenting a step out runs only the remaining steps.

    python run_pipeline_manual.py [config.json]
"""

import json
import logging
import os
import time

os.environ.setdefault("ANNDATA_ALLOW_WRITE_NULLABLE_STRINGS", "1")

# kermac (used by xRFM in Step 4) JIT-compiles CUDA kernels at runtime, so point
# CUDA_HOME at a CUDA toolkit if it is not already set.
if not os.environ.get("CUDA_HOME"):
    for _cuda_dir in ("/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-12.8"):
        if os.path.isdir(_cuda_dir):
            os.environ["CUDA_HOME"] = _cuda_dir
            break

# Cast string obs/var indexes and columns to object dtype before writing, so anndata
# serializes them to .h5ad consistently across pandas versions.
import pandas as pd
try:
    pd.options.future.infer_string = False
except Exception:
    pass

import anndata as ad


def _cast_string_to_object(df):
    if df is None:
        return df
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)
    if pd.api.types.is_string_dtype(df.index) and df.index.dtype != object:
        df.index = df.index.astype(object)
    return df


_orig_adata_write_h5ad = ad.AnnData.write_h5ad
def _patched_write_h5ad(self, *args, **kwargs):
    _cast_string_to_object(self.obs)
    _cast_string_to_object(self.var)
    return _orig_adata_write_h5ad(self, *args, **kwargs)
ad.AnnData.write_h5ad = _patched_write_h5ad
ad.AnnData.write = _patched_write_h5ad

logger = logging.getLogger(__name__)

from pipeline_steps import __version__
import pipeline_steps.preprocess as preprocess
import pipeline_steps.select_cell_types as select_cell_types
import pipeline_steps.clip_outliers as clip_outliers
import pipeline_steps.compute_ids as compute_ids
import pipeline_steps.statistical_program_finding as statistical_program_finding
import pipeline_steps.postprocess_adjacency_matrix as postprocess_adjacency_matrix
import pipeline_steps.program_finding as clique_program_finding
import pipeline_steps.llm_program_summaries as llm_program_summaries
import pipeline_steps.generate_program_report as generate_program_report
import pipeline_steps.report_plots as report_plots
import pipeline_steps.program_activity as program_activity

def load_args_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    args = {}

    for section, section_dict in config.items():
        if isinstance(section_dict, dict):
            args[section] = section_dict.copy()
        else:
            args[section] = section_dict

    return args

def run_preprocessor(args):
    raw_adata_path = args['global']['raw_data_path']
    preprocessed_data_path = args['global']['output_prefix']
    output_dir = args['global']['output_dir']
    umi_threshold = args['preprocess']['umi_threshold']
    nonzero_gene_percentage = args['preprocess']['nonzero_gene_percentage']
    mt_ratio_percentage = args['preprocess']['mt_ratio_percentage']

    preprocess.clean_raw_data(raw_adata_path, umi_threshold=umi_threshold, 
                              nonzero_gene_percentage=nonzero_gene_percentage, 
                              mt_ratio_percentage=mt_ratio_percentage, 
                              output_path=os.path.join(output_dir, preprocessed_data_path))


def run_cell_type_selector(args, input_dir=None):
    cell_types = args['global']['cell_types']
    output_dir = args['global']['output_dir']
    cell_type_label = args['global'].get('cell_type_label', 'cell_type')
    working_dir_name_override = args['global'].get('working_dir_name')
    if input_dir is None:
        preprocessed_data_path = args['global']['output_prefix']
        input_dir = os.path.join(output_dir, preprocessed_data_path)
    working_dir_name = select_cell_types.extract_cell_types(
        input_dir, cell_types, output_dir=output_dir,
        cell_type_label=cell_type_label,
        working_dir_name=working_dir_name_override)
    return working_dir_name

def compute_clipping_vals(args, working_dir_name):
    COOCCUR_THRESHOLD = args['global']['cooccurrence_threshold']
    TOP_K_PERCENTAGE = args['outlier_clip']['top_k_percentage']

    anndata_dir = os.path.join(working_dir_name, 'anndata_files')
    supp_material_dir = os.path.join(working_dir_name, 'supp_material')

    clip_outliers.clip_outliers(anndata_dir, supp_material_dir, TOP_K_PERCENTAGE, COOCCUR_THRESHOLD)

def run_ids_computation(args, working_dir_name):

    anndata_dir = os.path.join(working_dir_name, 'anndata_files')
    output_dir = os.path.join(working_dir_name, 'ids_cooccur')
    clip_vals_dir = os.path.join(working_dir_name, 'supp_material')

    os.makedirs(f'{output_dir}', exist_ok=True)

    compute_ids.compute_ids_and_cooccur(anndata_dir, output_dir, clip_vals_dir, working_dir=working_dir_name)


def run_statistically_significant_gene_dependencies(args, working_dir_name):
    COOCCUR_THRESHOLD = args['global']['cooccurrence_threshold']
    spf_args = args['statistical_program_finding']
    MSE_THRESHOLD = spf_args['mse_threshold']
    P_VAL = spf_args['p_val']
    MAX_SIGNIFICANT_NEIGHBORS = spf_args.get('max_significant_neighbors', 1000)
    mu_filename = spf_args.get('model_mu_filename')
    logkappa_filename = spf_args.get('model_logkappa_filename')
    statistical_program_finding.statistical_program_finding(working_dir_name,
                                                            COOCCUR_THRESHOLD=COOCCUR_THRESHOLD,
                                                            MSE_THRESHOLD=MSE_THRESHOLD,
                                                            P_VAL=P_VAL,
                                                            MAX_SIGNIFICANT_NEIGHBORS=MAX_SIGNIFICANT_NEIGHBORS,
                                                            mu_filename=mu_filename,
                                                            logkappa_filename=logkappa_filename)


def run_clean_and_symmetrize_adj_matrix(arg, working_dir_name):
    postprocess_adjacency_matrix.clean_and_symmetrize(working_dir_name)

def run_clique_program_finding(args, working_dir_name):
    args_dict = args['program_finding']
    clique_program_finding.run_program_finder(
        working_dir_name,
        MIN_CLIQUE_SIZE=args_dict['min_clique_size'],
        GROWTH_THRESHOLD=args_dict['growth_threshold'],
        CLUSTER_START_RESOLUTION=args_dict['cluster_start_resolution'],
        MIN_PROGRAM_SIZE=args_dict['min_program_size'],
        PROGRAM_PATH_DIR=args_dict['program_path_dir'],
        PROGRAM_PATH_NAME=args_dict['program_path_name'],
        CONTAINMENT_MERGE_THRESHOLD=args_dict['containment_merge_threshold'],
        LOUVAIN_BACKEND=args_dict.get('louvain_backend', 'cpu'),
    )


def run_program_activity(args, working_dir_name):
    args_dict = args['program_activity']
    args_dict_program_finding = args['program_finding']
    program_activity.compute_program_activity(working_dir_name,
                                              program_path_dir=args_dict_program_finding['program_path_dir'],
                                              program_path_name=args_dict_program_finding['program_path_name'],
                                              loading_type=args_dict['loading_type'],
                                              program_vector_scaling=args_dict['program_vector_scaling'],
                                              weighting=args_dict.get('weighting', 'positive'),
                                              )


def run_report_plots(args, working_dir_name):
    leiden_resolution = args['report_plots']['leiden_resolution']
    report_plots.generate_all_plots(working_dir_name, leiden_resolution)

def _has_api_key(api_key):
    """True only for a real key, so the placeholder in the shipped config counts as absent."""
    return bool(api_key) and str(api_key).strip().startswith("sk-")


def run_llm_summaries(args, working_dir_name):
    api_key = args['llm_program_summaries']['api_key']
    provide_celltype = args['llm_program_summaries'].get('provide_celltype', False)
    if _has_api_key(api_key):
        llm_program_summaries.generate_llm_program_summaries(
            api_key, working_dir_name, provide_celltype=provide_celltype
        )
    else:
        llm_program_summaries.write_empty_program_summaries(working_dir_name)

def run_report_generation(args, working_dir_name):
    generate_program_report.generate_report(working_dir_name)

def main():
    import sys as _sys
    config_path = _sys.argv[1] if len(_sys.argv) > 1 else './pipeline_config.json'

    args = load_args_from_config(config_path)

    # Log to the console and to output_dir/pipeline.log. force=True overrides the root
    # handler rapids_singlecell attaches at import, which would otherwise silently win.
    output_dir = args['global']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "pipeline.log")),
        ],
    )

    # Silence third-party INFO noise (anndata "storing as categorical", matplotlib units).
    logging.getLogger("anndata").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger.info("POSSE %s", __version__)

    g = args['global']
    spf_cfg = args.get('statistical_program_finding', {})
    pf_cfg = args.get('program_finding', {})
    logger.info(
        "Run configuration: input=%s, output=%s, filtering=%s, cooccurrence>=%s, "
        "MSE<%s, p<%s, louvain=%s",
        g.get('raw_data_path'), output_dir,
        args.get('preprocess', {}).get('filter_raw_data', True),
        g.get('cooccurrence_threshold'), spf_cfg.get('mse_threshold'),
        spf_cfg.get('p_val'), pf_cfg.get('louvain_backend', 'cpu'),
    )

    # Full configuration is kept at DEBUG for reproducibility, with the API key redacted.
    redacted = {
        section: ({k: ("<redacted>" if "api_key" in k.lower() else v) for k, v in values.items()}
                  if isinstance(values, dict) else values)
        for section, values in args.items()
    }
    logger.debug("Full configuration: %s", redacted)

    elapsed_times = {}

    # Step 1: preprocessing. Filter the raw counts by UMI count, expressed genes, and
    # mitochondrial fraction. Skipped when filter_raw_data=false (input already filtered).
    input_dir_override = None
    if args['preprocess'].get('filter_raw_data', True):
        logger.info("Step 1: preprocessing data")
        start_time = time.time()
        run_preprocessor(args)
        elapsed_times['preprocessing'] = time.time() - start_time
        logger.info("Step 1 completed in %.2f seconds", elapsed_times['preprocessing'])
    else:
        logger.info("Step 1: preprocessing skipped (filter_raw_data=false)")
        input_dir_override = args['global']['raw_data_path']

    # Step 2: cell selection. Subset to the requested cell types and rewrite the cells as
    # fixed-size chunks under output_dir, which every later step streams over.
    logger.info("Step 2: selecting cell types")
    start_time = time.time()
    working_dir_name = run_cell_type_selector(args, input_dir=input_dir_override)
    elapsed_times['cell_type_selection'] = time.time() - start_time
    logger.info("Working directory: %s", working_dir_name)
    logger.info("Step 2 completed in %.2f seconds", elapsed_times['cell_type_selection'])


    # Step 2.5: outlier clipping. Set a per-gene expression ceiling so a few extreme
    # cells cannot dominate the gene-gene dependence estimates.
    logger.info("Step 2.5: computing outlier clipping values")
    start_time = time.time()
    compute_clipping_vals(args, working_dir_name)
    elapsed_times['clipping_values'] = time.time() - start_time
    logger.info("Step 2.5 completed in %.2f seconds", elapsed_times['clipping_values'])

    # Step 3: dependence. Compute the gene-by-gene IDS, correlation, and cooccurrence
    # matrices. IDS is tested for significance next, correlation gives each pair its sign,
    # and cooccurrence decides which pairs carry enough evidence to test.
    logger.info("Step 3: computing gene-gene dependencies (IDS, correlation, cooccurrence)")
    start_time = time.time()
    run_ids_computation(args, working_dir_name)
    elapsed_times['ids_computation'] = time.time() - start_time
    logger.info("Step 3 completed in %.2f seconds", elapsed_times['ids_computation'])

    # Step 4: significance. Fit each pair's null beta distribution with the xRFM models,
    # convert to p-values, and keep the pairs surviving Benjamini-Hochberg correction.
    # The surviving pairs form the edges of the gene graph.
    logger.info("Step 4: identifying statistically significant gene dependencies")
    start_time = time.time()
    run_statistically_significant_gene_dependencies(args, working_dir_name)
    elapsed_times['statistically_significant_gene_dependencies'] = time.time() - start_time
    logger.info("Step 4 completed in %.2f seconds", elapsed_times['statistically_significant_gene_dependencies'])

    # Step 5: symmetrize. Significance is tested per direction, so make the edge matrix
    # undirected, as program finding requires.
    logger.info("Step 5: symmetrizing the adjacency matrix")
    start_time = time.time()
    run_clean_and_symmetrize_adj_matrix(args, working_dir_name)
    elapsed_times['clean_and_symmetrize_adj_matrix'] = time.time() - start_time
    logger.info("Step 5 completed in %.2f seconds", elapsed_times['clean_and_symmetrize_adj_matrix'])

    # Clear GPU memory before Step 6 (Step 6 is CPU-only)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc; gc.collect()
        logger.debug("GPU memory cleared: %.2f GB allocated, %.2f GB reserved",
                     torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

    # Step 6: program finding. Peel cliques from the gene graph, then grow and merge
    # them into the final gene programs.
    logger.info("Step 6: finding clique-based gene programs")
    start_time = time.time()
    run_clique_program_finding(args, working_dir_name)
    elapsed_times['clique_program_finding'] = time.time() - start_time
    logger.info("Step 6 completed in %.2f seconds", elapsed_times['clique_program_finding'])
    
    # Step 7: program activity. Score every cell against every program two ways -- wePAS
    # (loading-weighted projection) and sePAS (co-expression) -- each with a per-program
    # Otsu on/off call.
    logger.info("Step 7: computing program activity (wePAS and sePAS)")
    start_time = time.time()
    run_program_activity(args, working_dir_name)
    elapsed_times['program_activity'] = time.time() - start_time
    logger.info("Step 7 completed in %.2f seconds", elapsed_times['program_activity'])

    # Step 8: report figures. Build the UMAP, Leiden clustering, and per-program activity
    # plots from the scaled wePAS activities.
    logger.info("Step 8: generating report plots")
    start_time = time.time()
    run_report_plots(args, working_dir_name)
    elapsed_times['supp_material_and_plots'] = time.time() - start_time
    logger.info("Step 8 completed in %.2f seconds", elapsed_times['supp_material_and_plots'])

    # Step 9: program summaries. Name and describe each program with a language model.
    # Runs only if a real API key is set in pipeline_config.json; without one, an empty
    # summaries file (gene lists only) is written so Step 10 still runs.
    logger.info("Step 9: generating program summaries")
    start_time = time.time()
    run_llm_summaries(args, working_dir_name)
    elapsed_times['llm_program_summaries'] = time.time() - start_time
    logger.info("Step 9 completed in %.2f seconds", elapsed_times['llm_program_summaries'])

    # Step 10: report. Assemble the programs, summaries, and figures into a .docx, with or
    # without the Step 9 annotations.
    logger.info("Step 10: generating the program report")
    start_time = time.time()
    run_report_generation(args, working_dir_name)
    elapsed_times['report_generation'] = time.time() - start_time
    logger.info("Step 10 completed in %.2f seconds", elapsed_times['report_generation'])

    total_time = sum(elapsed_times.values())
    logger.info("Elapsed time per step:")
    for key, seconds in elapsed_times.items():
        logger.info("  %-44s %8.2f s", key, seconds)
    logger.info("Total elapsed time: %.2f seconds", total_time)

if __name__ == "__main__":
    main()
