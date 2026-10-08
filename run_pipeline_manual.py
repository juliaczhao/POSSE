"""Run the POSSE pipeline end to end.

Reads pipeline_config.json (or a config path passed as the first argument) and runs
Steps 1-10 in order, writing every output under `output_dir`. Each step reads the
previous step's output, so commenting a step out runs only the remaining steps.

    python run_pipeline_manual.py [config.json]
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("ANNDATA_ALLOW_WRITE_NULLABLE_STRINGS", "1")

from pipeline_steps.anndata_compat import enable_anndata_write_compatibility

enable_anndata_write_compatibility()

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = os.path.dirname(os.path.abspath(__file__))

import pipeline_steps.clip_outliers as clip_outliers
import pipeline_steps.clique_based_program_finding as clique_program_finding
import pipeline_steps.compute_ids as compute_ids
import pipeline_steps.compute_program_activity as program_activity
import pipeline_steps.generate_html_report as generate_html_report
import pipeline_steps.generate_report_plots as report_plots
import pipeline_steps.llm_program_summaries as llm_program_summaries
import pipeline_steps.preprocess_data as preprocess_data
import pipeline_steps.runtime_metadata as runtime_metadata
import pipeline_steps.select_cell_types as select_cell_types
import pipeline_steps.statistical_ids_significance_testing as ids_significance_testing
from pipeline_steps import __version__


def load_args_from_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    args = {}

    for section, section_dict in config.items():
        if isinstance(section_dict, dict):
            args[section] = section_dict.copy()
        else:
            args[section] = section_dict

    spf_config = args.setdefault("statistical_program_finding", {})
    model_dir = os.path.expanduser(spf_config.get("model_dir") or "beta_fit_model")
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(REPOSITORY_ROOT, model_dir)
    spf_config["model_dir"] = os.path.abspath(model_dir)

    return args


def run_preprocessing(args):
    raw_adata_path = args["global"]["raw_data_path"]
    preprocessed_data_path = args["global"]["output_prefix"]
    output_dir = args["global"]["output_dir"]
    umi_threshold = args["preprocess"]["umi_threshold"]
    nonzero_gene_percentage = args["preprocess"]["nonzero_gene_percentage"]
    mt_ratio_percentage = args["preprocess"]["mt_ratio_percentage"]

    preprocess_data.clean_raw_data(
        raw_adata_path,
        umi_threshold=umi_threshold,
        nonzero_gene_percentage=nonzero_gene_percentage,
        mt_ratio_percentage=mt_ratio_percentage,
        output_path=os.path.join(output_dir, preprocessed_data_path),
    )


def run_cell_type_selector(args, input_dir=None):
    cell_types = args["global"]["cell_types"]
    output_dir = args["global"]["output_dir"]
    cell_type_label = args["global"].get("cell_type_label", "cell_type")
    working_dir_name_override = args["global"].get("working_dir_name")
    if input_dir is None:
        preprocessed_data_path = args["global"]["output_prefix"]
        input_dir = os.path.join(output_dir, preprocessed_data_path)
    working_dir_name = select_cell_types.extract_cell_types(
        input_dir,
        cell_types,
        output_dir=output_dir,
        cell_type_label=cell_type_label,
        working_dir_name=working_dir_name_override,
    )
    return working_dir_name


def run_clipping_values(args, working_dir_name):
    COOCCUR_THRESHOLD = args["global"]["cooccurrence_threshold"]
    TOP_K_PERCENTAGE = args["outlier_clip"]["top_k_percentage"]

    anndata_dir = os.path.join(working_dir_name, "anndata_files")
    supp_material_dir = os.path.join(working_dir_name, "supp_material")

    clip_outliers.clip_outliers(
        anndata_dir, supp_material_dir, TOP_K_PERCENTAGE, COOCCUR_THRESHOLD
    )


def run_ids_computation(args, working_dir_name):
    anndata_dir = os.path.join(working_dir_name, "anndata_files")
    output_dir = os.path.join(working_dir_name, "ids_cooccur")
    clip_vals_dir = os.path.join(working_dir_name, "supp_material")

    os.makedirs(f"{output_dir}", exist_ok=True)

    compute_ids.compute_ids_and_cooccur(
        anndata_dir, output_dir, clip_vals_dir, working_dir=working_dir_name
    )


def run_statistical_ids_significance_testing(args, working_dir_name):
    COOCCUR_THRESHOLD = args["global"]["cooccurrence_threshold"]
    spf_args = args["statistical_program_finding"]
    MSE_THRESHOLD = spf_args["mse_threshold"]
    P_VAL = spf_args["p_val"]
    MAX_SIGNIFICANT_NEIGHBORS = spf_args.get("max_significant_neighbors", 1000)
    model_dir = spf_args.get("model_dir")
    ids_significance_testing.test_ids_significance(
        working_dir_name,
        COOCCUR_THRESHOLD=COOCCUR_THRESHOLD,
        MSE_THRESHOLD=MSE_THRESHOLD,
        P_VAL=P_VAL,
        MAX_SIGNIFICANT_NEIGHBORS=MAX_SIGNIFICANT_NEIGHBORS,
        model_dir=model_dir,
    )


def run_clique_program_finding(args, working_dir_name):
    args_dict = args["program_finding"]
    clique_program_finding.run_program_finder(
        working_dir_name,
        gamma=args_dict["gamma"],
        min_clique_size=args_dict["min_clique_size"],
        seed=args_dict["seed"],
    )


def run_program_activity(args, working_dir_name):
    args_dict = args["program_activity"]
    program_activity.compute_program_activity(
        working_dir_name,
        loading_type=args_dict["loading_type"],
        program_vector_scaling=args_dict["program_vector_scaling"],
        weighting=args_dict.get("weighting", "positive"),
    )


def run_report_plots(args, working_dir_name):
    leiden_resolution = args["report_plots"]["leiden_resolution"]
    cell_type_column = args["global"].get("cell_type_label", "cell_type")
    report_plots.generate_all_plots(
        working_dir_name,
        leiden_resolution,
        cell_type_column=cell_type_column,
    )


def run_llm_summaries(args, working_dir_name):
    llm_config = args["llm_program_summaries"]
    api_key = str(llm_config.get("api_key") or "").strip()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    provide_celltype = llm_config.get("provide_celltype", True)
    model = llm_config.get("model", "gpt-5.6-sol")
    reasoning_effort = llm_config.get("reasoning_effort", "high")
    cell_type_column = args["global"].get("cell_type_label", "cell_type")
    if api_key:
        llm_program_summaries.generate_llm_program_summaries(
            api_key,
            working_dir_name,
            provide_celltype=provide_celltype,
            cell_type_column=cell_type_column,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    else:
        llm_program_summaries.write_empty_program_summaries(working_dir_name)


def run_report_generation(args, working_dir_name):
    cell_type_column = args["global"].get("cell_type_label", "cell_type")
    generate_html_report.generate_html_report(
        working_dir_name,
        cell_type_column=cell_type_column,
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./pipeline_config.json"

    args = load_args_from_config(config_path)

    # Log to the console and to output_dir/pipeline.log. ``force=True`` ensures
    # consistent handlers when an imported dependency configures root logging.
    output_dir = args["global"]["output_dir"]
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
    run_started_at = datetime.now(timezone.utc)

    import torch

    git_metadata = runtime_metadata.git_metadata(REPOSITORY_ROOT)
    logger.info(
        "Git: commit=%s, branch=%s, dirty=%s",
        git_metadata["commit"],
        git_metadata["branch"],
        git_metadata["dirty"],
    )
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s", runtime_metadata.platform_description())
    logger.info("Torch: %s (CUDA %s)", torch.__version__, torch.version.cuda)
    logger.info("CUDA_HOME=%s", os.environ.get("CUDA_HOME", "<not found>"))
    logger.info(
        "CUDA toolkit: %s",
        runtime_metadata.cuda_toolkit_version() or "<unavailable>",
    )
    logger.info(
        "NVIDIA driver version(s): %s",
        runtime_metadata.nvidia_driver_versions() or "<unavailable>",
    )
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset; auto-detecting>"),
    )
    if torch.cuda.is_available():
        logger.info("Detected %d CUDA device(s)", torch.cuda.device_count())
        for gpu_id in range(torch.cuda.device_count()):
            logger.info(
                "CUDA device %d: %s", gpu_id, torch.cuda.get_device_name(gpu_id)
            )
    else:
        logger.info("Detected 0 CUDA devices")

    g = args["global"]
    spf_cfg = args.get("statistical_program_finding", {})
    pf_cfg = args.get("program_finding", {})
    logger.info(
        "Run configuration: input=%s, output=%s, filtering=%s, cooccurrence>=%s, "
        "MSE<%s, p<%s, quasi-clique gamma=%s, min_size=%s, seed=%s, model_dir=%s",
        g.get("raw_data_path"),
        output_dir,
        args.get("preprocess", {}).get("filter_raw_data", True),
        g.get("cooccurrence_threshold"),
        spf_cfg.get("mse_threshold"),
        spf_cfg.get("p_val"),
        pf_cfg.get("gamma"),
        pf_cfg.get("min_clique_size"),
        pf_cfg.get("seed"),
        spf_cfg.get("model_dir"),
    )

    # Full configuration is kept at DEBUG for reproducibility, with the API key redacted.
    redacted = {
        section: (
            {
                k: ("<redacted>" if "api_key" in k.lower() else v)
                for k, v in values.items()
            }
            if isinstance(values, dict)
            else values
        )
        for section, values in args.items()
    }
    logger.debug("Full configuration: %s", redacted)

    elapsed_times = {}

    # Step 1: preprocessing. Filter the raw counts by UMI count, expressed genes, and
    # mitochondrial fraction. Skipped when filter_raw_data=false (input already filtered).
    input_dir_override = None
    if args["preprocess"].get("filter_raw_data", True):
        logger.info("Step 1: preprocessing data")
        start_time = time.time()
        run_preprocessing(args)
        elapsed_times["preprocessing"] = time.time() - start_time
        logger.info("Step 1 completed in %.2f seconds", elapsed_times["preprocessing"])
    else:
        logger.info("Step 1: preprocessing skipped (filter_raw_data=false)")
        input_dir_override = args["global"]["raw_data_path"]

    # Step 2: cell selection. Subset to the requested cell types and rewrite the cells as
    # fixed-size chunks under output_dir, which every later step streams over.
    logger.info("Step 2: selecting cell types")
    start_time = time.time()
    working_dir_name = run_cell_type_selector(args, input_dir=input_dir_override)
    elapsed_times["cell_type_selection"] = time.time() - start_time
    logger.info("Working directory: %s", working_dir_name)
    logger.info(
        "Step 2 completed in %.2f seconds", elapsed_times["cell_type_selection"]
    )

    # Step 3: outlier clipping. Set a per-gene expression ceiling so a few extreme
    # cells cannot dominate the gene-gene dependence estimates.
    logger.info("Step 3: computing outlier clipping values")
    start_time = time.time()
    run_clipping_values(args, working_dir_name)
    elapsed_times["clipping_values"] = time.time() - start_time
    logger.info("Step 3 completed in %.2f seconds", elapsed_times["clipping_values"])

    # Step 4: dependence. Compute the gene-by-gene IDS, correlation, and cooccurrence
    # matrices. IDS is tested for significance next, correlation gives each pair its sign,
    # and cooccurrence decides which pairs carry enough evidence to test.
    logger.info(
        "Step 4: computing gene-gene dependencies (IDS, correlation, cooccurrence)"
    )
    start_time = time.time()
    run_ids_computation(args, working_dir_name)
    elapsed_times["ids_computation"] = time.time() - start_time
    logger.info("Step 4 completed in %.2f seconds", elapsed_times["ids_computation"])

    # Step 5: significance. Fit each pair's null beta distribution, retain the
    # significant edges, and write the symmetric graph used for program finding.
    logger.info("Step 5: testing IDS significance and building the adjacency graph")
    start_time = time.time()
    run_statistical_ids_significance_testing(args, working_dir_name)
    elapsed_times["statistical_ids_significance_testing"] = time.time() - start_time
    logger.info(
        "Step 5 completed in %.2f seconds",
        elapsed_times["statistical_ids_significance_testing"],
    )

    # Release allocations from Step 5 before GPU Louvain in Step 6.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        logger.debug(
            "GPU memory cleared: %.2f GB allocated, %.2f GB reserved",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )

    # Step 6: program finding. Seed positive cliques, then build and review signed
    # positive quasi-clique programs.
    logger.info("Step 6: finding clique-based gene programs")
    start_time = time.time()
    run_clique_program_finding(args, working_dir_name)
    elapsed_times["clique_program_finding"] = time.time() - start_time
    logger.info(
        "Step 6 completed in %.2f seconds", elapsed_times["clique_program_finding"]
    )

    # Step 7: program activity. Score every cell against every program two ways -- wePAS
    # (loading-weighted projection) and sePAS (co-expression) -- each with a per-program
    # Otsu on/off call.
    logger.info("Step 7: computing program activity (wePAS and sePAS)")
    start_time = time.time()
    run_program_activity(args, working_dir_name)
    elapsed_times["program_activity"] = time.time() - start_time
    logger.info("Step 7 completed in %.2f seconds", elapsed_times["program_activity"])

    # Step 8: report figures. Build the UMAP, Leiden clustering, and per-program activity
    # plots from the scaled wePAS activities.
    logger.info("Step 8: generating report plots")
    start_time = time.time()
    run_report_plots(args, working_dir_name)
    elapsed_times["supp_material_and_plots"] = time.time() - start_time
    logger.info(
        "Step 8 completed in %.2f seconds", elapsed_times["supp_material_and_plots"]
    )

    # Step 9: program summaries. Name and describe each program with a language model.
    # Runs only if an API key is configured; without one, an empty
    # summaries file (gene lists only) is written so Step 10 still runs.
    logger.info("Step 9: generating program summaries")
    start_time = time.time()
    run_llm_summaries(args, working_dir_name)
    elapsed_times["llm_program_summaries"] = time.time() - start_time
    logger.info(
        "Step 9 completed in %.2f seconds", elapsed_times["llm_program_summaries"]
    )

    # Step 10: report. Assemble the programs, summaries, and figures into HTML, with or
    # without the Step 9 annotations.
    logger.info("Step 10: generating the program report")
    start_time = time.time()
    run_report_generation(args, working_dir_name)
    elapsed_times["report_generation"] = time.time() - start_time
    logger.info("Step 10 completed in %.2f seconds", elapsed_times["report_generation"])

    total_time = sum(elapsed_times.values())
    logger.info("Elapsed time per step:")
    for key, seconds in elapsed_times.items():
        logger.info("  %-44s %8.2f s", key, seconds)
    logger.info("Total elapsed time: %.2f seconds", total_time)
    runtime_metadata.write_runtime_summary(
        output_dir,
        working_dir_name,
        args,
        REPOSITORY_ROOT,
        run_started_at,
        datetime.now(timezone.utc),
        elapsed_times,
    )


if __name__ == "__main__":
    main()
