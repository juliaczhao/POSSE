"""Program summaries from a language model.

Sends each program's gene list to a language model and records the returned
title, summary and any genes flagged as inconsistent. Requires an API key in
pipeline_config.json.

Reads
-----
    <working_dir>/clique_based_programs/programs_with_loadings.json
    prompt_templates/

Writes
------
    <working_dir>/llm_summaries/program_summaries_with_genes.csv
    <working_dir>/llm_summaries/program_summaries_no_genes.csv
"""

import logging
import re
import csv
from openai import OpenAI
import json
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


def _parse_llm_response(content: str):
    """Parse LLM output containing Title:, Inconsistent Genes:, and Summary: sections.

    Returns (title, inconsistent_genes_str, summary). Falls back gracefully if
    any section is missing.
    """
    title = ""
    inconsistent = ""
    summary = ""

    title_match = re.search(r"^\s*Title:\s*(.+?)\s*$", content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    inc_match = re.search(r"^\s*Inconsistent Genes:\s*(.+?)\s*$", content, re.MULTILINE)
    if inc_match:
        inconsistent = inc_match.group(1).strip()

    if "Summary:" in content:
        summary = content.split("Summary:", 1)[1].strip()
    else:
        summary = content.strip()

    return title, inconsistent, summary


def generate_llm_program_summaries(api_key, working_dir, provide_celltype=False):


    """Name and describe each program from its gene list using a language model.

    Reads the programs and their activity, prompts with the templates in
    prompt_templates/, and writes the CSV the report step reads for its titles
    and descriptions. Optional; the pipeline runs without it.
    """
    llm_summary_csv_dir = os.path.join(working_dir, 'llm_summaries')
    if not os.path.exists(llm_summary_csv_dir):
        os.makedirs(llm_summary_csv_dir)

    llm_with_genes_path = os.path.join(llm_summary_csv_dir, 'program_summaries_with_genes.csv')
    llm_no_genes_path = os.path.join(llm_summary_csv_dir, 'program_summaries_no_genes.csv')

    programs_dir = os.path.join(working_dir, 'clique_based_programs')
    programs_json = os.path.join(programs_dir, 'programs_with_loadings.json')

    cell_types = []
    if provide_celltype:
        import anndata
        program_activity_path = os.path.join(working_dir, 'program_activity', 'wePAS.h5ad')
        adata = anndata.read_h5ad(program_activity_path)
        if 'cell_type' not in adata.obs.columns:
            raise ValueError(
                f"provide_celltype=True but {program_activity_path} has no obs['cell_type']"
            )
        cell_types = sorted(adata.obs['cell_type'].astype(str).unique().tolist())
        logger.info(f"Using cell types from wePAS.h5ad: {cell_types}")
        with open('./prompt_templates/program_description.txt', 'r') as f:
            prompt_template = f.read()
    else:
        with open('./prompt_templates/program_description_no_cell_type.txt', 'r') as f:
            prompt_template = f.read()

    client = OpenAI(
        api_key=api_key
    )

    with open(programs_json, 'r') as f:
        programs_data = json.load(f)

    programs = []
    for program_info in programs_data.values():
        loadings = program_info.get("loadings", {})
        positively_loaded_genes = [gene for gene, loading in loadings.items() if loading > 0]
        negatively_loaded_genes = [gene for gene, loading in loadings.items() if loading < 0]
        programs.append({
            "positively_loaded_genes": positively_loaded_genes,
            "negatively_loaded_genes": negatively_loaded_genes
        })

    full_data = []
    summary_data = []
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_single_program(idx, program, prompt_template, cell_types):
        """Process a single program and return the results"""
        if len(cell_types) == 0:
            prompt = prompt_template.format(
                positively_loaded_genes=program["positively_loaded_genes"],
                negatively_loaded_genes=program["negatively_loaded_genes"]
            )
        else:
            prompt = prompt_template.format(
                cell_types=cell_types,
                positively_loaded_genes=program["positively_loaded_genes"],
                negatively_loaded_genes=program["negatively_loaded_genes"]
            )
        
        output = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model='o3-2025-04-16'
        )
    
        content = output.choices[0].message.content
        title, inconsistent_genes_str, summary = _parse_llm_response(content)

        all_program_genes = set(program["positively_loaded_genes"]) | set(program["negatively_loaded_genes"])
        inconsistent_list = []
        if inconsistent_genes_str and inconsistent_genes_str.upper() != "NONE":
            for g in inconsistent_genes_str.split(","):
                g = g.strip()
                if g and g in all_program_genes:
                    inconsistent_list.append(g)
        num_inconsistent = len(inconsistent_list)
        inconsistent_joined = ", ".join(inconsistent_list)

        genes_str = ", ".join(sorted(all_program_genes))
        num_genes = len(all_program_genes)

        return idx, genes_str, num_genes, title, inconsistent_joined, num_inconsistent, summary
    
    results = [None] * len(programs)
    
    batch_size = 3
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in tqdm(range(0, len(programs), batch_size), desc="Processing program batches"):
            batch_programs = programs[i:i+batch_size]
            batch_indices = list(range(i+1, i+len(batch_programs)+1))
            
            logger.info(f"Processing batch: programs {batch_indices[0]}-{batch_indices[-1]}")
            
            futures = []
            for idx, program in zip(batch_indices, batch_programs):
                future = executor.submit(process_single_program, idx, program, prompt_template, cell_types)
                futures.append(future)
            
            for future in futures:
                idx, genes_str, num_genes, title, inconsistent_joined, num_inconsistent, summary = future.result()
                logger.info(f"Program {idx} title: {title}")

                results[idx-1] = (idx, genes_str, num_genes, title, inconsistent_joined, num_inconsistent, summary)

    for result in results:
        if result is not None:
            idx, genes_str, num_genes, title, inconsistent_joined, num_inconsistent, summary = result
            full_data.append([f"Program {idx-1}", title, genes_str, num_genes, inconsistent_joined, num_inconsistent, summary])
            summary_data.append([f"Program {idx-1}", title, num_genes, num_inconsistent, summary])


    with open(llm_with_genes_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter='$', quoting=csv.QUOTE_ALL)
        writer.writerow(["Program", "Title", "Genes", "Number of Genes", "Inconsistent Genes", "Number of Inconsistent Genes", "Summary"])
        writer.writerows(full_data)

    with open(llm_no_genes_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter='$', quoting=csv.QUOTE_ALL)
        writer.writerow(["Program", "Title", "Number of Genes", "Number of Inconsistent Genes", "Summary"])
        writer.writerows(summary_data)
        
    logger.info(f"Generated two CSV files:")
    logger.info(f"- program_summaries_with_genes.csv (includes gene lists)")
    logger.info(f"- program_summaries_no_genes.csv (summary table only)")


_WITH_GENES_HEADER = ["Program", "Title", "Genes", "Number of Genes",
                      "Inconsistent Genes", "Number of Inconsistent Genes", "Summary"]
_NO_GENES_HEADER = ["Program", "Title", "Number of Genes", "Number of Inconsistent Genes", "Summary"]


def write_empty_program_summaries(working_dir):
    """Write the summaries CSVs with gene lists but no language-model annotations.

    Used when no API key is configured, so the report step can still run: the report
    then contains the gene lists and plots but no titles or descriptions. To annotate
    afterwards, fill the Title and Summary columns (by hand or with any LLM) and re-run
    the report step.
    """
    out_dir = os.path.join(working_dir, 'llm_summaries')
    os.makedirs(out_dir, exist_ok=True)
    programs_json = os.path.join(working_dir, 'clique_based_programs', 'programs_with_loadings.json')
    with open(programs_json) as f:
        programs_data = json.load(f)

    full_data, summary_data = [], []
    for pid in sorted(programs_data, key=int):
        genes = programs_data[pid].get("program", [])
        full_data.append([f"Program {pid}", "", ", ".join(genes), len(genes), "", 0, ""])
        summary_data.append([f"Program {pid}", "", len(genes), 0, ""])

    with open(os.path.join(out_dir, 'program_summaries_with_genes.csv'), "w",
              encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter='$', quoting=csv.QUOTE_ALL)
        writer.writerow(_WITH_GENES_HEADER)
        writer.writerows(full_data)
    with open(os.path.join(out_dir, 'program_summaries_no_genes.csv'), "w",
              encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter='$', quoting=csv.QUOTE_ALL)
        writer.writerow(_NO_GENES_HEADER)
        writer.writerows(summary_data)
    logger.info("No API key configured; wrote empty program summaries (gene lists only) to %s", out_dir)

