"""Program summaries from a language model.

Sends each program's gene list to a language model and records the returned
title, summary and any genes flagged as inconsistent. The pipeline passes the
configured credential into this module.

Reads
-----
    <working_dir>/clique_based_programs/programs_with_loadings.json
    prompt_templates/

Writes
------
    <working_dir>/llm_summaries/program_summaries_with_genes.csv
    <working_dir>/llm_summaries/program_summaries_no_genes.csv
"""

import csv
import json
import logging
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _program_annotation_model():
    """Create the optional Pydantic schema used for structured LLM output."""
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "LLM annotation requires Pydantic, which is installed with the OpenAI client. "
            "Install the optional client with 'uv pip install openai==2.24.0'."
        ) from exc

    class ProgramAnnotation(BaseModel):
        title: str = Field(
            description="Concise cell type or state title of at most six words"
        )
        inconsistent_genes: list[str] = Field(
            description="Gene symbols from the supplied lists that conflict with the interpretation"
        )
        interpretation: str = Field(
            description="Concise top-level biological interpretation without a heading"
        )
        evidence: list[str] = Field(
            description="Concise evidence statements grounded in genes from the supplied lists"
        )
        inconsistencies: list[str] = Field(
            description="Concise caveats or conflicting evidence; an empty list when none"
        )

    return ProgramAnnotation


def _summary_text(interpretation, evidence, inconsistencies):
    """Build a readable compatibility summary from structured annotation fields."""
    parts = [interpretation.strip()]
    if evidence:
        parts.append("Evidence: " + " ".join(f"• {item.strip()}" for item in evidence))
    if inconsistencies:
        parts.append(
            "Inconsistencies: "
            + " ".join(f"• {item.strip()}" for item in inconsistencies)
        )
    else:
        parts.append("Inconsistencies: • None identified.")
    return " ".join(part for part in parts if part)


def generate_llm_program_summaries(
    api_key,
    working_dir,
    provide_celltype=True,
    cell_type_column="cell_type",
    model="gpt-5.6-sol",
    reasoning_effort="high",
):
    """Name and describe each program from its gene list using a language model.

    Reads the programs and their activity, prompts with the templates in
    prompt_templates/, and writes the CSV the report step reads for its titles
    and descriptions. Optional; the pipeline runs without it.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM annotation requires the optional OpenAI client. Install it with "
            "'uv pip install openai==2.24.0' after activating the POSSE environment."
        ) from exc
    ProgramAnnotation = _program_annotation_model()

    llm_summary_csv_dir = os.path.join(working_dir, "llm_summaries")
    if not os.path.exists(llm_summary_csv_dir):
        os.makedirs(llm_summary_csv_dir)

    llm_with_genes_path = os.path.join(
        llm_summary_csv_dir, "program_summaries_with_genes.csv"
    )
    llm_no_genes_path = os.path.join(
        llm_summary_csv_dir, "program_summaries_no_genes.csv"
    )

    programs_dir = os.path.join(working_dir, "clique_based_programs")
    programs_json = os.path.join(programs_dir, "programs_with_loadings.json")

    cell_types = []
    if provide_celltype:
        import anndata

        program_activity_path = os.path.join(
            working_dir, "program_activity", "wePAS.h5ad"
        )
        adata = anndata.read_h5ad(program_activity_path)
        if cell_type_column not in adata.obs.columns:
            raise ValueError(
                f"provide_celltype=True but {program_activity_path} has no "
                f"obs[{cell_type_column!r}]"
            )
        cell_types = sorted(adata.obs[cell_type_column].astype(str).unique().tolist())
        logger.info(
            "Using cell types from wePAS.h5ad obs[%r]: %s",
            cell_type_column,
            cell_types,
        )
        with open("./prompt_templates/program_description.txt", "r") as f:
            prompt_template = f.read()
    else:
        with open("./prompt_templates/program_description_no_cell_type.txt", "r") as f:
            prompt_template = f.read()

    client = OpenAI(api_key=api_key)
    logger.info(
        "Annotating programs with OpenAI model %s (reasoning_effort=%s)",
        model,
        reasoning_effort,
    )

    with open(programs_json, "r") as f:
        programs_data = json.load(f)

    programs = []
    for program_info in programs_data.values():
        loadings = program_info.get("loadings", {})
        positively_loaded_genes = [
            gene for gene, loading in loadings.items() if loading > 0
        ]
        negatively_loaded_genes = [
            gene for gene, loading in loadings.items() if loading < 0
        ]
        programs.append(
            {
                "positively_loaded_genes": positively_loaded_genes,
                "negatively_loaded_genes": negatively_loaded_genes,
            }
        )

    full_data = []
    summary_data = []

    from concurrent.futures import ThreadPoolExecutor

    def process_single_program(idx, program, prompt_template, cell_types):
        """Annotate one program and return validated structured fields."""
        if len(cell_types) == 0:
            prompt = prompt_template.format(
                positively_loaded_genes=program["positively_loaded_genes"],
                negatively_loaded_genes=program["negatively_loaded_genes"],
            )
        else:
            prompt = prompt_template.format(
                cell_types=cell_types,
                positively_loaded_genes=program["positively_loaded_genes"],
                negatively_loaded_genes=program["negatively_loaded_genes"],
            )

        output = client.responses.parse(
            input=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model,
            reasoning={"effort": reasoning_effort},
            text_format=ProgramAnnotation,
        )

        annotation = output.output_parsed
        if annotation is None:
            raise RuntimeError(
                f"OpenAI returned no structured annotation for program {idx - 1}; "
                f"response status was {output.status!r}"
            )

        all_program_genes = set(program["positively_loaded_genes"]) | set(
            program["negatively_loaded_genes"]
        )
        inconsistent_list = [
            gene.strip()
            for gene in annotation.inconsistent_genes
            if gene.strip() in all_program_genes
        ]
        num_inconsistent = len(inconsistent_list)
        inconsistent_joined = ", ".join(inconsistent_list)

        genes_str = ", ".join(sorted(all_program_genes))
        num_genes = len(all_program_genes)
        interpretation = annotation.interpretation.strip()
        evidence = [item.strip() for item in annotation.evidence if item.strip()]
        inconsistencies = [
            item.strip() for item in annotation.inconsistencies if item.strip()
        ]
        summary = _summary_text(interpretation, evidence, inconsistencies)

        return (
            idx,
            genes_str,
            num_genes,
            annotation.title.strip(),
            inconsistent_joined,
            num_inconsistent,
            interpretation,
            evidence,
            inconsistencies,
            summary,
        )

    results = [None] * len(programs)

    batch_size = 3
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in tqdm(
            range(0, len(programs), batch_size), desc="Processing program batches"
        ):
            batch_programs = programs[i : i + batch_size]
            batch_indices = list(range(i + 1, i + len(batch_programs) + 1))

            logger.info(
                f"Processing batch: programs {batch_indices[0]}-{batch_indices[-1]}"
            )

            futures = []
            for idx, program in zip(batch_indices, batch_programs):
                future = executor.submit(
                    process_single_program, idx, program, prompt_template, cell_types
                )
                futures.append(future)

            for future in futures:
                result = future.result()
                logger.info("Program %d title: %s", result[0], result[3])
                results[result[0] - 1] = result

    for result in results:
        if result is not None:
            (
                idx,
                genes_str,
                num_genes,
                title,
                inconsistent_joined,
                num_inconsistent,
                interpretation,
                evidence,
                inconsistencies,
                summary,
            ) = result
            structured = [
                interpretation,
                json.dumps(evidence, ensure_ascii=False),
                json.dumps(inconsistencies, ensure_ascii=False),
            ]
            full_data.append(
                [
                    f"Program {idx - 1}",
                    title,
                    genes_str,
                    num_genes,
                    inconsistent_joined,
                    num_inconsistent,
                    *structured,
                    summary,
                ]
            )
            summary_data.append(
                [
                    f"Program {idx - 1}",
                    title,
                    num_genes,
                    num_inconsistent,
                    *structured,
                    summary,
                ]
            )

    with open(llm_with_genes_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="$", quoting=csv.QUOTE_ALL)
        writer.writerow(_WITH_GENES_HEADER)
        writer.writerows(full_data)

    with open(llm_no_genes_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="$", quoting=csv.QUOTE_ALL)
        writer.writerow(_NO_GENES_HEADER)
        writer.writerows(summary_data)

    logger.info("Generated program summaries with structured annotation fields")


_STRUCTURED_HEADER = ["Interpretation", "Evidence", "Inconsistencies"]
_WITH_GENES_HEADER = [
    "Program",
    "Title",
    "Genes",
    "Number of Genes",
    "Inconsistent Genes",
    "Number of Inconsistent Genes",
    *_STRUCTURED_HEADER,
    "Summary",
]
_NO_GENES_HEADER = [
    "Program",
    "Title",
    "Number of Genes",
    "Number of Inconsistent Genes",
    *_STRUCTURED_HEADER,
    "Summary",
]


def write_empty_program_summaries(working_dir):
    """Write the summaries CSVs with gene lists but no language-model annotations.

    Used when no API key is configured, so the report step can still run: the report
    then contains the gene lists and plots but no titles or descriptions. To annotate
    afterwards, fill the Title and Summary columns (by hand or with any LLM) and re-run
    the report step.
    """
    out_dir = os.path.join(working_dir, "llm_summaries")
    os.makedirs(out_dir, exist_ok=True)
    programs_json = os.path.join(
        working_dir, "clique_based_programs", "programs_with_loadings.json"
    )
    with open(programs_json) as f:
        programs_data = json.load(f)

    full_data, summary_data = [], []
    for pid in sorted(programs_data, key=int):
        genes = programs_data[pid].get("program", [])
        empty_structured = ["", "[]", "[]", ""]
        full_data.append(
            [
                f"Program {pid}",
                "",
                ", ".join(genes),
                len(genes),
                "",
                0,
                *empty_structured,
            ]
        )
        summary_data.append([f"Program {pid}", "", len(genes), 0, *empty_structured])

    with open(
        os.path.join(out_dir, "program_summaries_with_genes.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter="$", quoting=csv.QUOTE_ALL)
        writer.writerow(_WITH_GENES_HEADER)
        writer.writerows(full_data)
    with open(
        os.path.join(out_dir, "program_summaries_no_genes.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter="$", quoting=csv.QUOTE_ALL)
        writer.writerow(_NO_GENES_HEADER)
        writer.writerows(summary_data)
    logger.info(
        "No API key configured; wrote empty program summaries (gene lists only) to %s",
        out_dir,
    )
