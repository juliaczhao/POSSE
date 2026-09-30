"""Self-contained, interactive HTML report for the program-finding pipeline.

Bundles the dataset KPIs, the sePAS program-enrichment heatmap, a searchable
program explorer and per-program flashcards into a single .html file with every
image embedded (base64), so it can be shared as one file. Styled with the Broad
Institute palette (primary blue #00609F), Outfit for headers and Plus Jakarta
Sans for body text.

Reads (all produced by earlier pipeline steps):
    <working_dir>/program_activity/sePAS.h5ad
    <working_dir>/clique_based_programs/programs_with_loadings.json
    <working_dir>/llm_summaries/program_summaries_no_genes.csv
    <working_dir>/gene_names/gene_names.npy
    <working_dir>/report_plots/global_plots/{umap_by_cell_type,global_umap_leiden}.png
    <working_dir>/report_plots/program_plots/sepas_otsu_binary/program_<n>_sepas_otsu_binary.png

Writes:
    <working_dir>/reports/program_analysis_report.html
"""

import base64
import csv
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


def _available_cpu_count():
    """Return CPUs available to this process, respecting scheduler affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


# Broad Institute palette.
BROAD = {
    "blue": "#00609F",
    "blue_dark": "#003E66",
    "blue_deep": "#00243B",
    "blue_mid": "#2E86C1",
    "sky": "#7FB2D6",
    "tint": "#E7F0F7",
    "ink": "#15242E",
    "body": "#34495B",
    "muted": "#6B7C8C",
    "line": "#E2E9EF",
    "canvas": "#F3F6F9",
    "white": "#FFFFFF",
    "pos": "#00609F",
    "neg": "#C0573B",
}


# ──────────────────────────────────────────────────────────────────────────
# Image embedding
# ──────────────────────────────────────────────────────────────────────────
def _embed_img(path, max_w=900, quality=86):
    """Downscale + JPEG-recompress an on-disk PNG into a base64 data URI so the
    final HTML stays a manageable size while remaining fully self-contained."""
    if not path or not os.path.exists(path):
        return ""
    from PIL import Image

    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, "white")
        bg.paste(rgba, mask=rgba.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _embed_img_task(task):
    """Embed one image task expressed as ``(path, max_width, quality)``."""
    return _embed_img(*task)


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
def _load_summaries(path):
    """Parse the $-delimited LLM summaries CSV into report-ready records."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        text = f.read().lstrip("﻿")
    reader = csv.reader(io.StringIO(text), delimiter="$", quotechar='"')
    header = next(reader, None)
    if not header:
        return out
    idx = {h.strip(): i for i, h in enumerate(header)}
    p_i, t_i, s_i = idx.get("Program"), idx.get("Title"), idx.get("Summary")

    def value(row, column):
        position = idx.get(column)
        return (
            row[position].strip()
            if position is not None and position < len(row)
            else ""
        )

    def list_value(row, column):
        raw = value(row, column)
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    for row in reader:
        if not row or p_i is None or p_i >= len(row):
            continue
        num = row[p_i].replace("Program", "").strip()
        out[num] = {
            "title": row[t_i].strip() if t_i is not None and t_i < len(row) else "",
            "summary": row[s_i].strip() if s_i is not None and s_i < len(row) else "",
            "interpretation": value(row, "Interpretation"),
            "evidence": list_value(row, "Evidence"),
            "inconsistencies": list_value(row, "Inconsistencies"),
        }
    return out


def _load_programs(working_dir):
    """Return programs sorted by number, each with title, extended summary, and
    alphabetically-sorted positive/negative-loading gene lists."""
    with open(
        os.path.join(
            working_dir, "clique_based_programs", "programs_with_loadings.json"
        )
    ) as f:
        pj = json.load(f)
    summaries = _load_summaries(
        os.path.join(working_dir, "llm_summaries", "program_summaries_no_genes.csv")
    )

    programs = []
    for key in sorted(pj.keys(), key=lambda k: int(k) if str(k).isdigit() else k):
        loadings = pj[key].get("loadings", {})
        genes = pj[key].get("program", [])
        all_genes = sorted(set(genes) | set(loadings.keys()))
        s = summaries.get(str(key), {})
        programs.append(
            {
                "num": int(key) if str(key).isdigit() else key,
                "title": s.get("title") or f"Program {key}",
                "summary": s.get("summary", ""),
                "interpretation": s.get("interpretation", ""),
                "evidence": s.get("evidence", []),
                "inconsistencies": s.get("inconsistencies", []),
                "n_genes": len(all_genes),
                "pos_genes": sorted([g for g, v in loadings.items() if v > 0]),
                "neg_genes": sorted([g for g, v in loadings.items() if v < 0]),
                "all_genes": all_genes,
            }
        )
    return programs


def _format_description(program):
    """Render structured annotations, with compatibility for older CSV files."""
    interpretation = program.get("interpretation", "").strip()
    evidence_items = program.get("evidence", [])
    inconsistency_items = program.get("inconsistencies", [])
    if interpretation or evidence_items or inconsistency_items:
        html = f"<p>{_esc(interpretation)}</p>" if interpretation else ""
        if evidence_items:
            items = "".join(f"<li>{_esc(item)}</li>" for item in evidence_items)
            html += f"<p class='desc-h'>Evidence</p><ul>{items}</ul>"
        items = inconsistency_items or ["None identified."]
        rendered = "".join(f"<li>{_esc(item)}</li>" for item in items)
        html += f"<p class='desc-h'>Inconsistencies</p><ul>{rendered}</ul>"
        return html

    summary = program.get("summary", "")
    import re

    s = (summary or "").strip()
    if not s:
        return "<p>No description available.</p>"
    s = re.sub(
        r"\*{1,2}(Evidence|Inconsisten\w*)\s*:\*{1,2}",
        r"\1:",
        s,
        flags=re.I,
    )
    ev = re.search(r"Evidence\s*:", s, re.I)
    inc = re.search(r"Inconsisten\w*\s*:", s, re.I)
    if ev and inc and inc.start() > ev.start():
        intro, evidence, incons = (
            s[: ev.start()],
            s[ev.end() : inc.start()],
            s[inc.end() :],
        )
    elif ev:
        intro, evidence, incons = s[: ev.start()], s[ev.end() :], ""
    else:
        intro, evidence, incons = s, "", ""

    def bullet_html(t):
        items = [b.strip(" .-–•\t\n") for b in re.split(r"\s+[-–•]\s+", t.strip())]
        out = []
        for b in items:
            if not b:
                continue
            m = re.match(r"([^:]{2,60}:)\s*(.*)", b, re.S)  # bold a short "lead-in:"
            out.append(
                f"<li><b>{_esc(m.group(1))}</b> {_esc(m.group(2))}</li>"
                if m
                else f"<li>{_esc(b)}</li>"
            )
        return "".join(out)

    html = ""
    if intro.strip():
        html += f"<p>{_esc(intro.strip())}</p>"
    if evidence.strip():
        html += f"<p class='desc-h'>Evidence</p><ul>{bullet_html(evidence)}</ul>"
    if incons.strip():
        html += f"<p class='desc-h'>Inconsistencies</p><ul>{bullet_html(incons)}</ul>"
    return html


# Exact normalized aliases avoid treating composite metadata such as
# ``PatientTypeID`` as either a sample or patient identifier.
_SAMPLE_ALIASES = (
    "sampleid",
    "sample",
    "libraryid",
    "library",
    "batchid",
    "origident",
)
_PATIENT_ALIASES = (
    "patientid",
    "pid",
    "donorid",
    "donor",
    "subjectid",
    "subject",
    "individualid",
    "individual",
)


def _normalized_column_name(value):
    import re

    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_metadata_column(columns, aliases):
    normalized = {_normalized_column_name(column): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _find_report_cell_type_column(columns, requested=None):
    """Resolve an explicit report label or the finest recognizable annotation."""
    columns = list(columns)
    if requested:
        if requested not in columns:
            raise KeyError(
                f"Configured report cell-type column {requested!r} is absent from .obs"
            )
        return requested

    normalized = {_normalized_column_name(column): column for column in columns}
    preferred = (
        "celltypelong",
        "celltypelabellong",
        "celltypefull",
        "celltypefine",
        "cl295v11subfull",
    )
    for alias in preferred:
        if alias in normalized:
            return normalized[alias]
    for name, column in normalized.items():
        if name.endswith("subfull") or (
            "celltype" in name
            and any(token in name for token in ("long", "full", "fine"))
        ):
            return column
    return normalized.get("celltype")


def _dataset_stats(working_dir, programs, cell_type_column=None, activity_adata=None):
    """Dataset + program KPIs. Cells/cell-types/studies/patients from sePAS.h5ad
    (small, backed), total genes from gene_names.npy."""
    import anndata as ad

    stats = {}
    owns_activity = activity_adata is None
    a = activity_adata
    if a is None:
        a = ad.read_h5ad(
            os.path.join(working_dir, "program_activity", "sePAS.h5ad"),
            backed="r",
        )
    obs = a.obs
    cell_type_column = _find_report_cell_type_column(obs.columns, cell_type_column)
    stats["n_cells"] = int(a.n_obs)
    stats["n_cell_types"] = (
        int(obs[cell_type_column].nunique(dropna=True)) if cell_type_column else 0
    )
    stats["cell_type_column"] = cell_type_column
    stats["n_studies"] = (
        int(obs["Study"].astype(str).nunique()) if "Study" in obs.columns else None
    )
    sample_column = _find_metadata_column(obs.columns, _SAMPLE_ALIASES)
    patient_column = _find_metadata_column(obs.columns, _PATIENT_ALIASES)
    stats["n_samples"] = (
        int(obs[sample_column].nunique(dropna=True)) if sample_column else None
    )
    stats["n_patients"] = (
        int(obs[patient_column].nunique(dropna=True)) if patient_column else None
    )
    stats["sample_column"] = sample_column
    stats["patient_column"] = patient_column
    if owns_activity:
        a.file.close()

    gn = os.path.join(working_dir, "gene_names", "gene_names.npy")
    if os.path.exists(gn):
        stats["n_genes"] = int(len(np.load(gn, allow_pickle=True)))
    else:
        r = ad.read_h5ad(
            os.path.join(working_dir, "anndata_files", "raw_adata_chunk_1.h5ad"),
            backed="r",
        )
        stats["n_genes"] = int(r.n_vars)
        r.file.close()

    stats["n_programs"] = len(programs)
    stats["n_genes_in_programs"] = (
        len(set().union(*[p["all_genes"] for p in programs])) if programs else 0
    )
    return stats


# ──────────────────────────────────────────────────────────────────────────
# Enrichment heatmap (one-vs-rest AUROC of sePAS activity per cell type)
# These helpers keep report assembly independent of RAPIDS.
# ──────────────────────────────────────────────────────────────────────────
def _minmax_per_program(score):
    lo = np.nanmin(score, axis=0, keepdims=True)
    rng = np.nanmax(score, axis=0, keepdims=True) - lo
    rng[rng == 0] = 1.0
    return (score - lo) / rng


def _auc_one_vs_rest(score_norm, codes, n_types):
    from scipy.stats import rankdata

    n_cells, n_programs = score_norm.shape
    n_pos = np.bincount(codes, minlength=n_types).astype(float)
    n_neg = n_cells - n_pos
    auc = np.full((n_types, n_programs), np.nan)

    def compute_column(j):
        ranks = rankdata(score_norm[:, j])
        sums = np.zeros(n_types)
        np.add.at(sums, codes, ranks)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (sums - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    worker_count = min(32, _available_cpu_count(), max(1, n_programs))
    if worker_count == 1:
        columns = map(compute_column, range(n_programs))
        for j, column in enumerate(columns):
            auc[:, j] = column
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for j, column in enumerate(executor.map(compute_column, range(n_programs))):
                auc[:, j] = column
    return auc


def _diagonal_order(auc):
    safe = np.where(np.isnan(auc), -np.inf, auc)
    return np.lexsort((-np.max(safe, axis=0), np.argmax(safe, axis=0)))


def _blue_ramp(value):
    """Map a zero-to-one value onto the report's blue color ramp."""
    stops = (
        (0.00, (255, 255, 255)),
        (0.25, (207, 226, 239)),
        (0.50, (127, 178, 214)),
        (0.80, (0, 96, 159)),
        (1.00, (0, 62, 102)),
    )
    value = float(np.clip(value, 0.0, 1.0))
    for (left, left_rgb), (right, right_rgb) in zip(stops, stops[1:]):
        if value <= right:
            fraction = (value - left) / (right - left)
            rgb = tuple(
                round(a + fraction * (b - a)) for a, b in zip(left_rgb, right_rgb)
            )
            return "#{:02X}{:02X}{:02X}".format(*rgb)
    return "#003E66"


def _build_heatmap_svg(values, x_labels, y_labels, tooltip, minimum, maximum):
    """Build an accessible inline-SVG heatmap with explicit tooltip metadata."""
    values = np.asarray(values, dtype=float)
    n_rows, n_cols = values.shape
    cell_w, cell_h = 32, 24
    left, top, right, bottom = 190, 12, 90, 190
    width = left + n_cols * cell_w + right
    height = top + n_rows * cell_h + bottom
    span = maximum - minimum if maximum > minimum else 1.0

    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            value = values[row, col]
            normalized = 0.0 if not np.isfinite(value) else (value - minimum) / span
            title = _esc(tooltip(row, col, value))
            cells.append(
                f'<rect class="hm-cell" x="{left + col * cell_w}" '
                f'y="{top + row * cell_h}" width="{cell_w - 1}" height="{cell_h - 1}" '
                f'fill="{_blue_ramp(normalized)}" tabindex="0" role="img" '
                f'aria-label="{title}" data-tooltip="{title}"><title>{title}</title></rect>'
            )

    y_ticks = "".join(
        f'<text class="hm-y" x="{left - 10}" y="{top + row * cell_h + 16}" '
        f'text-anchor="end">{_esc(label)}</text>'
        for row, label in enumerate(y_labels)
    )
    x_ticks = "".join(
        f'<text class="hm-x" transform="translate({left + col * cell_w + 15},'
        f'{top + n_rows * cell_h + 10}) rotate(-48)" text-anchor="end">'
        f"{_esc(label)}</text>"
        for col, label in enumerate(x_labels)
    )
    legend_x = width - 55
    legend = "".join(
        f'<rect x="{legend_x}" y="{top + i * 18}" width="18" height="18" '
        f'fill="{_blue_ramp(1.0 - i / 4)}"/>'
        for i in range(5)
    )
    return (
        f'<div class="heatmap-wrap"><svg class="inline-heatmap" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Interactive heatmap">{"".join(cells)}'
        f"{y_ticks}{x_ticks}{legend}</svg></div>"
    )


def _build_enrichment_heatmap(
    working_dir,
    programs,
    cell_type_column,
    activity_adata=None,
):
    """Build a cell-type AUROC heatmap, with a Leiden fallback for one type."""
    import anndata as ad

    a = activity_adata
    if a is None:
        a = ad.read_h5ad(os.path.join(working_dir, "program_activity", "sePAS.h5ad"))
    score = np.asarray(a.obsm["synchronized_program_activity_unscaled"], dtype=float)
    cell_types = a.obs[cell_type_column].astype(str).values
    types_sorted = sorted(np.unique(cell_types))

    if len(types_sorted) > 1:
        group_name = "Author cell type"
        group_labels = cell_types
        groups_sorted = types_sorted
        title = "Program enrichment across author cell types"
        note = (
            f"One-vs-rest AUROC between each program's sePAS activity and each author "
            f"cell type from .obs[{cell_type_column!r}]. "
            "Programs are ordered for diagonal dominance; hover any cell for details."
        )
    else:
        wepas_path = os.path.join(working_dir, "program_activity", "wePAS.h5ad")
        wepas = ad.read_h5ad(wepas_path, backed="r")
        if "leiden" not in wepas.obs.columns:
            wepas.file.close()
            return {
                "title": "Program enrichment",
                "note": (
                    "Enrichment requires at least two cell types or report-stage Leiden "
                    "clusters; neither is available for this run."
                ),
                "html": '<div class="missing-figure">No comparison groups are available.</div>',
            }
        if not a.obs_names.equals(wepas.obs_names):
            wepas.file.close()
            raise ValueError(
                "sePAS and wePAS cell order differs; cannot build cluster heatmap"
            )
        group_labels = wepas.obs["leiden"].astype(str).values
        groups_sorted = sorted(np.unique(group_labels), key=lambda value: int(value))
        wepas.file.close()
        group_name = "Leiden cluster"
        title = "Program enrichment across Leiden clusters"
        note = (
            f"This run contains one broad cell type ({types_sorted[0]}), so cell-type "
            "one-vs-rest AUROC is undefined. This fallback shows one-vs-rest AUROC "
            "across the report-stage Leiden clusters; hover any cell for details."
        )

    code_of = {label: index for index, label in enumerate(groups_sorted)}
    codes = np.array([code_of[label] for label in group_labels])

    auc = _auc_one_vs_rest(_minmax_per_program(score), codes, len(groups_sorted))
    order = _diagonal_order(auc)
    auc_ord = auc[:, order]

    import re

    title_by_num = {str(p["num"]): p["title"] for p in programs}
    program_ids = []
    for index, value in enumerate(a.var_names):
        match = re.search(r"(?:^|_)program_(\d+)(?:_|$)", str(value))
        program_ids.append(match.group(1) if match else str(index))
    nums = [program_ids[int(index)] for index in order]

    def _name(n):
        title = title_by_num.get(str(n), "").strip()
        if not title or title.casefold() == f"program {n}".casefold():
            return f"P{n}"
        return title

    def _label(n):
        name = _name(n)
        if name == f"P{n}":
            return name
        shortened = name[:22] + "…" if len(name) > 23 else name
        return f"{shortened} (P{n})"

    full = [f"{_name(n)} (P{n})" for n in nums]
    html = _build_heatmap_svg(
        auc_ord,
        [_label(n) for n in nums],
        groups_sorted,
        lambda row, col, value: (
            f"{group_name}: {groups_sorted[row]} | Program: {full[col]} | AUROC: {value:.3f}"
        ),
        0.5,
        1.0,
    )
    return {"title": title, "note": note, "html": html}


def _build_study_celltype_heatmap(working_dir, cell_type_column, activity_adata=None):
    """Interactive study x cell-type heatmap: rows = cell types, columns =
    studies, each cell = number of cells of that type contributed by the study.
    Color is on a log scale; browser tooltips show exact counts."""
    import anndata as ad
    import pandas as pd

    owns_activity = activity_adata is None
    a = activity_adata
    if a is None:
        a = ad.read_h5ad(
            os.path.join(working_dir, "program_activity", "sePAS.h5ad"),
            backed="r",
        )
    obs = a.obs
    if "Study" not in obs.columns or cell_type_column not in obs.columns:
        if owns_activity:
            a.file.close()
        return ""
    ct = pd.crosstab(obs[cell_type_column].astype(str), obs["Study"].astype(str))
    if owns_activity:
        a.file.close()

    cell_types = sorted(ct.index.tolist())
    studies = (
        ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    )  # biggest first
    counts = ct.loc[cell_types, studies].values.astype(int)
    z = np.log10(counts + 1.0)  # color on a log scale

    return _build_heatmap_svg(
        z,
        studies,
        cell_types,
        lambda row, col, _value: (
            f"Study: {studies[col]} | Cell type: {cell_types[row]} | Cells: {counts[row, col]:,}"
        ),
        0.0,
        float(z.max()) if z.size else 1.0,
    )


# ──────────────────────────────────────────────────────────────────────────
# Inline SVG icons (stroke = currentColor)
# ──────────────────────────────────────────────────────────────────────────
def _svg(body, sw=1.8):
    return (
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


ICONS = {
    "home": _svg(
        '<path d="M3 10.6 12 3l9 7.6"/><path d="M5.5 9.4V21h13V9.4"/><path d="M9.5 21v-6h5v6"/>'
    ),
    "umap": _svg(
        '<path d="M4 4v16h16"/><circle cx="8.5" cy="14" r="1.4"/><circle cx="12" cy="9" r="1.4"/>'
        '<circle cx="16.5" cy="12.5" r="1.4"/><circle cx="18" cy="6.5" r="1.4"/>'
    ),
    "explorer": _svg('<circle cx="11" cy="11" r="6.5"/><path d="m20 20-3.6-3.6"/>'),
    "cells": _svg(
        '<circle cx="8" cy="9" r="3"/><circle cx="16" cy="10.5" r="2.4"/><circle cx="11.5" cy="16" r="2.6"/>'
    ),
    "genes": _svg(
        '<path d="M7 3c0 5 10 7 10 13M7 21c0-5 10-7 10-13"/><path d="M8.5 6h6M9.5 9.2h6M8 14.8h6M9.5 18h6"/>'
    ),
    "celltypes": _svg(
        '<path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z"/><path d="m4 12 8 4.5 8-4.5"/>'
    ),
    "programs": _svg(
        '<rect x="3.5" y="3.5" width="7" height="7" rx="1.4"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.4"/>'
        '<rect x="3.5" y="13.5" width="7" height="7" rx="1.4"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.4"/>'
    ),
    "genesin": _svg(
        '<rect x="3.5" y="4.5" width="17" height="15" rx="2.2"/><path d="M8 8.5c0 3 8 4.5 8 8M8 16.5c0-3 8-4.5 8-8"/>'
    ),
    "samples": _svg(
        '<path d="M9 3h6"/><path d="M10 3.5v13a2 2 0 0 0 4 0v-13"/><path d="M10 9.5h4"/>'
    ),
    "studies": _svg(
        '<path d="M9.5 3h5M10.5 3.4v5L6 17a2 2 0 0 0 1.8 3h8.4a2 2 0 0 0 1.8-3l-4.5-8.6v-5"/>'
        '<path d="M8 14h8"/>'
    ),
}


def _kpi_card(value, label, icon, onclick=None):
    cls = "kpi kpi-link" if onclick else "kpi"
    attr = (
        f' onclick="{onclick}" role="button" tabindex="0" '
        f"onkeydown=\"if(event.key==='Enter'||event.key===' '){{event.preventDefault();{onclick}}}\""
        if onclick
        else ""
    )
    return (
        f'<div class="{cls}"{attr}><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-body"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div></div>'
    )


# ──────────────────────────────────────────────────────────────────────────
# CSS / JS (plain strings — no f-string brace escaping needed)
# ──────────────────────────────────────────────────────────────────────────
CSS = """
:root{
  --blue:#00609F; --blue-dark:#003E66; --blue-deep:#00243B; --blue-mid:#2E86C1;
  --sky:#7FB2D6; --tint:#E7F0F7; --ink:#15242E; --body:#34495B; --muted:#6B7C8C;
  --line:#E2E9EF; --canvas:#F3F6F9; --white:#fff;
  --head:'Outfit',system-ui,sans-serif; --sans:'Plus Jakarta Sans',system-ui,sans-serif;
  --shadow:0 1px 2px rgba(16,42,67,.06),0 8px 24px rgba(16,42,67,.06);
  --shadow-lg:0 10px 30px rgba(16,42,67,.12);
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font-family:var(--sans);background:var(--canvas);color:var(--body);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
h1,h2,h3,h4{font-family:var(--head);color:var(--ink);margin:0;font-weight:700;letter-spacing:-.01em}
a{color:var(--blue);text-decoration:none}

/* sidebar */
.sidebar{position:fixed;top:0;left:0;width:252px;height:100vh;
  background:linear-gradient(180deg,var(--blue-dark),var(--blue-deep));
  color:#cfe0ee;display:flex;flex-direction:column;padding:30px 16px 22px;z-index:20}
.brand{display:flex;align-items:center;gap:12px;padding:6px 8px 20px}
.brand-mark{width:38px;height:38px;border-radius:11px;flex:none;
  background:linear-gradient(135deg,#1e88c9,#00609F);display:flex;align-items:center;
  justify-content:center;color:#fff;font-family:var(--head);font-weight:800;font-size:19px;
  box-shadow:0 6px 16px rgba(0,0,0,.25)}
.brand-text .b-title{font-family:var(--head);font-weight:700;color:#fff;font-size:16px;line-height:1.15}
.brand-text .b-sub{font-size:12px;color:#8fb4d2;margin-top:2px;
  max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nav{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.nav-label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#6f9bbd;
  padding:8px 12px 6px;font-weight:600}
.nav-item{display:flex;align-items:center;gap:12px;padding:11px 13px;border-radius:11px;
  color:#cfe0ee;background:none;border:0;cursor:pointer;font-family:var(--sans);font-size:14.5px;
  font-weight:500;width:100%;text-align:left;transition:background .15s,color .15s}
.nav-item svg{width:20px;height:20px;flex:none;opacity:.92}
.nav-item:hover{background:rgba(255,255,255,.08);color:#fff}
.nav-item.active{background:rgba(255,255,255,.14);color:#fff;font-weight:600;
  box-shadow:inset 3px 0 0 var(--sky)}
.side-foot{margin-top:auto;font-size:11.5px;color:#6f9bbd;padding:12px 12px 2px;
  border-top:1px solid rgba(255,255,255,.08)}

/* layout */
.content{margin-left:252px;padding:38px 46px 70px;max-width:1280px}
.page-head{margin-bottom:26px}
.page-head h1{font-size:30px}
.page-head .sub{color:var(--muted);margin-top:6px;font-size:15px}
.eyebrow{font-family:var(--head);font-size:13px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--blue);margin:30px 0 14px;display:flex;align-items:center;gap:10px}
.eyebrow::after{content:"";flex:1;height:1px;background:var(--line)}

/* KPI cards */
.kpi-grid{display:grid;gap:18px}
.kpi-3{grid-template-columns:repeat(3,1fr)}
.kpi-2{grid-template-columns:repeat(2,1fr);max-width:680px}
.kpi-auto{grid-template-columns:repeat(auto-fit,minmax(186px,1fr))}
.kpi{background:var(--white);border:1px solid var(--line);border-radius:18px;padding:22px;
  display:flex;align-items:center;gap:18px;box-shadow:var(--shadow);transition:transform .15s,box-shadow .15s,border-color .15s}
.kpi:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}
.kpi-link{cursor:pointer}
.kpi-link:hover{border-color:var(--sky)}
.kpi-link .kpi-label::after{content:" ↗";color:var(--blue);font-weight:700}
.kpi-icon{width:52px;height:52px;border-radius:14px;flex:none;display:flex;align-items:center;
  justify-content:center;color:var(--blue);background:var(--tint)}
.kpi-icon svg{width:26px;height:26px}
.kpi-value{font-family:var(--head);font-size:30px;font-weight:800;color:var(--ink);line-height:1}
.kpi-label{color:var(--muted);font-size:13.5px;margin-top:6px;font-weight:500}
#card-celltype-counts,#card-study-heatmap{scroll-margin-top:24px}

/* cards */
.card{background:var(--white);border:1px solid var(--line);border-radius:18px;
  padding:26px 28px;box-shadow:var(--shadow);margin-top:18px}
.card-title{font-size:20px;margin-bottom:4px}
.card-note{color:var(--muted);font-size:13.5px;margin:0 0 16px}
.heatmap-wrap{overflow-x:auto;border-radius:12px}
.heatmap-wrap img{display:block;width:100%;min-width:760px}
.inline-heatmap{display:block;width:100%;min-width:760px;height:auto;background:#fff}
.hm-cell{outline:none}.hm-cell:hover,.hm-cell:focus{stroke:var(--blue-deep);stroke-width:2px}
.hm-x,.hm-y{font-family:var(--sans);font-size:11px;fill:var(--body)}
.heatmap-tooltip{position:fixed;z-index:1000;max-width:360px;padding:9px 12px;border-radius:8px;
  background:var(--blue-deep);color:#fff;font-size:12.5px;line-height:1.4;
  box-shadow:var(--shadow-lg);pointer-events:none;white-space:normal}
.missing-figure{width:100%;height:100%;min-height:180px;border:1px dashed var(--line);
  border-radius:10px;display:flex;align-items:center;justify-content:center;text-align:center;
  color:var(--muted);background:var(--canvas);padding:18px;font-size:13px}

/* table */
.ptable{width:100%;border-collapse:separate;border-spacing:0;font-size:14px}
.ptable th{font-family:var(--head);text-align:left;font-size:12px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--blue-dark);background:var(--tint);padding:12px 16px;font-weight:700}
.ptable th:first-child{border-radius:10px 0 0 0} .ptable th:last-child{border-radius:0 10px 0 0}
.ptable td{padding:12px 16px;border-bottom:1px solid var(--line);color:var(--body)}
.ptable tbody tr{cursor:pointer;transition:background .12s}
.ptable tbody tr:hover{background:var(--tint)}
.ptable .c-num{font-family:var(--head);font-weight:700;color:var(--blue);width:90px}
.ptable .c-title{font-weight:600;color:var(--ink)}
.ptable .c-genes{width:120px;color:var(--muted)}
.pill{display:inline-block;background:var(--tint);color:var(--blue-dark);border-radius:999px;
  padding:3px 11px;font-size:12.5px;font-weight:600}

/* explorer */
.search-box{position:relative;max-width:640px;margin:4px 0 8px}
.search-box svg{position:absolute;left:18px;top:50%;transform:translateY(-50%);
  width:20px;height:20px;color:var(--muted)}
#searchInput{width:100%;padding:15px 18px 15px 50px;border:1.5px solid var(--line);border-radius:14px;
  font-family:var(--sans);font-size:15.5px;color:var(--ink);background:var(--white);box-shadow:var(--shadow)}
#searchInput:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 4px rgba(0,96,159,.12)}
.search-meta{color:var(--muted);font-size:13px;margin:14px 2px}
.result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.pcard{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:18px 18px 16px;
  cursor:pointer;box-shadow:var(--shadow);transition:transform .15s,box-shadow .15s,border-color .15s;
  display:flex;flex-direction:column;gap:10px}
.pcard:hover{transform:translateY(-3px);box-shadow:var(--shadow-lg);border-color:var(--sky)}
.pcard:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 4px rgba(0,96,159,.12)}
.pcard-top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.pcard-num{font-family:var(--head);font-weight:700;font-size:12.5px;color:var(--blue);
  background:var(--tint);border-radius:8px;padding:3px 9px}
.pcard-genes{font-size:12.5px;color:var(--muted)}
.pcard-title{font-family:var(--head);font-weight:700;font-size:16px;color:var(--ink);line-height:1.3}
.pcard-hit{font-size:12.5px;color:var(--body);background:#f3f8fc;border:1px solid var(--line);
  border-radius:8px;padding:6px 9px}
.pcard-hit b{color:var(--blue-dark);font-weight:700}
.empty{color:var(--muted);padding:40px;text-align:center;background:var(--white);
  border:1px dashed var(--line);border-radius:16px}

/* flashcard */
.back-btn{display:inline-flex;align-items:center;gap:8px;background:none;border:0;color:var(--blue);
  font-family:var(--sans);font-weight:600;font-size:14px;cursor:pointer;padding:6px 0;margin-bottom:10px}
.fc-head{display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:6px}
.fc-num{font-family:var(--head);font-weight:800;font-size:14px;color:#fff;background:var(--blue);
  border-radius:10px;padding:7px 13px;white-space:nowrap}
.fc-title{font-size:27px;line-height:1.15;flex:1;min-width:240px}
.fc-chip{align-self:center}
.fc-imgs{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}
.fc-fig{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:10px;
  box-shadow:var(--shadow);aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;overflow:hidden}
.fc-fig img{max-width:100%;max-height:100%;width:auto;height:auto;border-radius:8px}
.fc-desc{color:var(--body);font-size:14.5px;line-height:1.62}
.fc-desc p{margin:0 0 12px}
.fc-desc .desc-h{font-family:var(--head);font-weight:700;color:var(--ink);font-size:15.5px;margin:16px 0 8px}
.fc-desc ul{margin:0 0 14px;padding-left:20px}
.fc-desc li{margin-bottom:7px}
.fc-desc li b{color:var(--ink);font-weight:700}
.wide-img{width:100%;display:block;border-radius:10px}
.gene-cols{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:6px}
.gene-head{font-family:var(--head);font-weight:700;font-size:14px;margin-bottom:12px;display:flex;
  align-items:center;gap:9px}
.gene-head .dot{width:10px;height:10px;border-radius:50%}
.gene-head .ct{font-weight:600;color:var(--muted);font-size:12.5px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{font-size:12.5px;border-radius:7px;padding:3px 9px;font-weight:600;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.chip-pos{background:var(--tint);color:var(--blue-dark)}
.chip-neg{background:#fbeae4;color:#9c3f27}
.gene-none{color:var(--muted);font-size:13px;font-style:italic}

@media(max-width:980px){
  .sidebar{width:64px;padding:18px 8px}.brand-text,.nav-label,.nav-item span,.side-foot{display:none}
  .brand{justify-content:center;padding:6px 0 18px}.nav-item{justify-content:center;padding:12px}
  .content{margin-left:64px;padding:26px 20px 60px}
  .kpi-3,.kpi-2{grid-template-columns:1fr}.fc-imgs,.gene-cols{grid-template-columns:1fr}
}
"""

JS = r"""
const D = window.__REPORT__;
const byNum = Object.fromEntries(D.programs.map(p => [String(p.num), p]));
const pageHashes = {'page-home':'#home','page-umaps':'#global','page-explorer':'#programs'};

function esc(s){ return String(s).replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c])); }

function setPage(id, el){
  document.querySelectorAll('.page').forEach(p => p.hidden = (p.id !== id));
  document.querySelectorAll('.nav-item[data-page]').forEach(n => n.classList.remove('active'));
  const active = el || document.querySelector('.nav-item[data-page="'+id+'"]');
  if(active) active.classList.add('active');
  window.scrollTo(0,0);
}

function showPage(id, el){
  setPage(id, el);
  const next = pageHashes[id];
  if(next && location.hash !== next) history.pushState(null, '', next);
}

function showPageAt(id, anchorId){
  showPage(id);
  const el = document.getElementById(anchorId);
  if(el) setTimeout(() => el.scrollIntoView({behavior:'smooth', block:'start'}), 80);
}

function matchInfo(p, q){
  if(!q) return {hit:true, genes:[]};
  const ql = q.toLowerCase();
  const numQ = ql.replace(/program/g,'').trim();
  if(/^\d+$/.test(numQ)) return {hit:String(p.num).startsWith(numQ), genes:[]};
  const titleHit = p.title.toLowerCase().includes(ql);
  const genes = p.pos.concat(p.neg).filter(g => g.toLowerCase().includes(ql));
  return {hit:titleHit || genes.length>0, genes};
}

function renderResults(q){
  const grid = document.getElementById('resultGrid');
  const meta = document.getElementById('searchMeta');
  const rows = [];
  for(const p of D.programs){
    const match = matchInfo(p, q);
    if(match.hit) rows.push({p, genes:match.genes});
  }
  meta.textContent = q ? rows.length+' program'+(rows.length===1?'':'s')+' match “'+q+'”'
                       : D.programs.length+' programs';
  if(!rows.length){
    grid.innerHTML = '<div class="empty">No programs match your search.</div>';
    return;
  }
  grid.innerHTML = rows.map(({p, genes}) => {
    let hit = '';
    if(genes.length){
      const shown = genes.slice(0,6).map(esc).join(', ');
      const more = genes.length>6 ? ' +'+(genes.length-6)+' more' : '';
      hit = '<div class="pcard-hit">genes: <b>'+shown+'</b>'+more+'</div>';
    }
    return '<div class="pcard" role="button" tabindex="0" data-program="'+p.num+'">'+
      '<div class="pcard-top"><span class="pcard-num">Program '+p.num+'</span>'+
      '<span class="pcard-genes">'+p.nGenes+' genes</span></div>'+
      '<div class="pcard-title">'+esc(p.title)+'</div>'+hit+'</div>';
  }).join('');
}

function geneChips(list, cls){
  if(!list.length) return '<div class="gene-none">None</div>';
  return '<div class="chips">'+list.map(g => '<span class="chip '+cls+'">'+esc(g)+'</span>').join('')+'</div>';
}

function figure(uri, alt){
  if(!uri) return '<div class="missing-figure">'+esc(alt)+' is unavailable for this run.</div>';
  return '<img src="'+uri+'" alt="'+esc(alt)+'">';
}

const heatmapTooltip = document.getElementById('heatmapTooltip');

function placeHeatmapTooltip(text, clientX, clientY){
  heatmapTooltip.textContent = text;
  heatmapTooltip.hidden = false;
  const box = heatmapTooltip.getBoundingClientRect();
  const left = Math.max(12, Math.min(clientX + 14, window.innerWidth - box.width - 12));
  const top = Math.max(12, Math.min(clientY + 14, window.innerHeight - box.height - 12));
  heatmapTooltip.style.left = left + 'px';
  heatmapTooltip.style.top = top + 'px';
}

function hideHeatmapTooltip(){
  heatmapTooltip.hidden = true;
}

function openProgram(num, recordHistory=true){
  const p = byNum[String(num)];
  if(!p) return;
  const el = document.getElementById('flashcard');
  el.innerHTML =
    '<button class="back-btn" onclick="showPage(\'page-explorer\')">&larr; Back to Program Explorer</button>'+
    '<div class="fc-head"><span class="fc-num">Program '+p.num+'</span>'+
      '<h1 class="fc-title">'+esc(p.title)+'</h1>'+
      '<span class="pill fc-chip">'+p.nGenes+' genes</span></div>'+
    '<div class="fc-imgs">'+
      '<div class="fc-fig">'+figure(p.activityUmap,'Program activity UMAP')+'</div>'+
      '<div class="fc-fig">'+figure(p.activityViolin,'Program activity violin plot')+'</div>'+
    '</div>'+
    '<div class="card"><h2 class="card-title">Program description</h2><div class="fc-desc">'+p.descHtml+'</div></div>'+
    '<div class="card"><h2 class="card-title">Genes in program</h2><div class="gene-cols">'+
      '<div><div class="gene-head"><span class="dot" style="background:var(--blue)"></span>Positive loading'+
        '<span class="ct">'+p.pos.length+'</span></div>'+geneChips(p.pos,'chip-pos')+'</div>'+
      '<div><div class="gene-head"><span class="dot" style="background:#C0573B"></span>Negative loading'+
        '<span class="ct">'+p.neg.length+'</span></div>'+geneChips(p.neg,'chip-neg')+'</div>'+
    '</div></div>';
  setPage('page-flashcard');
  if(recordHistory && location.hash !== '#program-'+p.num){
    history.pushState(null, '', '#program-'+p.num);
  }
}

function routeFromHash(){
  const hash = location.hash || '#home';
  if(hash.startsWith('#program-')){
    openProgram(hash.slice('#program-'.length), false);
  }else if(hash === '#global'){
    setPage('page-umaps');
  }else if(hash === '#programs'){
    setPage('page-explorer');
  }else{
    setPage('page-home');
  }
}

document.getElementById('searchInput').addEventListener('input', e => renderResults(e.target.value.trim()));
document.getElementById('resultGrid').addEventListener('click', e => {
  const card = e.target.closest('[data-program]');
  if(card) openProgram(card.dataset.program);
});
document.getElementById('resultGrid').addEventListener('keydown', e => {
  const card = e.target.closest('[data-program]');
  if(card && (e.key === 'Enter' || e.key === ' ')){
    e.preventDefault();
    openProgram(card.dataset.program);
  }
});
window.addEventListener('popstate', routeFromHash);
document.querySelectorAll('.hm-cell[data-tooltip]').forEach(cell => {
  cell.addEventListener('pointerenter', event => {
    placeHeatmapTooltip(cell.dataset.tooltip, event.clientX, event.clientY);
  });
  cell.addEventListener('pointermove', event => {
    placeHeatmapTooltip(cell.dataset.tooltip, event.clientX, event.clientY);
  });
  cell.addEventListener('pointerleave', hideHeatmapTooltip);
  cell.addEventListener('focus', () => {
    const box = cell.getBoundingClientRect();
    placeHeatmapTooltip(cell.dataset.tooltip, box.left + box.width / 2, box.bottom);
  });
  cell.addEventListener('blur', hideHeatmapTooltip);
});
renderResults('');
routeFromHash();
"""


# ──────────────────────────────────────────────────────────────────────────
# HTML assembly
# ──────────────────────────────────────────────────────────────────────────
def _fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def _dataset_label(working_dir):
    base = os.path.basename(os.path.normpath(working_dir))
    if base.lower() in {"all", "data", "dataset", "out", "output"}:
        parent = os.path.basename(os.path.dirname(os.path.normpath(working_dir)))
        return parent or base
    return base


def _build_home(stats, programs, heatmap):
    # order: cells, genes, patients, studies, cell types
    cards = [
        _kpi_card(_fmt(stats["n_cells"]), "Cells", ICONS["cells"]),
        _kpi_card(_fmt(stats["n_genes"]), "Genes", ICONS["genes"]),
    ]
    if stats.get("n_samples"):
        cards.append(_kpi_card(_fmt(stats["n_samples"]), "Samples", ICONS["samples"]))
    if stats.get("n_patients"):
        cards.append(_kpi_card(_fmt(stats["n_patients"]), "Patients", ICONS["samples"]))
    if stats.get("n_studies"):
        cards.append(
            _kpi_card(
                _fmt(stats["n_studies"]),
                "Studies",
                ICONS["studies"],
                onclick="showPageAt('page-umaps','card-study-heatmap')",
            )
        )
    cards.append(
        _kpi_card(
            _fmt(stats["n_cell_types"]),
            "Cell types",
            ICONS["celltypes"],
            onclick="showPageAt('page-umaps','card-celltype-counts')",
        )
    )
    kpis_dataset = "".join(cards)
    kpis_programs = _kpi_card(
        _fmt(stats["n_programs"]), "Programs discovered", ICONS["programs"]
    ) + _kpi_card(
        _fmt(stats["n_genes_in_programs"]), "Genes in programs", ICONS["genesin"]
    )
    rows = "".join(
        f'<tr onclick="openProgram({p["num"]})" role="button" tabindex="0" '
        f"onkeydown=\"if(event.key==='Enter'||event.key===' '){{event.preventDefault();openProgram({p['num']})}}\">"
        f'<td class="c-num">{p["num"]}</td>'
        f'<td class="c-title">{_esc(p["title"])}</td>'
        f'<td class="c-genes">{p["n_genes"]} genes</td></tr>'
        for p in programs
    )
    return f"""<section id="page-home" class="page">
  <div class="page-head"><h1>Cellular Program Explorer</h1></div>

  <h2 class="eyebrow">Dataset Overview</h2>
  <div class="kpi-grid kpi-auto">{kpis_dataset}</div>

  <h2 class="eyebrow">Programs Overview</h2>
  <div class="kpi-grid kpi-2">{kpis_programs}</div>

  <div class="card">
    <h2 class="card-title">{_esc(heatmap["title"])}</h2>
    <p class="card-note">{_esc(heatmap["note"])}</p>
    {heatmap["html"]}
  </div>

  <div class="card">
    <h2 class="card-title">Programs</h2>
    <table class="ptable">
      <thead><tr><th>Program</th><th>Title</th><th># Genes</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>"""


def _figure_html(uri, alt, css_class="wide-img"):
    if not uri:
        return f'<div class="missing-figure">{_esc(alt)} is unavailable for this run.</div>'
    return f'<img class="{css_class}" src="{uri}" alt="{_esc(alt)}">'


def _build_global(umap_celltype_uri, umap_leiden_uri, counts_uri, study_html):
    study_card = (
        (
            f'<div class="card" id="card-study-heatmap">'
            f'<h2 class="card-title">Cell-type composition by study</h2>'
            f'<p class="card-note">Number of cells of each cell type contributed by each study '
            f"(log-scaled color). Hover any cell for the exact count.</p>{study_html}</div>"
        )
        if study_html
        else ""
    )
    return f"""<section id="page-umaps" class="page" hidden>
  <div class="page-head"><h1>Global visualizations</h1>
    <div class="sub">Global embedding of cells by program activity</div></div>
  <div class="card">{_figure_html(umap_celltype_uri, "UMAP by cell type")}</div>
  <div class="card">{_figure_html(umap_leiden_uri, "UMAP by Leiden cluster")}</div>
  <div class="card" id="card-celltype-counts"><h2 class="card-title">Cell counts per cell type</h2>
    {_figure_html(counts_uri, "Cell type counts")}</div>
  {study_card}
</section>"""


def _build_explorer():
    return f"""<section id="page-explorer" class="page" hidden>
  <div class="page-head"><h1>Program Explorer</h1>
    <div class="sub">Search by gene name, program title, or program number</div></div>
  <div class="search-box">{ICONS["explorer"]}
    <input id="searchInput" type="text" placeholder="Search genes (e.g. CD3D), titles, or numbers…"
           autocomplete="off" spellcheck="false"></div>
  <div class="search-meta" id="searchMeta"></div>
  <div class="result-grid" id="resultGrid"></div>
</section>
<section id="page-flashcard" class="page" hidden><div id="flashcard"></div></section>"""


def _build_sidebar():
    def item(page, icon, text):
        return (
            f'<button class="nav-item" data-page="{page}" '
            f"onclick=\"showPage('{page}', this)\">{icon}<span>{text}</span></button>"
        )

    return f"""<aside class="sidebar">
  <div class="nav-label">Navigation</div>
  <nav class="nav">
    {item("page-home", ICONS["home"], "Home")}
    {item("page-umaps", ICONS["umap"], "Global visualizations")}
    {item("page-explorer", ICONS["explorer"], "Program explorer")}
  </nav>
</aside>"""


def _esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_html_report(working_dir, out_path=None, cell_type_column=None):
    """Build the self-contained interactive HTML report for ``working_dir``."""
    import anndata as ad

    logger.info("Building HTML report for %s", working_dir)
    programs = _load_programs(working_dir)
    activity_adata = ad.read_h5ad(
        os.path.join(working_dir, "program_activity", "sePAS.h5ad")
    )
    stats = _dataset_stats(
        working_dir,
        programs,
        cell_type_column=cell_type_column,
        activity_adata=activity_adata,
    )
    label = _dataset_label(working_dir)
    logger.info(
        "Report data: %d programs, %d cells, %d cell types",
        stats["n_programs"],
        stats["n_cells"],
        stats["n_cell_types"],
    )
    logger.info(
        "Report identifiers: samples=%s from %s; patients=%s from %s",
        stats.get("n_samples"),
        stats.get("sample_column"),
        stats.get("n_patients"),
        stats.get("patient_column"),
    )
    logger.info("Report cell types: %s", stats.get("cell_type_column"))

    logger.info("Building interactive heatmaps")
    heatmap = _build_enrichment_heatmap(
        working_dir,
        programs,
        stats["cell_type_column"],
        activity_adata=activity_adata,
    )
    study_html = _build_study_celltype_heatmap(
        working_dir,
        stats["cell_type_column"],
        activity_adata=activity_adata,
    )

    gp = os.path.join(working_dir, "report_plots", "global_plots")
    program_dir = os.path.join(working_dir, "report_plots", "program_plots")
    activity_umap_dir = os.path.join(program_dir, "program_activity_umaps")
    activity_violin_dir = os.path.join(program_dir, "program_activity_violins")
    embedding_tasks = [
        (os.path.join(gp, "umap_by_cell_type.png"), 1400, 92),
        (os.path.join(gp, "global_umap_leiden.png"), 1400, 92),
        (os.path.join(gp, "cell_type_counts_bar.png"), 1400, 92),
    ]
    for program in programs:
        embedding_tasks.extend(
            [
                (
                    os.path.join(
                        activity_umap_dir,
                        f"umap_program_activity_{program['num']}.png",
                    ),
                    1000,
                    90,
                ),
                (
                    os.path.join(
                        activity_violin_dir,
                        f"program_{program['num']}_activity_violin.png",
                    ),
                    1000,
                    90,
                ),
            ]
        )
    embedding_workers = min(32, _available_cpu_count(), max(1, len(embedding_tasks)))
    logger.info(
        "Embedding %d report images with %d CPU workers",
        len(embedding_tasks),
        embedding_workers,
    )
    if embedding_workers == 1:
        embedded_images = list(map(_embed_img_task, embedding_tasks))
    else:
        with ThreadPoolExecutor(max_workers=embedding_workers) as executor:
            embedded_images = list(executor.map(_embed_img_task, embedding_tasks))

    umap_celltype, umap_leiden, counts_bar = embedded_images[:3]
    js_programs = []
    for index, p in enumerate(programs):
        activity_umap = embedded_images[3 + 2 * index]
        activity_violin = embedded_images[4 + 2 * index]
        js_programs.append(
            {
                "num": p["num"],
                "title": p["title"],
                "nGenes": p["n_genes"],
                "descHtml": _format_description(p),
                "pos": p["pos_genes"],
                "neg": p["neg_genes"],
                "activityUmap": activity_umap,
                "activityViolin": activity_violin,
            }
        )

    data_json = json.dumps({"programs": js_programs}).replace("</", "<\\/")

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cellular Program Explorer · {_esc(label)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head><body>
{_build_sidebar()}
<main class="content">
{_build_home(stats, programs, heatmap)}
{_build_global(umap_celltype, umap_leiden, counts_bar, study_html)}
{_build_explorer()}
</main>
<div id="heatmapTooltip" class="heatmap-tooltip" role="tooltip" hidden></div>
<script>window.__REPORT__ = {data_json};</script>
<script>{JS}</script>
</body></html>"""

    if out_path is None:
        out_path = os.path.join(working_dir, "reports", "program_analysis_report.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote %s (%.1f MB)", out_path, len(html) / 1e6)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("working_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell-type-column", default=None)
    args = ap.parse_args()
    generate_html_report(
        args.working_dir,
        out_path=args.out,
        cell_type_column=args.cell_type_column,
    )
