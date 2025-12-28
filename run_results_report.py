#!/usr/bin/env python3
"""
Result reporting/visualization helper
====================================

Reads PACD/PC outputs and synthetic benchmark summaries to generate a Markdown report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _format_edge_table(df: pd.DataFrame, cols: List[str], limit: int = 20) -> str:
    if df is None or df.empty:
        return "_No data available._\n"
    view = df.loc[:, [c for c in cols if c in df.columns]].head(limit)
    return _dataframe_to_markdown(view)


def _format_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (pd.isna(value) or pd.isnull(value)):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = df.values.tolist()
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(_format_cell(v) for v in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines]) + "\n"


def build_report(
    pacd_dir: Path,
    pc_dir: Path,
    synthetic_dir: Path,
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# PACD/PC Results Report\n")

    pacd_edges = _load_csv(pacd_dir / "final_edges.csv")
    pacd_pruned = _load_csv(pacd_dir / "pruned_edges.csv")
    pacd_graph = _load_json(pacd_dir / "causal_graph.json")
    lines.append(f"_PACD directory: `{pacd_dir}`_\n")
    lines.append("## PACD Final Edges\n")
    lines.append(
        _format_edge_table(
            pacd_edges,
            ["source", "target", "tau_direct", "ci_direct", "p_direct", "method"],
            limit=20,
        )
    )
    lines.append("## PACD Pruned Edges (sample)\n")
    lines.append(
        _format_edge_table(
            pacd_pruned,
            ["source", "target", "prune_reason", "tau_direct", "p_direct"],
            limit=20,
        )
    )

    if pacd_graph:
        lines.append("## PACD Graph Summary\n")
        lines.append(f"- Nodes: {len(pacd_graph.get('nodes', []))}\n")
        lines.append(f"- Edges: {len(pacd_graph.get('edges', []))}\n")
    elif pacd_edges is not None:
        lines.append("## PACD Graph Summary\n")
        lines.append(
            f"- Nodes: {0 if pacd_edges.empty else len(set(pacd_edges['source']) | set(pacd_edges['target']))}\n"
        )
        lines.append(f"- Edges: {0 if pacd_edges.empty else len(pacd_edges)}\n")

    pc_skeleton = _load_csv(pc_dir / "skeleton.csv")
    pc_cpdag = _load_csv(pc_dir / "cpdag.csv")
    lines.append("## PC Baseline Summary\n")
    lines.append(f"_PC directory: `{pc_dir}`_\n")
    if pc_skeleton is None or pc_cpdag is None:
        lines.append(
            "_PC outputs missing. Run `run_pc_baseline.py` with causal-learn to generate them._\n"
        )
    lines.append(f"- Skeleton edges: {0 if pc_skeleton is None else len(pc_skeleton)}\n")
    lines.append(f"- CPDAG edges: {0 if pc_cpdag is None else len(pc_cpdag)}\n")

    synthetic_summary = _load_json(synthetic_dir / "summary.json")
    if synthetic_summary:
        lines.append("## Synthetic Benchmark Summary\n")
        lines.append(f"_Synthetic directory: `{synthetic_dir}`_\n")
        summary_rows = []
        for entry in synthetic_summary:
            pacd = entry.get("pacd", {})
            pc = entry.get("pc", {})
            summary_rows.append(
                {
                    "scenario": entry.get("scenario"),
                    "n": entry.get("n"),
                    "pacd_f1": pacd.get("f1"),
                    "pc_f1": pc.get("f1") if pc else None,
                    "pacd_shd": pacd.get("shd"),
                    "pc_shd": pc.get("shd") if pc else None,
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        lines.append(_dataframe_to_markdown(summary_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PACD/PC result report")
    parser.add_argument("--pacd", default="results/pacd", help="PACD output directory")
    parser.add_argument("--pc", default="results/pc", help="PC output directory")
    parser.add_argument(
        "--synthetic", default="results/synthetic", help="Synthetic benchmark directory"
    )
    parser.add_argument(
        "--output", default="results/report.md", help="Output Markdown report"
    )
    args = parser.parse_args()

    build_report(
        pacd_dir=Path(args.pacd),
        pc_dir=Path(args.pc),
        synthetic_dir=Path(args.synthetic),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
