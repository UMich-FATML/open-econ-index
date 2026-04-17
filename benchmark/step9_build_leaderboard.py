#!/usr/bin/env python3
"""
Build a compact leaderboard for smoke-test model folders.

Expected per-model artifacts (produced by run_hero_step321_eval_openrouter.sh):
  - processed_eval/*_eval_<dimension>_*_processed.jsonl
  - aggregated_scores_all_dims.jsonl
  - aggregated_scores_multi_turn_core.jsonl
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional


DIMENSIONS = [
    "tool_call",
    "workflow_completion",
    "followup_quality",
    "autonomy",
    "grounding",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build smoke-test leaderboard across model folders.")
    parser.add_argument(
        "--run_root",
        type=str,
        default="../data/smoke_hero_step321_eval",
        help="Root directory containing one folder per model run.",
    )
    parser.add_argument(
        "--model_dirs",
        type=str,
        default="",
        help="Optional comma-separated explicit model directories. Overrides --run_root discovery.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional output CSV path. Defaults to <run_root>/mini_leaderboard.csv",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="",
        help="Optional output Markdown path. Defaults to <run_root>/mini_leaderboard.md",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_file(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def mean_score_from_processed(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    rows = load_jsonl(path)
    scores = [r.get("score") for r in rows if r.get("score") is not None]
    return mean(scores)


def mean_overall(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    rows = load_jsonl(path)
    vals = [r.get("overall") for r in rows if r.get("overall") is not None]
    return mean(vals)


def normalized_dimension_score(dimension: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if dimension == "tool_call":
        return float(value)
    return (float(value) - 1.0) / 4.0


def extract_model_name_from_results(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        first = load_jsonl(path)[0]
    except Exception:
        return ""
    cfgs = first.get("metadata", {}).get("synthetic_data_gen_configs", [])
    if not cfgs:
        return ""
    gen = cfgs[-1].get("generation_params", {})
    return gen.get("model_path", "")


def extract_judge_model_from_eval_result(model_dir: str) -> str:
    # Pull from any eval result file metadata.
    path = latest_file(os.path.join(model_dir, "*_eval_*_results.jsonl"))
    return extract_model_name_from_results(path)


def discover_model_dirs(run_root: str) -> List[str]:
    if not os.path.isdir(run_root):
        return []
    children = [os.path.join(run_root, n) for n in os.listdir(run_root)]
    return sorted([p for p in children if os.path.isdir(p)])


def fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def build_row(model_dir: str) -> Dict:
    processed_dir = os.path.join(model_dir, "processed_eval")

    dim_means = {}
    for dim in DIMENSIONS:
        dim_path = latest_file(os.path.join(processed_dir, f"*_eval_{dim}_*_processed.jsonl"))
        dim_means[dim] = mean_score_from_processed(dim_path)

    agg_all = latest_file(os.path.join(model_dir, "aggregated_scores_all_dims.jsonl"))
    agg_mt = latest_file(os.path.join(model_dir, "aggregated_scores_multi_turn_core.jsonl"))
    overall_4dim = mean_overall(agg_all)
    overall_mt = mean_overall(agg_mt)

    normalized_values = [
        normalized_dimension_score(dim, dim_means[dim])
        for dim in DIMENSIONS
        if dim_means[dim] is not None
    ]
    macro_5dim = mean(normalized_values)

    candidate_path = latest_file(os.path.join(model_dir, "*_multiagent_pfc_results*.jsonl"))
    candidate_model = extract_model_name_from_results(candidate_path)
    judge_model = extract_judge_model_from_eval_result(model_dir)

    return {
        "model_folder": os.path.basename(model_dir.rstrip("/")),
        "candidate_model": candidate_model,
        "judge_model": judge_model,
        "tool_call_mean": dim_means["tool_call"],
        "workflow_completion_mean": dim_means["workflow_completion"],
        "followup_quality_mean": dim_means["followup_quality"],
        "autonomy_mean": dim_means["autonomy"],
        "grounding_mean": dim_means["grounding"],
        "overall_4dim_norm": overall_4dim,
        "overall_mt_norm": overall_mt,
        "macro_5dim_norm": macro_5dim,
    }


def write_csv(rows: List[Dict], output_csv: str):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "model_folder",
        "candidate_model",
        "judge_model",
        "tool_call_mean",
        "workflow_completion_mean",
        "followup_quality_mean",
        "autonomy_mean",
        "grounding_mean",
        "overall_4dim_norm",
        "overall_mt_norm",
        "macro_5dim_norm",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict], output_md: str):
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    lines = [
        "| Model Folder | Candidate | Judge | Tool (0-1) | Workflow (1-5) | Follow-up (1-5) | Autonomy (1-5) | Grounding (1-5) | Overall 4D (0-1) | Overall MT (0-1) | Macro 5D (0-1) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_folder']} | {r['candidate_model'] or '-'} | {r['judge_model'] or '-'} | "
            f"{fmt(r['tool_call_mean'])} | {fmt(r['workflow_completion_mean'])} | "
            f"{fmt(r['followup_quality_mean'])} | {fmt(r['autonomy_mean'])} | "
            f"{fmt(r['grounding_mean'])} | {fmt(r['overall_4dim_norm'])} | "
            f"{fmt(r['overall_mt_norm'])} | {fmt(r['macro_5dim_norm'])} |"
        )
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    if args.model_dirs.strip():
        model_dirs = [x.strip() for x in args.model_dirs.split(",") if x.strip()]
    else:
        model_dirs = discover_model_dirs(args.run_root)

    if not model_dirs:
        raise SystemExit("No model folders found.")

    rows = [build_row(d) for d in model_dirs]
    rows.sort(
        key=lambda r: (r["macro_5dim_norm"] is None, -(r["macro_5dim_norm"] or -1.0))
    )

    output_csv = args.output_csv or os.path.join(args.run_root, "mini_leaderboard.csv")
    output_md = args.output_md or os.path.join(args.run_root, "mini_leaderboard.md")

    write_csv(rows, output_csv)
    write_markdown(rows, output_md)

    print("Leaderboard built.")
    print(f"  CSV: {output_csv}")
    print(f"  MD:  {output_md}")

    print("\nTop rows:")
    for r in rows:
        print(
            f"  {r['model_folder']}: macro_5dim={fmt(r['macro_5dim_norm'])}, "
            f"overall_4dim={fmt(r['overall_4dim_norm'])}, overall_mt={fmt(r['overall_mt_norm'])}"
        )


if __name__ == "__main__":
    main()
