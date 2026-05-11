#!/usr/bin/env python3
"""Plot CMP674 benchmark CSV outputs.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py results/timings.csv --out results/plots

For every (distribution, radius_profile) scenario this writes line plots of
total time, kernel-only time, speedup vs CPU, candidate-pair count and the
candidate-pair pruning ratio (candidates / N(N-1)/2); plus a grid occupancy
histogram and a Markdown summary table (summary.md) of per-method metrics at
the largest object count.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: python -m pip install matplotlib"
        ) from exc
    return plt


def group_by(rows: list[dict[str, str]], *keys: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def sorted_xy(items: list[dict[str, str]], x_key: str, y_key: str):
    points = sorted((float(row[x_key]), float(row[y_key])) for row in items)
    return [p[0] for p in points], [p[1] for p in points]


def series_label(row: dict[str, str]) -> str:
    method = row["method_name"]
    if method == "cuda_uniform_grid":
        return f"{method} cell={float(row['grid_cell_size']):g}"
    return method


_CELL_RE = re.compile(r"cell=(\d+(?:\.\d+)?)")


def series_sort_key(label: str):
    """Non-grid methods first (alphabetical), then uniform-grid variants ordered
    by numeric cell size — so the legend reads cell=5, 10, 20, 40 rather than
    the string order 10, 20, 40, 5."""
    m = _CELL_RE.search(label)
    if m:
        return (1, float(m.group(1)), label)
    return (0, 0.0, label)


def plot_group_keys(rows: list[dict[str, str]]):
    return ("distribution_type", "radius_profile") if "radius_profile" in rows[0] else ("distribution_type",)


def group_suffix(key_values, has_radius: bool) -> str:
    return f"{key_values[0]}_{key_values[1]}" if has_radius else key_values[0]


def save_line_plot(rows, out_dir: Path, y_key: str, y_label: str, filename: str, log_y: bool = True):
    plt = require_matplotlib()
    keys = plot_group_keys(rows)
    has_radius = len(keys) == 2
    for key_values, subset in sorted(group_by(rows, *keys).items()):
        suffix = group_suffix(key_values, has_radius)
        by_series = defaultdict(list)
        for row in subset:
            by_series[series_label(row)].append(row)
        plt.figure(figsize=(9, 5))
        for label in sorted(by_series, key=series_sort_key):
            x, y = sorted_xy(by_series[label], "object_count", y_key)
            plt.plot(x, y, marker="o", linewidth=1.8, label=label)
        plt.xlabel("Object count")
        plt.ylabel(y_label)
        plt.xscale("log")
        if log_y:
            plt.yscale("log")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{suffix}_{filename}", dpi=160)
        plt.close()


def save_grid_histogram(rows, out_dir: Path):
    plt = require_matplotlib()
    grid_rows = [row for row in rows if row["method_name"] == "cuda_uniform_grid"]
    if not grid_rows:
        return

    latest_count = max(int(row["object_count"]) for row in grid_rows)
    subset = [row for row in grid_rows if int(row["object_count"]) == latest_count]
    subset.sort(key=lambda r: (r["distribution_type"], r.get("radius_profile", ""), float(r["grid_cell_size"])))

    labels = []
    for row in subset:
        radius = row.get("radius_profile")
        prefix = f"{row['distribution_type']} {radius}" if radius else row["distribution_type"]
        labels.append(f"{prefix} cell={float(row['grid_cell_size']):g}")
    values = [float(row["max_objects_in_cell"]) for row in subset]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=40, ha="right", fontsize=8)
    plt.ylabel(f"Max objects in cell (N = {latest_count})")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_max_objects_in_cell.png", dpi=160)
    plt.close()


def write_summary(rows, out_dir: Path):
    keys = plot_group_keys(rows)
    has_radius = len(keys) == 2
    lines = [
        "# Benchmark summary",
        "",
        "Per-method metrics at the largest object count of each scenario.",
        "`kernel_ms` is the GPU broad-phase kernels only; `total_ms` adds H2D/D2H "
        "transfers, allocation and (for the grid) the host-side stats pass.",
        "",
    ]
    for key_values, subset in sorted(group_by(rows, *keys).items()):
        title = group_suffix(key_values, has_radius).replace("_", " / ")
        max_n = max(int(r["object_count"]) for r in subset)
        at_n = [r for r in subset if int(r["object_count"]) == max_n]
        lines.append(f"## {title}  (N = {max_n})")
        lines.append("")
        lines.append("| method | kernel_ms | total_ms | speedup_vs_cpu | candidate_pairs | candidates / N(N-1)/2 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in sorted(at_n, key=lambda row: series_sort_key(series_label(row))):
            lines.append(
                "| {m} | {k:.3f} | {t:.3f} | {s:.2f} | {c} | {p:.3e} |".format(
                    m=series_label(r),
                    k=float(r["kernel_time_ms"]),
                    t=float(r["total_time_ms"]),
                    s=float(r["speedup_vs_cpu"]),
                    c=int(float(r["candidate_pair_count"])),
                    p=float(r["pruning_ratio"]),
                )
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def add_derived_columns(rows):
    for row in rows:
        n = float(row["object_count"])
        total_pairs = n * (n - 1.0) / 2.0 if n > 1.0 else 1.0
        candidates = float(row["candidate_pair_count"])
        row["pruning_ratio"] = repr(candidates / total_pairs if total_pairs > 0.0 else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark plots from results/timings.csv")
    parser.add_argument("csv", nargs="?", default="results/timings.csv", help="Input benchmark CSV")
    parser.add_argument("--out", default="results/plots", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in CSV: {csv_path}")
    add_derived_columns(rows)

    save_line_plot(rows, out_dir, "total_time_ms", "Total time (ms)", "total_time.png")
    save_line_plot(rows, out_dir, "kernel_time_ms", "Kernel time (ms)", "kernel_time.png")
    save_line_plot(rows, out_dir, "speedup_vs_cpu", "Speedup vs CPU", "speedup.png", log_y=False)
    save_line_plot(rows, out_dir, "candidate_pair_count", "Candidate pairs", "candidate_pairs.png")
    save_line_plot(rows, out_dir, "pruning_ratio", "Candidate pairs / N(N-1)/2", "pruning_ratio.png")
    save_grid_histogram(rows, out_dir)
    write_summary(rows, out_dir)

    print(f"Wrote plots and summary.md to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
