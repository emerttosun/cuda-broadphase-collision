#!/usr/bin/env python3
"""Plot CMP674 benchmark CSV outputs.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py results/timings.csv --out results/plots
"""

from __future__ import annotations

import argparse
import csv
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


def save_line_plot(rows, out_dir: Path, y_key: str, y_label: str, filename: str, log_y: bool = True):
    plt = require_matplotlib()
    for distribution in sorted({row["distribution_type"] for row in rows}):
        subset = [row for row in rows if row["distribution_type"] == distribution]
        plt.figure(figsize=(9, 5))
        for (method,), items in sorted(group_by(subset, "method_name").items()):
            x, y = sorted_xy(items, "object_count", y_key)
            plt.plot(x, y, marker="o", linewidth=1.8, label=method)
        plt.xlabel("Object count")
        plt.ylabel(y_label)
        plt.xscale("log")
        if log_y:
            plt.yscale("log")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{distribution}_{filename}", dpi=160)
        plt.close()


def save_grid_histogram(rows, out_dir: Path):
    plt = require_matplotlib()
    grid_rows = [row for row in rows if row["method_name"] == "cuda_uniform_grid"]
    if not grid_rows:
        return

    latest_count = max(int(row["object_count"]) for row in grid_rows)
    subset = [row for row in grid_rows if int(row["object_count"]) == latest_count]
    labels = [
        f"{row['distribution_type']} cell={float(row['grid_cell_size']):g}"
        for row in subset
    ]
    values = [float(row["max_objects_in_cell"]) for row in subset]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("Max objects in cell")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_max_objects_in_cell.png", dpi=160)
    plt.close()


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

    save_line_plot(rows, out_dir, "total_time_ms", "Total time (ms)", "total_time.png")
    save_line_plot(rows, out_dir, "speedup_vs_cpu", "Speedup vs CPU", "speedup.png", log_y=False)
    save_line_plot(rows, out_dir, "candidate_pair_count", "Candidate pairs", "candidate_pairs.png")
    save_grid_histogram(rows, out_dir)

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
