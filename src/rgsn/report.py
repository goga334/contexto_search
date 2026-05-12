from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_report(
    benchmark_json: str | Path,
    output_dir: str | Path,
    *,
    title: str = "RGSN Benchmark",
) -> dict[str, Path]:
    data = load_benchmark_json(benchmark_json)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    outputs = {
        "best_rank_curve": destination / "best_rank_curve.png",
        "success_rate": destination / "success_rate.png",
        "final_rank_distribution": destination / "final_rank_distribution.png",
        "summary": destination / "summary.md",
    }
    plot_best_rank_curve(data, outputs["best_rank_curve"], title=title)
    plot_success_rate(data, outputs["success_rate"], title=title)
    plot_final_rank_distribution(data, outputs["final_rank_distribution"], title=title)
    write_summary_markdown(data, outputs["summary"], title=title)
    return outputs


def load_benchmark_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_best_rank_curve(data: dict[str, Any], path: str | Path, *, title: str) -> None:
    summaries = data.get("summaries", [])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for summary in summaries:
        curve = summary.get("per_step_median_best_rank") or summary.get("per_step_mean_best_rank")
        if not curve:
            continue
        steps = list(range(1, len(curve) + 1))
        ax.plot(steps, curve, marker="o", linewidth=1.8, markersize=3.5, label=summary["strategy_name"])

    ax.set_title(f"{title}: Median Best Rank Over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Best rank so far")
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_success_rate(data: dict[str, Any], path: str | Path, *, title: str) -> None:
    summaries = sorted(data.get("summaries", []), key=lambda item: item["strategy_name"])
    labels = [item["strategy_name"] for item in summaries]
    values = [float(item.get("success_rate") or 0.0) for item in summaries]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color="#0f766e")
    ax.set_title(f"{title}: Success Rate")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_final_rank_distribution(data: dict[str, Any], path: str | Path, *, title: str) -> None:
    grouped: dict[str, list[float]] = {}
    for trace in data.get("traces", []):
        best_rank = trace.get("best_rank")
        if best_rank is None:
            continue
        grouped.setdefault(trace["strategy_name"], []).append(float(best_rank))

    labels = sorted(grouped)
    values = [grouped[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if values:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title(f"{title}: Final Best Rank Distribution")
    ax.set_ylabel("Final best rank")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, which="both", axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary_markdown(data: dict[str, Any], path: str | Path, *, title: str) -> None:
    lines = [f"# {title}", ""]
    metadata = data.get("metadata", {})
    if metadata:
        lines.append("## Metadata")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    lines.append("## Strategy Summary")
    lines.append("")
    lines.append("| Strategy | Success Rate | Median Best Rank | Median AUC | Median Success Step |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in sorted(data.get("summaries", []), key=lambda row: row["strategy_name"]):
        lines.append(
            "| {strategy} | {success:.3f} | {rank} | {auc} | {step} |".format(
                strategy=item["strategy_name"],
                success=float(item.get("success_rate") or 0.0),
                rank=_fmt_optional(item.get("median_best_rank")),
                auc=_fmt_optional(item.get("median_normalized_auc")),
                step=_fmt_optional(item.get("median_success_step")),
            )
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots from an RGSN benchmark JSON file.")
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="RGSN Benchmark")
    args = parser.parse_args(argv)

    outputs = generate_report(args.benchmark_json, args.output_dir, title=args.title)
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
