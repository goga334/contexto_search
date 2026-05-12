import json
from pathlib import Path

from rgsn.report import generate_report


def test_generate_report_creates_plots_and_summary(tmp_path: Path) -> None:
    benchmark = {
        "metadata": {"budget": 3, "candidate_count": 100},
        "summaries": [
            {
                "strategy_name": "random",
                "success_rate": 0.5,
                "median_best_rank": 20,
                "median_normalized_auc": 0.4,
                "median_success_step": 2,
                "per_step_median_best_rank": [90, 40, 20],
            },
            {
                "strategy_name": "pairwise",
                "success_rate": 1.0,
                "median_best_rank": 5,
                "median_normalized_auc": 0.8,
                "median_success_step": 2,
                "per_step_median_best_rank": [80, 10, 5],
            },
        ],
        "traces": [
            {"strategy_name": "random", "best_rank": 20},
            {"strategy_name": "random", "best_rank": 50},
            {"strategy_name": "pairwise", "best_rank": 5},
            {"strategy_name": "pairwise", "best_rank": 8},
        ],
    }
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(benchmark), encoding="utf-8")

    outputs = generate_report(source, tmp_path / "plots", title="Tiny")

    assert outputs["best_rank_curve"].exists()
    assert outputs["success_rate"].exists()
    assert outputs["final_rank_distribution"].exists()
    assert outputs["summary"].exists()
    assert outputs["best_rank_curve"].stat().st_size > 0
    assert "Strategy Summary" in outputs["summary"].read_text(encoding="utf-8")
