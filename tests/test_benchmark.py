from pathlib import Path

from rgsn.benchmark import BenchmarkRunner, default_strategies
from rgsn.store import CandidateStore


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_words.vec"


def test_benchmark_runner_compares_multiple_strategies() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    result = BenchmarkRunner(store).run(
        strategies=default_strategies(["random", "best_neighbor", "pairwise_acquisition"]),
        target_ids=["river", "forest"],
        seed_ids=["road", "tree", "water"],
        budget=5,
        stop_rank=2,
        repeats=2,
        random_seed=11,
    )

    assert len(result.traces) == 12
    assert {summary.strategy_name for summary in result.summaries} == {
        "random",
        "best_neighbor",
        "pairwise_acquisition",
    }
    assert all(trace.steps_taken <= 5 for trace in result.traces)
    assert result.metadata["candidate_count"] == 12
    assert all(summary.mean_normalized_auc is not None for summary in result.summaries)
    assert all(len(summary.per_step_median_best_rank) == 5 for summary in result.summaries)


def test_benchmark_result_exports_json_and_csv(tmp_path: Path) -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    result = BenchmarkRunner(store).run(
        strategies=default_strategies(["random"]),
        target_ids=["river"],
        seed_ids=["road", "tree"],
        budget=4,
        random_seed=3,
    )
    out_json = tmp_path / "benchmark.json"
    out_summary = tmp_path / "summary.csv"
    out_traces = tmp_path / "traces.csv"

    result.write_json(out_json)
    result.write_summary_csv(out_summary)
    result.write_traces_csv(out_traces)

    assert '"strategy_name": "random"' in out_json.read_text(encoding="utf-8")
    assert "mean_normalized_auc" in out_summary.read_text(encoding="utf-8")
    assert "best_rank_history" in out_traces.read_text(encoding="utf-8")
