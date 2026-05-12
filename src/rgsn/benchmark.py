from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from rgsn.contexto import ContextoSolver
from rgsn.metrics import (
    normalized_best_rank_auc,
    per_step_mean_best_rank,
    per_step_median_best_rank,
)
from rgsn.oracle import SimilarityRankOracle
from rgsn.store import CandidateStore
from rgsn.strategies import (
    BestNeighborStrategy,
    CentroidStrategy,
    PairwiseAcquisitionStrategy,
    PairwiseDirectionStrategy,
    RandomStrategy,
    RocchioStrategy,
    SearchStrategy,
)
from rgsn.types import FeedbackObservation


@dataclass(frozen=True, slots=True)
class BenchmarkTrace:
    strategy_name: str
    target_id: str
    run_index: int
    budget: int
    stop_rank: int
    random_seed: int | None
    guesses: list[FeedbackObservation] = field(default_factory=list)
    best_rank_history: list[int] = field(default_factory=list)
    success_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.success_step is not None

    @property
    def steps_taken(self) -> int:
        return len(self.guesses)

    @property
    def best_rank(self) -> int | None:
        if not self.best_rank_history:
            return None
        return self.best_rank_history[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "target_id": self.target_id,
            "run_index": self.run_index,
            "budget": self.budget,
            "stop_rank": self.stop_rank,
            "random_seed": self.random_seed,
            "success": self.success,
            "success_step": self.success_step,
            "steps_taken": self.steps_taken,
            "best_rank": self.best_rank,
            "best_rank_history": list(self.best_rank_history),
            "metadata": dict(self.metadata),
            "guesses": [
                {
                    "candidate_id": item.candidate_id,
                    "rank": item.rank,
                    "metadata": dict(item.metadata),
                }
                for item in self.guesses
            ],
        }


@dataclass(frozen=True, slots=True)
class StrategySummary:
    strategy_name: str
    run_count: int
    success_count: int
    success_rate: float
    mean_best_rank: float | None
    median_best_rank: float | None
    mean_success_step: float | None
    median_success_step: float | None
    mean_normalized_auc: float | None
    median_normalized_auc: float | None
    per_step_mean_best_rank: list[float | None] = field(default_factory=list)
    per_step_median_best_rank: list[float | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "mean_best_rank": self.mean_best_rank,
            "median_best_rank": self.median_best_rank,
            "mean_success_step": self.mean_success_step,
            "median_success_step": self.median_success_step,
            "mean_normalized_auc": self.mean_normalized_auc,
            "median_normalized_auc": self.median_normalized_auc,
            "per_step_mean_best_rank": list(self.per_step_mean_best_rank),
            "per_step_median_best_rank": list(self.per_step_median_best_rank),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    traces: list[BenchmarkTrace]
    summaries: list[StrategySummary]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "summaries": [item.to_dict() for item in self.summaries],
            "traces": [item.to_dict() for item in self.traces],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def write_summary_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [item.to_dict() for item in self.summaries]
        _write_csv(destination, rows)

    def write_traces_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for trace in self.traces:
            rows.append(
                {
                    "strategy_name": trace.strategy_name,
                    "target_id": trace.target_id,
                    "run_index": trace.run_index,
                    "budget": trace.budget,
                    "stop_rank": trace.stop_rank,
                    "random_seed": trace.random_seed,
                    "success": trace.success,
                    "success_step": trace.success_step,
                    "steps_taken": trace.steps_taken,
                    "best_rank": trace.best_rank,
                    "best_rank_history": " ".join(str(value) for value in trace.best_rank_history),
                    "guesses": " ".join(item.candidate_id for item in trace.guesses),
                    "ranks": " ".join(str(int(item.rank)) for item in trace.guesses),
                }
            )
        _write_csv(destination, rows)


class BenchmarkRunner:
    def __init__(self, store: CandidateStore) -> None:
        self.store = store

    def run(
        self,
        *,
        strategies: Iterable[SearchStrategy],
        target_ids: Iterable[str],
        seed_ids: Iterable[str] | None = None,
        budget: int = 25,
        stop_rank: int = 1,
        repeats: int = 1,
        random_seed: int | None = None,
    ) -> BenchmarkResult:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if stop_rank <= 0:
            raise ValueError("stop_rank must be positive")
        if repeats <= 0:
            raise ValueError("repeats must be positive")

        strategy_list = list(strategies)
        if not strategy_list:
            raise ValueError("Benchmark requires at least one strategy")
        targets = list(target_ids)
        if not targets:
            raise ValueError("Benchmark requires at least one target")
        for target_id in targets:
            self.store.get(target_id)
        base_seed_ids = list(seed_ids or [])
        for seed_id in base_seed_ids:
            self.store.get(seed_id)

        traces: list[BenchmarkTrace] = []
        for strategy in strategy_list:
            for run_index in range(repeats):
                run_seed = None if random_seed is None else random_seed + run_index
                for target_id in targets:
                    oracle = SimilarityRankOracle(self.store, target_id)
                    target_seed_ids = _seed_ids_for_target(base_seed_ids, target_id)
                    strategy.reset(self.store, random_seed=run_seed)
                    traces.append(
                        self._run_one(
                            strategy=strategy,
                            oracle=oracle,
                            target_id=target_id,
                            run_index=run_index,
                            budget=budget,
                            stop_rank=stop_rank,
                            random_seed=run_seed,
                            seed_ids=target_seed_ids,
                        )
                    )

        return BenchmarkResult(
            traces=traces,
            summaries=_summarize_traces(
                traces,
                candidate_count=len(self.store.candidates),
                budget=budget,
            ),
            metadata={
                "target_ids": targets,
                "seed_ids": base_seed_ids,
                "budget": budget,
                "stop_rank": stop_rank,
                "repeats": repeats,
                "random_seed": random_seed,
                "candidate_count": len(self.store.candidates),
            },
        )

    def _run_one(
        self,
        *,
        strategy: SearchStrategy,
        oracle: SimilarityRankOracle,
        target_id: str,
        run_index: int,
        budget: int,
        stop_rank: int,
        random_seed: int | None,
        seed_ids: list[str],
    ) -> BenchmarkTrace:
        guesses: list[FeedbackObservation] = []
        best_rank_history: list[int] = []
        seed_queue = list(seed_ids)
        success_step = None

        for step in range(1, budget + 1):
            candidate_id = None
            while seed_queue and candidate_id is None:
                seed_id = seed_queue.pop(0)
                if seed_id not in {item.candidate_id for item in guesses}:
                    candidate_id = seed_id
            if candidate_id is None:
                suggestions = strategy.suggest(self.store, guesses, k=1)
                if not suggestions:
                    break
                candidate_id = suggestions[0].candidate.id

            rank = oracle.rank(candidate_id)
            observation = FeedbackObservation(
                candidate_id=candidate_id,
                rank=rank,
                metadata={"step": step, "source": "seed" if step <= len(seed_ids) else "strategy"},
            )
            guesses.append(observation)
            best_rank_history.append(int(min(item.rank for item in guesses)))
            if rank <= stop_rank:
                success_step = step
                break

        return BenchmarkTrace(
            strategy_name=strategy.name,
            target_id=target_id,
            run_index=run_index,
            budget=budget,
            stop_rank=stop_rank,
            random_seed=random_seed,
            guesses=guesses,
            best_rank_history=best_rank_history,
            success_step=success_step,
            metadata={"strategy_class": type(strategy).__name__},
        )


def default_strategies(names: Iterable[str]) -> list[SearchStrategy]:
    registry = {
        "random": RandomStrategy,
        "best_neighbor": BestNeighborStrategy,
        "centroid": CentroidStrategy,
        "rocchio": RocchioStrategy,
        "pairwise_direction": PairwiseDirectionStrategy,
        "pairwise_acquisition": PairwiseAcquisitionStrategy,
    }
    strategies = []
    for name in names:
        key = name.strip()
        if not key:
            continue
        try:
            strategies.append(registry[key]())
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown strategy '{key}'. Available: {available}") from exc
    return strategies


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an RGSN weak-feedback benchmark.")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--dictionary", default=None, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument(
        "--strategies",
        default="random,best_neighbor,centroid,rocchio,pairwise_direction,pairwise_acquisition",
    )
    parser.add_argument("--seed-words", default="")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--stop-rank", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-summary-csv", type=Path, default=None)
    parser.add_argument("--out-traces-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    solver = ContextoSolver.from_embedding_file(args.embeddings, dictionary_path=args.dictionary)
    targets = _read_word_file(args.targets)
    seed_ids = _split_words(args.seed_words)
    result = BenchmarkRunner(solver.store).run(
        strategies=default_strategies(_split_words(args.strategies)),
        target_ids=targets,
        seed_ids=seed_ids,
        budget=args.budget,
        stop_rank=args.stop_rank,
        repeats=args.repeats,
        random_seed=args.random_seed,
    )

    if args.out_json is not None:
        result.write_json(args.out_json)
    if args.out_summary_csv is not None:
        result.write_summary_csv(args.out_summary_csv)
    if args.out_traces_csv is not None:
        result.write_traces_csv(args.out_traces_csv)
    print(json.dumps({"summaries": [item.to_dict() for item in result.summaries]}, indent=2))


def _summarize_traces(
    traces: list[BenchmarkTrace],
    *,
    candidate_count: int,
    budget: int,
) -> list[StrategySummary]:
    summaries = []
    for strategy_name in sorted({trace.strategy_name for trace in traces}):
        group = [trace for trace in traces if trace.strategy_name == strategy_name]
        best_ranks = [trace.best_rank for trace in group if trace.best_rank is not None]
        success_steps = [trace.success_step for trace in group if trace.success_step is not None]
        aucs = [
            value
            for value in (
                normalized_best_rank_auc(
                    trace.best_rank_history,
                    candidate_count=candidate_count,
                    budget=budget,
                )
                for trace in group
            )
            if value is not None
        ]
        histories = [trace.best_rank_history for trace in group]
        summaries.append(
            StrategySummary(
                strategy_name=strategy_name,
                run_count=len(group),
                success_count=sum(1 for trace in group if trace.success),
                success_rate=sum(1 for trace in group if trace.success) / len(group) if group else 0.0,
                mean_best_rank=mean(best_ranks) if best_ranks else None,
                median_best_rank=median(best_ranks) if best_ranks else None,
                mean_success_step=mean(success_steps) if success_steps else None,
                median_success_step=median(success_steps) if success_steps else None,
                mean_normalized_auc=mean(aucs) if aucs else None,
                median_normalized_auc=median(aucs) if aucs else None,
                per_step_mean_best_rank=per_step_mean_best_rank(histories, budget=budget),
                per_step_median_best_rank=per_step_median_best_rank(histories, budget=budget),
            )
        )
    return summaries


def _seed_ids_for_target(seed_ids: list[str], target_id: str) -> list[str]:
    cleaned: list[str] = []
    seen = {target_id}
    for seed_id in seed_ids:
        if seed_id in seen:
            continue
        cleaned.append(seed_id)
        seen.add(seed_id)
    return cleaned


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_word_file(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_words(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
