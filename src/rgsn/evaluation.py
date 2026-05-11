from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from rgsn.oracle import SimilarityRankOracle
from rgsn.solver import SimulationTrace, WeakFeedbackSolver
from rgsn.store import CandidateStore


SolverFactory = Callable[[CandidateStore], WeakFeedbackSolver]


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    target_id: str
    trace: SimulationTrace
    budget: int
    stop_rank: int

    @property
    def success(self) -> bool:
        return self.trace.success_step is not None

    @property
    def steps_taken(self) -> int:
        return len(self.trace.guesses)

    @property
    def best_rank(self) -> int | None:
        if not self.trace.best_rank_history:
            return None
        return self.trace.best_rank_history[-1]

    @property
    def first_guess_rank(self) -> int | None:
        if not self.trace.best_rank_history:
            return None
        return self.trace.best_rank_history[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "budget": self.budget,
            "stop_rank": self.stop_rank,
            "success": self.success,
            "success_step": self.trace.success_step,
            "steps_taken": self.steps_taken,
            "first_guess_rank": self.first_guess_rank,
            "best_rank": self.best_rank,
            "best_rank_history": list(self.trace.best_rank_history),
            "guesses": [
                {
                    "candidate_id": observation.candidate_id,
                    "rank": observation.rank,
                    "metadata": dict(observation.metadata),
                }
                for observation in self.trace.guesses
            ],
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    cases: list[EvaluationCaseResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_count(self) -> int:
        return len(self.cases)

    @property
    def success_count(self) -> int:
        return sum(1 for case in self.cases if case.success)

    @property
    def success_rate(self) -> float:
        if not self.cases:
            return 0.0
        return self.success_count / len(self.cases)

    @property
    def mean_best_rank(self) -> float | None:
        ranks = [case.best_rank for case in self.cases if case.best_rank is not None]
        return mean(ranks) if ranks else None

    @property
    def median_best_rank(self) -> float | None:
        ranks = [case.best_rank for case in self.cases if case.best_rank is not None]
        return median(ranks) if ranks else None

    @property
    def mean_success_step(self) -> float | None:
        steps = [case.trace.success_step for case in self.cases if case.trace.success_step is not None]
        return mean(steps) if steps else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_count": self.target_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "mean_best_rank": self.mean_best_rank,
            "median_best_rank": self.median_best_rank,
            "mean_success_step": self.mean_success_step,
            "metadata": dict(self.metadata),
            "cases": [case.to_dict() for case in self.cases],
        }


class EvaluationRunner:
    """Runs hidden-target weak-feedback simulations over a vector search space."""

    def __init__(
        self,
        store: CandidateStore,
        *,
        solver_factory: SolverFactory | None = None,
    ) -> None:
        self.store = store
        self.solver_factory = solver_factory if solver_factory is not None else WeakFeedbackSolver

    def run(
        self,
        target_ids: Iterable[str],
        *,
        budget: int = 25,
        stop_rank: int = 1,
        seed_ids: Iterable[str] | None = None,
    ) -> EvaluationSummary:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if stop_rank <= 0:
            raise ValueError("stop_rank must be positive")

        targets = list(target_ids)
        if not targets:
            raise ValueError("Evaluation requires at least one target")
        for target_id in targets:
            self.store.get(target_id)

        base_seed_ids = list(seed_ids or [])
        self._validate_seed_ids(base_seed_ids)

        cases: list[EvaluationCaseResult] = []
        for target_id in targets:
            solver = self.solver_factory(self.store)
            oracle = SimilarityRankOracle(self.store, target_id)
            target_seed_ids = self._seed_ids_for_target(base_seed_ids, target_id)
            trace = solver.simulate(
                oracle,
                budget=budget,
                stop_rank=stop_rank,
                seed_ids=target_seed_ids,
            )
            cases.append(
                EvaluationCaseResult(
                    target_id=target_id,
                    trace=trace,
                    budget=budget,
                    stop_rank=stop_rank,
                )
            )

        return EvaluationSummary(
            cases=cases,
            metadata={
                "budget": budget,
                "stop_rank": stop_rank,
                "seed_ids": base_seed_ids,
            },
        )

    def _validate_seed_ids(self, seed_ids: list[str]) -> None:
        for candidate_id in seed_ids:
            self.store.get(candidate_id)

    def _seed_ids_for_target(self, seed_ids: list[str], target_id: str) -> list[str]:
        cleaned: list[str] = []
        seen = {target_id}
        for candidate_id in seed_ids:
            if candidate_id in seen:
                continue
            cleaned.append(candidate_id)
            seen.add(candidate_id)
        return cleaned
