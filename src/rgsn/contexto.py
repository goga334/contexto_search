from __future__ import annotations

from pathlib import Path

from rgsn.evaluation import EvaluationRunner, EvaluationSummary
from rgsn.oracle import SimilarityRankOracle
from rgsn.solver import SimulationTrace, WeakFeedbackSolver
from rgsn.store import CandidateStore
from rgsn.types import FeedbackObservation, ScoredCandidate


class ContextoSolver:
    """Word-oriented wrapper around the generic weak-feedback solver."""

    def __init__(self, store: CandidateStore) -> None:
        self.store = store
        self.solver = WeakFeedbackSolver(store)

    @classmethod
    def from_embedding_file(
        cls,
        path: str | Path,
        *,
        lowercase_words: bool = True,
        max_words: int | None = None,
    ) -> ContextoSolver:
        store = CandidateStore.from_text_file(
            path,
            lowercase_ids=lowercase_words,
            max_items=max_words,
        )
        return cls(store)

    def observe(self, word: str, rank: float) -> FeedbackObservation:
        return self.solver.observe(word.lower(), rank)

    def suggest(self, k: int = 10) -> list[ScoredCandidate]:
        return self.solver.propose(k=k)

    def simulate(self, target_word: str, *, budget: int = 25, seed_words: list[str] | None = None) -> SimulationTrace:
        oracle = SimilarityRankOracle(self.store, target_word.lower())
        seeds = [word.lower() for word in seed_words] if seed_words is not None else None
        return self.solver.simulate(oracle, budget=budget, seed_ids=seeds)

    def evaluate(
        self,
        target_words: list[str],
        *,
        budget: int = 25,
        stop_rank: int = 1,
        seed_words: list[str] | None = None,
    ) -> EvaluationSummary:
        runner = EvaluationRunner(self.store)
        targets = [word.lower() for word in target_words]
        seeds = [word.lower() for word in seed_words] if seed_words is not None else None
        return runner.run(targets, budget=budget, stop_rank=stop_rank, seed_ids=seeds)
