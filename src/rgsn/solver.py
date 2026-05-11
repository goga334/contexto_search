from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rgsn.acquisition import AcquisitionConfig
from rgsn.constraints import PairwiseConstraintBuilder
from rgsn.direction import RankDirectionLearner
from rgsn.index import NumpyCandidateIndex
from rgsn.machine import RankGuidedSearchMachine
from rgsn.oracle import SimilarityRankOracle
from rgsn.store import CandidateStore
from rgsn.types import FeedbackObservation, ScoredCandidate


@dataclass(slots=True)
class SimulationTrace:
    target_id: str
    guesses: list[FeedbackObservation] = field(default_factory=list)
    best_rank_history: list[int] = field(default_factory=list)
    success_step: int | None = None


class WeakFeedbackSolver:
    """Reusable solver facade over a vectorized weak-rank search space."""

    def __init__(
        self,
        store: CandidateStore,
        *,
        learner: RankDirectionLearner | None = None,
        acquisition: AcquisitionConfig | None = None,
        constraint_builder: PairwiseConstraintBuilder | None = None,
        use_index: bool = True,
    ) -> None:
        self.store = store
        self.machine = RankGuidedSearchMachine(
            store.values(),
            learner=learner,
            acquisition=acquisition,
            constraint_builder=constraint_builder,
        )
        self.index = NumpyCandidateIndex.from_store(store) if use_index else None

    @property
    def observations(self) -> list[FeedbackObservation]:
        return list(self.machine.observations)

    def observe(self, candidate_id: str, rank: float, metadata: dict[str, Any] | None = None) -> FeedbackObservation:
        return self.machine.observe(candidate_id, rank, metadata=metadata)

    def propose(self, k: int = 10, *, include_seen: bool = False) -> list[ScoredCandidate]:
        if self.index is not None:
            return self.index.score_candidates(
                observations=self.machine.observations,
                direction=self.machine.direction(),
                config=self.machine.acquisition,
                k=k,
                include_seen=include_seen,
            )
        return self.machine.propose(k=k, include_seen=include_seen)

    def best_observation(self) -> FeedbackObservation | None:
        if not self.machine.observations:
            return None
        return min(self.machine.observations, key=lambda item: item.rank)

    def simulate(
        self,
        oracle: SimilarityRankOracle,
        *,
        budget: int = 25,
        stop_rank: int = 1,
        seed_ids: list[str] | None = None,
    ) -> SimulationTrace:
        if budget <= 0:
            raise ValueError("budget must be positive")
        trace = SimulationTrace(target_id=oracle.target_id)
        seed_queue = list(seed_ids or [])

        for step in range(1, budget + 1):
            if seed_queue:
                candidate_id = seed_queue.pop(0)
            else:
                proposals = self.propose(k=1)
                if not proposals:
                    break
                candidate_id = proposals[0].candidate.id

            rank = oracle.rank(candidate_id)
            observation = self.observe(candidate_id, rank, metadata={"simulation_step": step})
            trace.guesses.append(observation)
            best = min(item.rank for item in trace.guesses)
            trace.best_rank_history.append(int(best))
            if rank <= stop_rank:
                trace.success_step = step
                break

        return trace
