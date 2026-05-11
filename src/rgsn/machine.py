from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rgsn.acquisition import AcquisitionConfig, score_candidates
from rgsn.constraints import PairwiseConstraintBuilder
from rgsn.direction import RankDirectionLearner
from rgsn.types import Candidate, FeedbackObservation, PairwiseConstraint, ScoredCandidate
from rgsn.vectors import Vector


@dataclass(slots=True)
class RankGuidedSearchMachine:
    candidates: dict[str, Candidate]
    learner: RankDirectionLearner = field(default_factory=RankDirectionLearner)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    constraint_builder: PairwiseConstraintBuilder = field(default_factory=PairwiseConstraintBuilder)
    observations: list[FeedbackObservation] = field(default_factory=list)

    def __init__(
        self,
        candidates: Iterable[Candidate],
        *,
        learner: RankDirectionLearner | None = None,
        acquisition: AcquisitionConfig | None = None,
        constraint_builder: PairwiseConstraintBuilder | None = None,
    ) -> None:
        candidate_map = {candidate.id: candidate for candidate in candidates}
        if not candidate_map:
            raise ValueError("RankGuidedSearchMachine requires at least one candidate")
        self._validate_dimensions(candidate_map)
        self.candidates = candidate_map
        self.learner = learner if learner is not None else RankDirectionLearner()
        self.acquisition = acquisition if acquisition is not None else AcquisitionConfig()
        self.constraint_builder = constraint_builder if constraint_builder is not None else PairwiseConstraintBuilder()
        self.observations = []

    @classmethod
    def from_embeddings(cls, embeddings: Mapping[str, Sequence[float]], **kwargs: Any) -> RankGuidedSearchMachine:
        return cls([Candidate(id=item_id, embedding=list(embedding)) for item_id, embedding in embeddings.items()], **kwargs)

    def observe(self, candidate_id: str, rank: float, metadata: dict[str, Any] | None = None) -> FeedbackObservation:
        if candidate_id not in self.candidates:
            raise KeyError(f"Unknown candidate '{candidate_id}'")
        observation = FeedbackObservation(candidate_id=candidate_id, rank=rank, metadata=metadata or {})
        self.observations.append(observation)
        return observation

    def constraints(self) -> list[PairwiseConstraint]:
        return self.constraint_builder.build(self.observations)

    def direction(self) -> Vector | None:
        return self.learner.fit(self.candidates, self.constraints())

    def propose(self, k: int = 1, *, include_seen: bool = False) -> list[ScoredCandidate]:
        if k <= 0:
            raise ValueError("k must be positive")
        direction = self.direction()
        scored = score_candidates(
            candidates=self.candidates,
            observations=self.observations,
            direction=direction,
            config=self.acquisition,
            include_seen=include_seen,
        )
        return scored[:k]

    def state_summary(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.candidates),
            "observation_count": len(self.observations),
            "constraint_count": len(self.constraints()),
            "has_direction": self.direction() is not None,
        }

    def _validate_dimensions(self, candidates: dict[str, Candidate]) -> None:
        dims = {len(candidate.embedding) for candidate in candidates.values()}
        if len(dims) != 1:
            raise ValueError(f"All candidates must have the same embedding dimension, got {sorted(dims)}")
