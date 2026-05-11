from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rgsn.vectors import Vector, as_vector


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    embedding: Vector
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "embedding", as_vector(self.embedding))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class FeedbackObservation:
    candidate_id: str
    rank: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "rank", float(self.rank))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PairwiseConstraint:
    preferred_id: str
    rejected_id: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    components: dict[str, float] = field(default_factory=dict)
