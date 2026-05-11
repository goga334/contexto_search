from __future__ import annotations

from dataclasses import dataclass, field

from rgsn.store import CandidateStore
from rgsn.types import Candidate
from rgsn.vectors import cosine


@dataclass(slots=True)
class SimilarityRankOracle:
    """Offline rank oracle for simulations over a vector candidate store."""

    store: CandidateStore
    target_id: str
    _ranking: list[str] = field(init=False, repr=False)
    _ranks: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        target = self.store.get(self.target_id)
        scored = [
            (candidate.id, cosine(candidate.embedding, target.embedding))
            for candidate in self.store.values()
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        self._ranking = [candidate_id for candidate_id, _ in scored]
        self._ranks = {candidate_id: index + 1 for index, candidate_id in enumerate(self._ranking)}

    @property
    def target(self) -> Candidate:
        return self.store.get(self.target_id)

    def rank(self, candidate_id: str) -> int:
        self.store.get(candidate_id)
        return self._ranks[candidate_id]

    def top(self, k: int = 10) -> list[str]:
        if k <= 0:
            raise ValueError("k must be positive")
        return self._ranking[:k]
