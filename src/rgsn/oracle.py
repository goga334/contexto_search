from __future__ import annotations

from dataclasses import dataclass, field

from rgsn.index import NumpyCandidateIndex
from rgsn.store import CandidateStore
from rgsn.types import Candidate


@dataclass(slots=True)
class SimilarityRankOracle:
    """Offline rank oracle for simulations over a vector candidate store."""

    store: CandidateStore
    target_id: str
    index: NumpyCandidateIndex | None = field(default=None, repr=False)
    _ranking: list[str] = field(init=False, repr=False)
    _ranks: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.store.get(self.target_id)
        index = self.index if self.index is not None else NumpyCandidateIndex.from_store(self.store)
        self._ranking, self._ranks = index.rank_oracle(self.target_id)

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
