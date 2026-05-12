from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rgsn.acquisition import AcquisitionConfig
from rgsn.constraints import PairwiseConstraintBuilder
from rgsn.direction import RankDirectionLearner
from rgsn.index import NumpyCandidateIndex
from rgsn.store import CandidateStore
from rgsn.types import FeedbackObservation, ScoredCandidate
from rgsn.vectors import Vector, normalize


class SearchStrategy(Protocol):
    name: str

    def reset(self, store: CandidateStore, *, random_seed: int | None = None) -> None:
        ...

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        ...


@dataclass(slots=True)
class StrategyBase:
    name: str
    index: NumpyCandidateIndex | None = field(default=None, init=False, repr=False)
    rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def reset(self, store: CandidateStore, *, random_seed: int | None = None) -> None:
        self.index = NumpyCandidateIndex.from_store(store)
        self.rng = random.Random(random_seed)

    def _index(self, store: CandidateStore) -> NumpyCandidateIndex:
        if self.index is None or self.index.store is not store:
            self.reset(store)
        if self.index is None:
            raise RuntimeError("Strategy index was not initialized")
        return self.index

    def _unseen_ids(self, store: CandidateStore, observations: list[FeedbackObservation]) -> list[str]:
        seen = {item.candidate_id for item in observations}
        return [candidate_id for candidate_id in store.candidates if candidate_id not in seen]


@dataclass(slots=True)
class RandomStrategy(StrategyBase):
    name: str = "random"

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        if k <= 0:
            raise ValueError("k must be positive")
        index = self._index(store)
        unseen = self._unseen_ids(store, observations)
        sample_size = min(k, len(unseen))
        sampled = self.rng.sample(unseen, k=sample_size) if sample_size else []
        return index.score_by_ids(candidate_ids=sampled, scores=[0.0] * len(sampled), component_name="random")


@dataclass(slots=True)
class BestNeighborStrategy(StrategyBase):
    name: str = "best_neighbor"

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        index = self._index(store)
        if not observations:
            return _random_fallback(store, observations, k)
        best = min(observations, key=lambda item: item.rank)
        return index.score_by_vector(
            vector=store.get(best.candidate_id).embedding,
            observations=observations,
            k=k,
            component_name="best_neighbor",
        )


@dataclass(slots=True)
class CentroidStrategy(StrategyBase):
    top_m: int = 3
    name: str = "centroid"

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        index = self._index(store)
        if not observations:
            return _random_fallback(store, observations, k)
        centroid = _mean_embedding(store, _top_observations(observations, self.top_m))
        return index.score_by_vector(vector=centroid, observations=observations, k=k, component_name="centroid")


@dataclass(slots=True)
class RocchioStrategy(StrategyBase):
    top_m: int = 3
    bottom_m: int = 3
    alpha: float = 1.0
    beta: float = 0.5
    name: str = "rocchio"

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        index = self._index(store)
        if not observations:
            return _random_fallback(store, observations, k)
        best_mean = np.asarray(_mean_embedding(store, _top_observations(observations, self.top_m)), dtype=np.float32)
        worst_mean = np.asarray(_mean_embedding(store, _bottom_observations(observations, self.bottom_m)), dtype=np.float32)
        query = self.alpha * best_mean - self.beta * worst_mean
        return index.score_by_vector(vector=query.tolist(), observations=observations, k=k, component_name="rocchio")


@dataclass(slots=True)
class PairwiseDirectionStrategy(StrategyBase):
    learner: RankDirectionLearner = field(default_factory=RankDirectionLearner)
    constraint_builder: PairwiseConstraintBuilder = field(default_factory=PairwiseConstraintBuilder)
    name: str = "pairwise_direction"

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        index = self._index(store)
        direction = self._direction(store, observations)
        if direction is None:
            return _random_fallback(store, observations, k)
        return index.score_by_vector(vector=direction, observations=observations, k=k, component_name="direction")

    def _direction(self, store: CandidateStore, observations: list[FeedbackObservation]) -> Vector | None:
        constraints = self.constraint_builder.build(observations)
        return self.learner.fit(store.candidates, constraints)


@dataclass(slots=True)
class PairwiseAcquisitionStrategy(PairwiseDirectionStrategy):
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    name: str = "pairwise_acquisition"

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        index = self._index(store)
        return index.score_candidates(
            observations=observations,
            direction=self._direction(store, observations),
            config=self.acquisition,
            k=k,
        )


def _random_fallback(store: CandidateStore, observations: list[FeedbackObservation], k: int) -> list[ScoredCandidate]:
    strategy = RandomStrategy()
    strategy.reset(store, random_seed=0)
    return strategy.suggest(store, observations, k=k)


def _top_observations(observations: list[FeedbackObservation], count: int) -> list[FeedbackObservation]:
    return sorted(observations, key=lambda item: item.rank)[: max(1, count)]


def _bottom_observations(observations: list[FeedbackObservation], count: int) -> list[FeedbackObservation]:
    return sorted(observations, key=lambda item: item.rank, reverse=True)[: max(1, count)]


def _mean_embedding(store: CandidateStore, observations: list[FeedbackObservation]) -> Vector:
    vectors = [store.get(item.candidate_id).embedding for item in observations]
    mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    return normalize(mean.tolist())
