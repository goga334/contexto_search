from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rgsn.acquisition import AcquisitionConfig
from rgsn.store import CandidateStore
from rgsn.types import FeedbackObservation, ScoredCandidate
from rgsn.vectors import Vector


@dataclass(slots=True)
class NumpyCandidateIndex:
    """Vectorized scoring index for large candidate stores."""

    store: CandidateStore
    ids: list[str]
    matrix: np.ndarray
    id_to_row: dict[str, int]

    @classmethod
    def from_store(cls, store: CandidateStore) -> NumpyCandidateIndex:
        ids = list(store.candidates)
        matrix = np.asarray([store.candidates[item_id].embedding for item_id in ids], dtype=np.float32)
        return cls(
            store=store,
            ids=ids,
            matrix=matrix,
            id_to_row={item_id: index for index, item_id in enumerate(ids)},
        )

    def score_candidates(
        self,
        *,
        observations: list[FeedbackObservation],
        direction: Vector | None,
        config: AcquisitionConfig,
        k: int,
        include_seen: bool = False,
    ) -> list[ScoredCandidate]:
        if k <= 0:
            raise ValueError("k must be positive")
        if direction is None and not observations:
            return [
                ScoredCandidate(
                    candidate=self.store.candidates[item_id],
                    score=float(config.delta),
                    components={
                        "direction": 0.0,
                        "best_anchor": 0.0,
                        "redundancy": 0.0,
                        "exploration": 1.0,
                    },
                )
                for item_id in self.ids[:k]
            ]
        scores = np.full(len(self.ids), config.delta, dtype=np.float32)
        direction_scores = np.zeros(len(self.ids), dtype=np.float32)
        best_anchor_scores = np.zeros(len(self.ids), dtype=np.float32)
        redundancy_scores = np.zeros(len(self.ids), dtype=np.float32)
        exploration_scores = np.ones(len(self.ids), dtype=np.float32)

        if direction is not None:
            direction_vector = np.asarray(direction, dtype=np.float32)
            direction_scores = self.matrix @ direction_vector
            scores = scores + (config.alpha * direction_scores)

        if observations:
            best = min(observations, key=lambda item: item.rank)
            best_row = self.id_to_row[best.candidate_id]
            best_anchor_scores = self.matrix @ self.matrix[best_row]
            scores = scores + (config.beta * best_anchor_scores)

            observed_rows = [self.id_to_row[item.candidate_id] for item in observations]
            observed_matrix = self.matrix[observed_rows]
            redundancy_scores = np.maximum(0.0, np.max(self.matrix @ observed_matrix.T, axis=1))
            exploration_scores = 1.0 - redundancy_scores
            scores = (
                scores
                - (config.gamma * redundancy_scores)
                + (config.delta * exploration_scores)
                - config.delta
            )

            if not include_seen:
                scores[observed_rows] = -np.inf

        top_indices = self._top_indices(scores, k)
        return [
            ScoredCandidate(
                candidate=self.store.candidates[self.ids[index]],
                score=float(scores[index]),
                components={
                    "direction": float(direction_scores[index]),
                    "best_anchor": float(best_anchor_scores[index]),
                    "redundancy": float(redundancy_scores[index]),
                    "exploration": float(exploration_scores[index]),
                },
            )
            for index in top_indices
            if np.isfinite(scores[index])
        ]

    def score_by_vector(
        self,
        *,
        vector: Vector,
        observations: list[FeedbackObservation],
        k: int,
        component_name: str = "similarity",
        include_seen: bool = False,
    ) -> list[ScoredCandidate]:
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []
        query = query / query_norm
        scores = self.matrix @ query
        if observations and not include_seen:
            observed_rows = [self.id_to_row[item.candidate_id] for item in observations]
            scores[observed_rows] = -np.inf
        top_indices = self._top_indices(scores, k)
        return [
            ScoredCandidate(
                candidate=self.store.candidates[self.ids[index]],
                score=float(scores[index]),
                components={component_name: float(scores[index])},
            )
            for index in top_indices
            if np.isfinite(scores[index])
        ]

    def score_by_ids(
        self,
        *,
        candidate_ids: list[str],
        scores: list[float],
        component_name: str,
    ) -> list[ScoredCandidate]:
        return [
            ScoredCandidate(
                candidate=self.store.candidates[candidate_id],
                score=float(score),
                components={component_name: float(score)},
            )
            for candidate_id, score in zip(candidate_ids, scores)
        ]

    def rank_oracle(self, target_id: str) -> tuple[list[str], dict[str, int]]:
        target_row = self.id_to_row[target_id]
        similarities = self.matrix @ self.matrix[target_row]
        order = np.argsort(-similarities, kind="stable")
        ranking = [self.ids[int(index)] for index in order]
        ranks = {candidate_id: index + 1 for index, candidate_id in enumerate(ranking)}
        return ranking, ranks

    def _top_indices(self, scores: np.ndarray, k: int) -> list[int]:
        count = min(k, len(scores))
        if count == len(scores):
            indices = np.argsort(-scores, kind="stable")
        else:
            candidate_indices = np.argpartition(-scores, count - 1)[:count]
            indices = candidate_indices[np.argsort(-scores[candidate_indices], kind="stable")]
        return [int(index) for index in indices]
