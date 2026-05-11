from __future__ import annotations

import math
from dataclasses import dataclass

from rgsn.types import Candidate, PairwiseConstraint
from rgsn.vectors import Vector, add_scaled, dot, normalize, sub, zeros


@dataclass(frozen=True, slots=True)
class RankDirectionLearner:
    """Learns a latent search direction from pairwise rank constraints."""

    epochs: int = 80
    learning_rate: float = 0.05
    l2: float = 0.001
    normalize_output: bool = True

    def fit(self, candidates: dict[str, Candidate], constraints: list[PairwiseConstraint]) -> Vector | None:
        if not constraints:
            return None
        dim = len(next(iter(candidates.values())).embedding)
        q = self._initial_direction(candidates, constraints, dim)

        for _ in range(self.epochs):
            for constraint in constraints:
                preferred = candidates[constraint.preferred_id].embedding
                rejected = candidates[constraint.rejected_id].embedding
                diff = sub(preferred, rejected)
                margin = dot(q, diff)
                pressure = self._logistic_pressure(margin) * constraint.weight
                q = add_scaled(q, diff, self.learning_rate * pressure)
                if self.l2:
                    q = [value * (1.0 - self.learning_rate * self.l2) for value in q]

        return normalize(q) if self.normalize_output else q

    def _initial_direction(
        self,
        candidates: dict[str, Candidate],
        constraints: list[PairwiseConstraint],
        dim: int,
    ) -> Vector:
        q = zeros(dim)
        total_weight = 0.0
        for constraint in constraints:
            preferred = candidates[constraint.preferred_id].embedding
            rejected = candidates[constraint.rejected_id].embedding
            q = add_scaled(q, sub(preferred, rejected), constraint.weight)
            total_weight += constraint.weight
        if total_weight:
            q = [value / total_weight for value in q]
        return q

    def _logistic_pressure(self, margin: float) -> float:
        if margin > 30.0:
            return 0.0
        if margin < -30.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(margin))
