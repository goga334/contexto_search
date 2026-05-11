from __future__ import annotations

from dataclasses import dataclass

from rgsn.types import FeedbackObservation, PairwiseConstraint


@dataclass(frozen=True, slots=True)
class PairwiseConstraintBuilder:
    """Converts ordinal rank observations into preference constraints.

    Lower rank means better, so rank 1 is preferred over rank 10.
    """

    min_rank_gap: float = 0.0
    max_constraints: int | None = None

    def build(self, observations: list[FeedbackObservation]) -> list[PairwiseConstraint]:
        constraints: list[PairwiseConstraint] = []
        for i, left in enumerate(observations):
            for right in observations[i + 1 :]:
                gap = abs(left.rank - right.rank)
                if gap <= self.min_rank_gap:
                    continue
                if left.rank < right.rank:
                    preferred, rejected = left, right
                else:
                    preferred, rejected = right, left
                constraints.append(
                    PairwiseConstraint(
                        preferred_id=preferred.candidate_id,
                        rejected_id=rejected.candidate_id,
                        weight=max(1.0, gap),
                        metadata={"rank_gap": gap},
                    )
                )

        if self.max_constraints is not None and len(constraints) > self.max_constraints:
            return constraints[-self.max_constraints :]
        return constraints
