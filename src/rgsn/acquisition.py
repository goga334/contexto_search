from __future__ import annotations

from dataclasses import dataclass

from rgsn.types import Candidate, FeedbackObservation, ScoredCandidate
from rgsn.vectors import Vector, cosine


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    alpha: float = 1.0
    beta: float = 0.35
    gamma: float = 0.20
    delta: float = 0.15


def score_candidates(
    *,
    candidates: dict[str, Candidate],
    observations: list[FeedbackObservation],
    direction: Vector | None,
    config: AcquisitionConfig,
    include_seen: bool = False,
) -> list[ScoredCandidate]:
    visited_ids = {observation.candidate_id for observation in observations}
    best = _best_observed_candidate(candidates, observations)
    observed_embeddings = [candidates[item.candidate_id].embedding for item in observations]
    scored: list[ScoredCandidate] = []

    for candidate in candidates.values():
        if not include_seen and candidate.id in visited_ids:
            continue

        direction_score = cosine(candidate.embedding, direction) if direction is not None else 0.0
        best_anchor = cosine(candidate.embedding, best.embedding) if best is not None else 0.0
        redundancy = _max_positive_similarity(candidate.embedding, observed_embeddings)
        exploration = 1.0 - redundancy if observed_embeddings else 1.0
        score = (
            config.alpha * direction_score
            + config.beta * best_anchor
            - config.gamma * redundancy
            + config.delta * exploration
        )
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                score=score,
                components={
                    "direction": direction_score,
                    "best_anchor": best_anchor,
                    "redundancy": redundancy,
                    "exploration": exploration,
                },
            )
        )

    return sorted(scored, key=lambda item: (-item.score, item.candidate.id))


def _best_observed_candidate(
    candidates: dict[str, Candidate],
    observations: list[FeedbackObservation],
) -> Candidate | None:
    if not observations:
        return None
    best_observation = min(observations, key=lambda item: item.rank)
    return candidates[best_observation.candidate_id]


def _max_positive_similarity(embedding: Vector, others: list[Vector]) -> float:
    if not others:
        return 0.0
    return max(0.0, max(cosine(embedding, other) for other in others))
