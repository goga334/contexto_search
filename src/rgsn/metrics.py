from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, median


def success_at_rank(best_rank_history: Sequence[int], *, top_k: int) -> bool:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return bool(best_rank_history) and min(best_rank_history) <= top_k


def first_success_step(best_rank_history: Sequence[int], *, top_k: int) -> int | None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    for index, rank in enumerate(best_rank_history, start=1):
        if rank <= top_k:
            return index
    return None


def best_rank_at_budget(best_rank_history: Sequence[int], *, budget: int) -> int | None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not best_rank_history:
        return None
    index = min(budget, len(best_rank_history)) - 1
    return int(best_rank_history[index])


def normalized_best_rank_auc(
    best_rank_history: Sequence[int],
    *,
    candidate_count: int,
    budget: int | None = None,
) -> float | None:
    """Higher-is-better normalized area under best-rank progress.

    Rank is lower-is-better, so each rank is converted to a [0, 1] score:

    rank 1 -> 1.0
    rank candidate_count -> 0.0

    Missing tail steps are padded with the last known best rank so strategies
    that stop early remain comparable over the requested budget.
    """

    if candidate_count <= 1:
        raise ValueError("candidate_count must be greater than 1")
    if budget is not None and budget <= 0:
        raise ValueError("budget must be positive")
    if not best_rank_history:
        return None

    horizon = budget if budget is not None else len(best_rank_history)
    padded = pad_rank_history(best_rank_history, budget=horizon)
    scores = [
        (candidate_count - min(max(rank, 1), candidate_count)) / (candidate_count - 1)
        for rank in padded
    ]
    return mean(scores)


def pad_rank_history(best_rank_history: Sequence[int], *, budget: int) -> list[int]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not best_rank_history:
        return []
    values = [int(rank) for rank in best_rank_history[:budget]]
    if len(values) < budget:
        values.extend([values[-1]] * (budget - len(values)))
    return values


def per_step_mean_best_rank(histories: Sequence[Sequence[int]], *, budget: int) -> list[float | None]:
    padded = _padded_nonempty_histories(histories, budget=budget)
    if not padded:
        return [None for _ in range(budget)]
    return [mean(history[step] for history in padded) for step in range(budget)]


def per_step_median_best_rank(histories: Sequence[Sequence[int]], *, budget: int) -> list[float | None]:
    padded = _padded_nonempty_histories(histories, budget=budget)
    if not padded:
        return [None for _ in range(budget)]
    return [median(history[step] for history in padded) for step in range(budget)]


def _padded_nonempty_histories(histories: Sequence[Sequence[int]], *, budget: int) -> list[list[int]]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    return [pad_rank_history(history, budget=budget) for history in histories if history]
