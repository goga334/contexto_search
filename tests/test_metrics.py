from rgsn.metrics import (
    best_rank_at_budget,
    first_success_step,
    normalized_best_rank_auc,
    pad_rank_history,
    per_step_mean_best_rank,
    per_step_median_best_rank,
    success_at_rank,
)


def test_success_metrics_for_rank_history() -> None:
    history = [100, 50, 7, 3]

    assert success_at_rank(history, top_k=10) is True
    assert success_at_rank(history, top_k=2) is False
    assert first_success_step(history, top_k=10) == 3
    assert first_success_step(history, top_k=2) is None
    assert best_rank_at_budget(history, budget=2) == 50
    assert best_rank_at_budget(history, budget=10) == 3


def test_normalized_auc_is_higher_for_better_rank_curve() -> None:
    weak = normalized_best_rank_auc([100, 80, 60], candidate_count=100, budget=3)
    strong = normalized_best_rank_auc([100, 20, 1], candidate_count=100, budget=3)

    assert weak is not None
    assert strong is not None
    assert strong > weak


def test_rank_histories_are_padded_for_shared_budget() -> None:
    assert pad_rank_history([10, 5], budget=4) == [10, 5, 5, 5]


def test_per_step_aggregate_rank_curves() -> None:
    histories = [[10, 5, 1], [20, 10]]

    assert per_step_mean_best_rank(histories, budget=3) == [15, 7.5, 5.5]
    assert per_step_median_best_rank(histories, budget=3) == [15.0, 7.5, 5.5]
