from pathlib import Path

import pytest

from rgsn import (
    BestNeighborStrategy,
    CandidateStore,
    CentroidStrategy,
    PairwiseAcquisitionStrategy,
    PairwiseDirectionStrategy,
    RandomStrategy,
    RocchioStrategy,
)
from rgsn.types import FeedbackObservation


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_words.vec"


def _store() -> CandidateStore:
    return CandidateStore.from_text_file(FIXTURE)


def _observations() -> list[FeedbackObservation]:
    return [
        FeedbackObservation("road", rank=10),
        FeedbackObservation("water", rank=3),
        FeedbackObservation("tree", rank=20),
    ]


def _strategy_instances():
    return [
        RandomStrategy(),
        BestNeighborStrategy(),
        CentroidStrategy(),
        RocchioStrategy(),
        PairwiseDirectionStrategy(),
        PairwiseAcquisitionStrategy(),
    ]


def test_random_strategy_is_deterministic_with_fixed_seed() -> None:
    store = _store()
    first = RandomStrategy()
    second = RandomStrategy()

    first.reset(store, random_seed=7)
    second.reset(store, random_seed=7)

    assert [item.candidate.id for item in first.suggest(store, [], k=4)] == [
        item.candidate.id for item in second.suggest(store, [], k=4)
    ]


@pytest.mark.parametrize("strategy", _strategy_instances())
def test_strategy_suggests_from_empty_history(strategy) -> None:
    store = _store()
    strategy.reset(store, random_seed=5)

    suggestions = strategy.suggest(store, [], k=3)

    assert len(suggestions) == 3
    assert all(item.candidate.id in store.candidates for item in suggestions)


@pytest.mark.parametrize("strategy", _strategy_instances())
def test_strategy_avoids_seen_candidates(strategy) -> None:
    store = _store()
    observations = _observations()
    seen = {item.candidate_id for item in observations}
    strategy.reset(store, random_seed=5)

    suggestions = strategy.suggest(store, observations, k=5)

    assert suggestions
    assert all(item.candidate.id not in seen for item in suggestions)


def test_best_neighbor_moves_toward_best_observation() -> None:
    store = _store()
    strategy = BestNeighborStrategy()
    observations = [FeedbackObservation("water", rank=2), FeedbackObservation("road", rank=50)]

    suggestions = strategy.suggest(store, observations, k=3)

    assert suggestions[0].candidate.id in {"river", "stream", "lake", "boat"}


def test_pairwise_acquisition_uses_directional_feedback() -> None:
    store = _store()
    strategy = PairwiseAcquisitionStrategy()
    observations = [FeedbackObservation("water", rank=2), FeedbackObservation("road", rank=50)]

    suggestions = strategy.suggest(store, observations, k=3)

    assert suggestions[0].candidate.id in {"river", "stream", "lake", "boat"}
