from pathlib import Path

from rgsn import CandidateStore, ContextoSolver, SimilarityRankOracle, WeakFeedbackSolver


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_words.vec"


def test_candidate_store_loads_text_vectors_with_header() -> None:
    store = CandidateStore.from_text_file(FIXTURE)

    assert store.dimension == 3
    assert store.get("river").id == "river"
    assert "stream" in store.ids()


def test_similarity_rank_oracle_orders_by_target_similarity() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    oracle = SimilarityRankOracle(store, "river")

    assert oracle.rank("river") == 1
    assert oracle.rank("stream") < oracle.rank("tree")
    assert oracle.top(3) == ["river", "stream", "water"]


def test_weak_feedback_solver_simulation_improves_rank_with_seed_feedback() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    solver = WeakFeedbackSolver(store)
    oracle = SimilarityRankOracle(store, "river")

    trace = solver.simulate(oracle, budget=4, seed_ids=["road", "tree", "water"])

    assert trace.best_rank_history[0] > trace.best_rank_history[-1]
    assert trace.guesses[-1].candidate_id in {"river", "stream", "lake", "boat"}


def test_contexto_solver_wraps_word_search() -> None:
    solver = ContextoSolver.from_embedding_file(FIXTURE)

    solver.observe("road", 10)
    solver.observe("water", 3)
    suggestions = solver.suggest(k=3)

    assert suggestions
    assert suggestions[0].candidate.id in {"river", "stream", "lake", "boat"}


def test_contexto_solver_stream_filters_embeddings_with_dictionary() -> None:
    dictionary = Path(__file__).parent / "fixtures" / "tiny_dictionary.txt"

    solver = ContextoSolver.from_embedding_file(FIXTURE, dictionary_path=dictionary)

    assert len(solver.store.candidates) == 8
    assert "boat" not in solver.store.candidates
