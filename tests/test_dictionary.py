from pathlib import Path

from rgsn import CandidateStore, ContextoSolver, WordDictionary


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_words.vec"
DICTIONARY = Path(__file__).parent / "fixtures" / "tiny_dictionary.txt"


def test_word_dictionary_loads_and_filters_words() -> None:
    dictionary = WordDictionary.from_words(["River", "a", "city-2", "tree"], min_length=2)

    assert dictionary.sorted_words() == ["river", "tree"]


def test_word_dictionary_filters_candidate_store() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    dictionary = WordDictionary.from_text_file(DICTIONARY)

    filtered = dictionary.filter_store(store)
    coverage = dictionary.coverage(store)

    assert len(filtered.candidates) == 8
    assert "boat" not in filtered.candidates
    assert coverage.overlap_count == 8
    assert coverage.coverage == 1.0


def test_contexto_solver_can_use_dictionary_filter() -> None:
    solver = ContextoSolver.from_embedding_file(FIXTURE, dictionary_path=DICTIONARY)

    assert len(solver.store.candidates) == 8
    solver.observe("water", 3)
    suggestions = solver.suggest(k=10)

    assert all(item.candidate.id in solver.store.candidates for item in suggestions)
    assert all(item.candidate.id != "boat" for item in suggestions)
