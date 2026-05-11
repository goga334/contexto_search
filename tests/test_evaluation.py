from pathlib import Path

from rgsn import CandidateStore, ContextoSolver, EvaluationRunner


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_words.vec"


def test_evaluation_runner_runs_multiple_targets() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    runner = EvaluationRunner(store)

    summary = runner.run(
        ["river", "tree"],
        budget=5,
        stop_rank=2,
        seed_ids=["road", "music", "water"],
    )

    assert summary.target_count == 2
    assert 0.0 <= summary.success_rate <= 1.0
    assert summary.mean_best_rank is not None
    assert all(case.steps_taken <= 5 for case in summary.cases)
    assert summary.to_dict()["target_count"] == 2


def test_evaluation_runner_filters_target_from_seed_ids() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    runner = EvaluationRunner(store)

    summary = runner.run(["river"], budget=3, seed_ids=["river", "road", "road", "water"])
    case = summary.cases[0]
    guessed_ids = [observation.candidate_id for observation in case.trace.guesses]

    assert guessed_ids[0] == "road"
    assert guessed_ids.count("road") == 1
    assert "river" not in guessed_ids[:2]


def test_contexto_solver_evaluates_word_targets() -> None:
    solver = ContextoSolver.from_embedding_file(FIXTURE)

    summary = solver.evaluate(
        ["river", "forest"],
        budget=5,
        stop_rank=3,
        seed_words=["road", "water", "tree"],
    )

    assert summary.target_count == 2
    assert summary.metadata["seed_ids"] == ["road", "water", "tree"]
