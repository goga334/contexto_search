from rgsn import Candidate, PairwiseConstraintBuilder, RankDirectionLearner, RankGuidedSearchMachine
from rgsn.types import FeedbackObservation
from rgsn.vectors import cosine


def test_pairwise_constraints_treat_lower_rank_as_better() -> None:
    observations = [
        FeedbackObservation("bad", rank=50),
        FeedbackObservation("good", rank=3),
        FeedbackObservation("mid", rank=10),
    ]

    constraints = PairwiseConstraintBuilder().build(observations)
    pairs = {(item.preferred_id, item.rejected_id) for item in constraints}

    assert ("good", "bad") in pairs
    assert ("good", "mid") in pairs
    assert ("mid", "bad") in pairs


def test_direction_learner_recovers_preferred_axis() -> None:
    candidates = {
        "good": Candidate("good", [1.0, 0.0]),
        "bad": Candidate("bad", [-1.0, 0.0]),
        "off_axis": Candidate("off_axis", [0.0, 1.0]),
    }
    observations = [
        FeedbackObservation("good", rank=1),
        FeedbackObservation("bad", rank=20),
        FeedbackObservation("off_axis", rank=12),
    ]
    constraints = PairwiseConstraintBuilder().build(observations)

    direction = RankDirectionLearner(epochs=20).fit(candidates, constraints)

    assert direction is not None
    assert cosine(direction, [1.0, 0.0]) > 0.8


def test_search_machine_proposes_unseen_candidate_near_learned_direction() -> None:
    machine = RankGuidedSearchMachine(
        [
            Candidate("bad", [-1.0, 0.0]),
            Candidate("good_seen", [1.0, 0.0]),
            Candidate("good_unseen", [0.95, 0.05]),
            Candidate("off_axis", [0.0, 1.0]),
        ]
    )
    machine.observe("bad", rank=50)
    machine.observe("good_seen", rank=2)

    proposal = machine.propose(k=1)[0]

    assert proposal.candidate.id == "good_unseen"
    assert proposal.components["direction"] > proposal.components["redundancy"] - 0.1


def test_search_machine_rejects_mismatched_dimensions() -> None:
    try:
        RankGuidedSearchMachine([Candidate("a", [1.0]), Candidate("b", [1.0, 2.0])])
    except ValueError as exc:
        assert "same embedding dimension" in str(exc)
    else:
        raise AssertionError("Expected dimension validation error")
