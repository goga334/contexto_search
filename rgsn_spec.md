# RGSN System Spec (All-in-One)

## TL;DR

We design a system for semantic search with weak rank feedback:

- target is hidden
- no relevance labels
- only ordinal feedback

Core idea:

Convert rank history -> pairwise constraints -> learn search direction -> guide next candidate.

---

## Problem Definition

X = {x1, ..., xn}, embeddings ei in R^d  
Hidden target x*

Feedback:

f_t = r(x_t)

---

## Core Idea

If r_i < r_j -> x_i better than x_j  
=> q · e_i > q · e_j

We learn q from history.

---

## Algorithm

Score:

S(x) = alpha sim(e_x, q) + beta sim(e_x, best) - gamma redundancy + delta exploration

---

## Minimal Implementation

The initial code lives in `src/rgsn/`:

- `types.py`: candidates, observations, constraints, scored candidates
- `constraints.py`: rank history -> pairwise constraints
- `direction.py`: pairwise logistic direction learner
- `acquisition.py`: direction/best/redundancy/exploration scoring
- `machine.py`: user-facing search-machine facade

---

## Metrics

- steps_to_target
- best_rank(t)
- success@T

---

## Architecture

User -> Candidate -> Feedback -> Constraints -> Direction -> Next candidate

The implementation is split into reusable layers:

- `CandidateStore`: owns any vectorized search space, not only words
- `WordDictionary`: optional word-domain filter for allowed guesses
- `RankGuidedSearchMachine`: converts feedback into constraints and proposals
- `WeakFeedbackSolver`: domain-neutral session facade
- `SimilarityRankOracle`: offline hidden-target simulator
- `EvaluationRunner`: multi-target benchmark runner
- `ContextoSolver`: word-game wrapper over the generic solver
- `rgsn.web`: FastAPI web UI for manual and simulated sessions

---

## Contribution

Rank feedback ~= weak gradient -> recover direction -> guide search.
