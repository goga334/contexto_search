# Contexto Search

Small research sandbox for Rank-Guided Search Networks (RGSN): a weak-feedback search loop that converts ordinal rank history into pairwise constraints, learns a latent direction, and proposes the next candidate.

The first implementation is intentionally lightweight:

- pure Python vector math,
- pairwise rank constraints,
- logistic direction learning,
- acquisition scoring with direction, best-anchor, redundancy, and exploration terms.

```python
from rgsn import Candidate, RankGuidedSearchMachine

candidates = [
    Candidate(id="a", embedding=[1.0, 0.0]),
    Candidate(id="b", embedding=[0.8, 0.2]),
    Candidate(id="c", embedding=[0.0, 1.0]),
]

machine = RankGuidedSearchMachine(candidates)
machine.observe("c", rank=20)
machine.observe("a", rank=3)

proposal = machine.propose(k=1)[0]
print(proposal.candidate.id, proposal.score)
```

## Contexto-Style Word Search

```python
from rgsn import ContextoSolver

solver = ContextoSolver.from_embedding_file("data/words.vec")
solver.observe("road", rank=1200)
solver.observe("water", rank=42)

for suggestion in solver.suggest(k=5):
    print(suggestion.candidate.id, suggestion.score)
```

The word solver is a thin wrapper over the reusable vector search components:

- `CandidateStore`: stores any vectorized search space.
- `WeakFeedbackSolver`: observes weak ordinal feedback and proposes candidates.
- `SimilarityRankOracle`: simulates hidden-target games for offline evaluation.
- `EvaluationRunner`: runs many hidden-target simulations and aggregates metrics.

```python
summary = solver.evaluate(
    ["river", "forest", "city"],
    budget=30,
    seed_words=["road", "water", "tree"],
)
print(summary.to_dict())
```

To restrict guesses to an allowed word list:

```python
solver = ContextoSolver.from_embedding_file(
    "data/words.vec",
    dictionary_path="data/allowed_words.txt",
)
```

## Local Web UI

```bash
contexto-ui --embeddings tests/fixtures/tiny_words.vec --dictionary tests/fixtures/tiny_dictionary.txt --port 8765
```

During local source development:

```bash
$env:PYTHONPATH="src"
python -m rgsn.web --embeddings tests/fixtures/tiny_words.vec --port 8765
```

Open `http://127.0.0.1:8765`.
