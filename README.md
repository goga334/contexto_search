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
