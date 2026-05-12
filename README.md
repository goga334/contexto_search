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
- `SearchStrategy`: shared strategy interface for comparable weak-feedback methods.

Implemented strategy baselines:

- `RandomStrategy`
- `BestNeighborStrategy`
- `CentroidStrategy`
- `RocchioStrategy`
- `PairwiseDirectionStrategy`
- `PairwiseAcquisitionStrategy`

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

When a dictionary is provided, the embedding loader streams only matching words from the vector file. This is the intended path for large vocabularies: the candidate count is the dictionary/embedding overlap, not the raw dictionary size.

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

For a real-sized run, replace the fixture paths:

```bash
contexto-ui --embeddings data/wiki-news.vec --dictionary data/allowed_words.txt --port 8765
```

One practical GloVe setup:

```bash
python scripts/build_alpha_dictionary.py data/glove-wiki-gigaword-50.gz data/glove_ascii_words.txt
contexto-ui --embeddings data/glove-wiki-gigaword-50.gz --dictionary data/glove_ascii_words.txt --port 8765
```

## Benchmarks

Run a tiny reproducible smoke benchmark:

```bash
python -m rgsn.benchmark \
  --embeddings tests/fixtures/tiny_words.vec \
  --targets tests/fixtures/tiny_targets.txt \
  --strategies random,best_neighbor,centroid,rocchio,pairwise_direction,pairwise_acquisition \
  --seed-words road,tree,water \
  --budget 8 \
  --random-seed 42 \
  --out-json results/tiny_benchmark.json \
  --out-summary-csv results/tiny_summary.csv \
  --out-traces-csv results/tiny_traces.csv
```

Summary exports include success rate, final best-rank statistics, normalized AUC, and per-step mean/median best-rank curves.

Generate plots and a Markdown summary from a benchmark JSON:

```bash
python -m rgsn.report \
  --benchmark-json results/tiny_benchmark.json \
  --output-dir results/plots \
  --title "Tiny RGSN Benchmark"
```

Run the larger GloVe benchmark used for paper-style comparisons:

```bash
python -m rgsn.benchmark \
  --embeddings data/glove-wiki-gigaword-50.gz \
  --dictionary data/glove_ascii_words.txt \
  --targets benchmarks/targets/glove60.txt \
  --strategies random,best_neighbor,centroid,rocchio,pairwise_direction,pairwise_acquisition \
  --seed-words animal,tool,nature,emotion,object \
  --budget 25 \
  --stop-rank 25 \
  --repeats 3 \
  --random-seed 42 \
  --out-json results/glove60_benchmark.json \
  --out-summary-csv results/glove60_summary.csv \
  --out-traces-csv results/glove60_traces.csv
```

```bash
python -m rgsn.report \
  --benchmark-json results/glove60_benchmark.json \
  --output-dir results/glove60_plots \
  --title "GloVe-60 Weak Feedback Benchmark"
```

Reference GloVe-60 plots and the generated summary are committed under
`docs/figures/glove60_benchmark/`.
