# Paper Experiment Handoff: Rank-Guided Semantic Search Under Weak Ordinal Feedback

This document summarizes the current research framing, implemented methods, experimental setup, datasets, results, figures, and conclusions for the `contexto_search` project. It is written as a handoff for a future agent or author preparing a thesis, conference-style report, or journal manuscript.

The key message is that the project should not be presented only as a Contexto solver. Contexto-style semantic search is used as a controlled benchmark for a broader problem: search in a finite vectorized candidate space when the algorithm receives weak ordinal feedback rather than direct access to the hidden target.

## Repository State

Main repository:

```text
https://github.com/goga334/contexto_search
```

Important files:

```text
src/rgsn/strategies.py                Strategy interface and compared methods
src/rgsn/benchmark.py                 Multi-strategy benchmark runner and CLI
src/rgsn/report.py                    Plot and Markdown summary generator
src/rgsn/metrics.py                   Success, best-rank, and normalized AUC metrics
src/rgsn/oracle.py                    Hidden-target rank oracle for simulation
src/rgsn/index.py                     Vectorized large-vocabulary scoring index
src/rgsn/acquisition.py               Pairwise acquisition scoring formula
src/rgsn/direction.py                 Pairwise latent-direction learner
src/rgsn/constraints.py               Rank-to-pairwise-preference conversion
benchmarks/targets/glove24.txt        Smaller real-vocabulary target set
benchmarks/targets/glove60.txt        Main paper-style target set
docs/figures/glove24_benchmark/       GloVe-24 plots and summary
docs/figures/glove60_benchmark/       Main GloVe-60 plots and summary
tests/                                Fast unit and smoke tests
```

Current validation status at the time of this handoff:

```text
42 passed, 2 skipped
```

Only large local data and raw benchmark outputs are intentionally ignored by git:

```text
data/
results/
```

## Research Problem

Let:

```text
X = {x_1, ..., x_n}
phi(x_i) in R^d
```

where `X` is a finite candidate set and `phi(x_i)` is an embedding vector. A hidden target `x*` exists, but the search algorithm does not receive `phi(x*)` directly. At query step `t`, the algorithm proposes a candidate `x_t` and receives only ordinal rank feedback:

```text
r(x_t) in {1, ..., n}
```

Lower rank is better. Rank 1 is the hidden target or nearest item to the target under the benchmark oracle.

The observed history is:

```text
H_t = {(x_i, r_i)} for i <= t
```

The goal is to find candidates close to the hidden target within a small query budget. In the current semantic-search benchmark, "close" means achieving a rank at or below a fixed threshold, usually `top-25`.

The intended broader framing is:

```text
Weak-feedback search in finite vectorized spaces.
```

The Contexto-like game is only one instantiation. The same protocol can later be transferred to neural architecture search, design-space search, recommender search, hyperparameter search, or other domains where candidates have vector representations and feedback is ordinal, partial, noisy, expensive, or preference-based.

## Information Protocol

All implemented strategies follow the same protocol:

- A strategy sees only the observation history `H_t`.
- A strategy never receives the hidden target vector.
- A strategy does not receive future feedback.
- A strategy may score all unvisited candidates using the candidate embeddings.
- All strategies are evaluated with the same target set, seed words, budget, stop criterion, and random seed policy.
- Already seen candidates are excluded from future suggestions by default.
- Randomized methods support deterministic seeding and repeated runs.

The hidden-target oracle is used only by the benchmark runner to simulate rank feedback after a candidate is queried.

## Data

### Tiny Smoke Fixture

The tiny fixture is used only for development and tests:

```text
tests/fixtures/tiny_words.vec
tests/fixtures/tiny_targets.txt
tests/fixtures/tiny_dictionary.txt
```

It is not a meaningful scientific benchmark. It verifies that strategies, metrics, JSON/CSV exports, CLI commands, and plot generation work.

### Main GloVe Benchmark Data

The paper-style benchmark uses a local GloVe 50-dimensional embedding file:

```text
data/glove-wiki-gigaword-50.gz
```

This file is not committed to the repository. The benchmark expects it to exist locally.

The candidate dictionary is:

```text
data/glove_ascii_words.txt
```

It is produced by:

```bash
python scripts/build_alpha_dictionary.py data/glove-wiki-gigaword-50.gz data/glove_ascii_words.txt
```

Dictionary filtering keeps lowercase alphabetic ASCII words of minimum length 2. In the current local setup, the filtered benchmark store contains:

```text
317,730 candidates
```

This count is included in benchmark metadata.

### Target Sets

Two real-vocabulary target sets have been used:

```text
benchmarks/targets/glove24.txt
benchmarks/targets/glove60.txt
```

The main target set for paper-style reporting is `glove60.txt`. It contains 60 target words covering several broad semantic groups:

- nature and geography: `river`, `forest`, `ocean`, `mountain`, `island`, `desert`, `volcano`, `lake`
- weather and elements: `rain`, `snow`, `fire`
- institutions and social concepts: `city`, `election`, `hospital`, `university`, `church`, `police`, `government`
- professions and people: `doctor`, `teacher`, `lawyer`, `family`
- artifacts and materials: `computer`, `camera`, `engine`, `metal`, `glass`, `plastic`
- media and technology: `music`, `book`, `movie`, `phone`, `internet`, `software`, `hardware`, `television`, `newspaper`
- food and places: `coffee`, `bread`, `restaurant`, `market`, `hotel`, `kitchen`, `garden`
- transport and infrastructure: `airplane`, `airport`, `car`, `train`, `bridge`
- everyday categories: `dog`, `winter`, `summer`, `school`, `village`, `country`, `gold`, `medicine`, `painting`

All 60 targets and all shared seed words were validated to exist in the filtered GloVe candidate store.

Shared seed words for GloVe experiments:

```text
animal, tool, nature, emotion, object
```

These seed words are intentionally broad. They give each strategy an identical initial set of constraints without making the answer trivial.

## Hidden-Target Oracle

The oracle ranks candidates by cosine similarity to the hidden target vector. In code, the large-vocabulary oracle uses the vectorized `NumpyCandidateIndex`:

```text
similarity(x, x*) = phi(x) dot phi(x*)
```

The embedding loader stores normalized vectors, so dot product corresponds to cosine similarity.

For a target `x*`, the oracle sorts all candidates by decreasing similarity and assigns:

```text
rank 1 = most similar candidate
rank n = least similar candidate
```

During a benchmark, strategies do not see this ranking. They only receive the rank of candidates they queried.

## Compared Methods

All methods implement the common `SearchStrategy` interface:

```python
class SearchStrategy(Protocol):
    name: str

    def reset(self, store: CandidateStore, *, random_seed: int | None = None) -> None:
        ...

    def suggest(
        self,
        store: CandidateStore,
        observations: list[FeedbackObservation],
        k: int = 1,
    ) -> list[ScoredCandidate]:
        ...
```

### Random

File:

```text
src/rgsn/strategies.py
```

Name:

```text
random
```

Chooses uniformly random unseen candidates. It is deterministic when a fixed random seed is supplied.

Purpose in paper:

```text
Naive lower baseline.
```

### Best Neighbor

Name:

```text
best_neighbor
```

Finds the best observed candidate:

```text
x_best = argmin r(x_i), x_i in H_t
```

Then scores unseen candidates by cosine similarity to `x_best`:

```text
score(x) = cos(phi(x), phi(x_best))
```

Purpose in paper:

```text
Simple exploitation baseline: follow the best known clue.
```

### Centroid

Name:

```text
centroid
```

Default parameter:

```text
top_m = 3
```

Takes the top `m` observed candidates by rank, computes their mean embedding, normalizes it, and scores unseen candidates by similarity to that centroid:

```text
q = normalize(mean(phi(x_i)) for top-ranked observed x_i)
score(x) = cos(phi(x), q)
```

Purpose in paper:

```text
Aggregate positive-feedback baseline.
```

### Rocchio

Name:

```text
rocchio
```

Default parameters:

```text
top_m = 3
bottom_m = 3
alpha = 1.0
beta = 0.5
```

The strategy computes:

```text
q = alpha * mean(best observed) - beta * mean(worst observed)
```

Then scores unseen candidates by cosine similarity to `q`.

Purpose in paper:

```text
Classical relevance-feedback baseline adapted to ordinal semantic search.
```

### Pairwise Direction

Name:

```text
pairwise_direction
```

This is the first proposed rank-guided method.

Observed ranks induce pairwise preferences. For any two observations:

```text
r(x_i) < r(x_j) implies x_i preferred to x_j
```

The constraint builder creates:

```text
C_t = {(preferred_id, rejected_id, weight)}
```

where:

```text
weight = max(1.0, abs(rank_i - rank_j))
```

The method learns a latent search direction `q` such that:

```text
<q, phi(x_preferred) - phi(x_rejected)> > 0
```

The implementation uses an online gradient-style logistic update over pairwise constraints. Defaults:

```text
epochs = 80
learning_rate = 0.05
l2 = 0.001
normalize_output = True
```

The conceptual objective for paper text is:

```text
L(q) = sum log(1 + exp(-<q, phi(x_i) - phi(x_j)>)) + lambda ||q||^2
```

where `(i, j)` are preferred/rejected pairs.

After learning `q`, unseen candidates are scored by:

```text
score(x) = cos(phi(x), q)
```

Purpose in paper:

```text
Learn a latent semantic direction from ordinal rank comparisons rather than using only the single best clue.
```

### Pairwise Acquisition

Name:

```text
pairwise_acquisition
```

This is the strongest proposed method in current experiments.

It uses the same learned pairwise direction `q` as `pairwise_direction`, then combines exploitation and diversity terms:

```text
A(x) =
  alpha * direction_score(x)
  + beta * best_anchor(x)
  - gamma * redundancy(x, H_t)
  + delta * exploration(x, H_t)
```

Default weights:

```text
alpha = 1.00
beta = 0.35
gamma = 0.20
delta = 0.15
```

Components:

```text
direction_score(x) = cos(phi(x), q)
best_anchor(x) = cos(phi(x), phi(x_best))
redundancy(x, H_t) = max(0, max cos(phi(x), phi(x_i)) for x_i in H_t)
exploration(x, H_t) = 1 - redundancy(x, H_t)
```

Purpose in paper:

```text
Use learned pairwise direction while avoiding redundant guesses and retaining controlled exploration.
```

## Benchmark Protocol

Each benchmark run proceeds as follows:

1. Load embeddings and optional dictionary filter.
2. Build a candidate store and vectorized index.
3. For every target, build an oracle ranking from cosine similarity to the target.
4. For each strategy, each repeat, and each target:
   - reset the strategy with a deterministic seed;
   - query the shared seed words first, excluding the target if necessary;
   - after seed words are exhausted, let the strategy propose one unseen candidate per step;
   - query the oracle for that candidate's rank;
   - append `(candidate_id, rank)` to the observation history;
   - update best-rank history;
   - stop early if rank <= `stop_rank`.

Main GloVe-60 configuration:

```text
target_count = 60
candidate_count = 317,730
strategies = random, best_neighbor, centroid, rocchio, pairwise_direction, pairwise_acquisition
seed_words = animal, tool, nature, emotion, object
budget = 25
stop_rank = 25
repeats = 3
random_seed = 42
```

Because each of the 6 strategies is evaluated on 60 targets with 3 repeats, each strategy has:

```text
run_count = 180
```

## Reproduction Commands

### Build ASCII Dictionary

```bash
python scripts/build_alpha_dictionary.py data/glove-wiki-gigaword-50.gz data/glove_ascii_words.txt
```

### Run GloVe-60 Benchmark

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

### Generate Figures

```bash
python -m rgsn.report \
  --benchmark-json results/glove60_benchmark.json \
  --output-dir results/glove60_plots \
  --title "GloVe-60 Weak Feedback Benchmark"
```

### Run Tests

```bash
$env:PYTHONPATH="src"
python -m pytest -q
```

## Metrics

### Success Rate

A run is successful if it reaches:

```text
best_rank <= stop_rank
```

For the main benchmark:

```text
stop_rank = 25
```

Thus the reported success rate is `success@top-25`.

### Final Best Rank

The benchmark records the best rank achieved so far after every query. The final best rank is the last value in this history, padded for early stops when needed for aggregate curves.

Lower is better.

### Median Success Step

For successful runs only, this is the median query step at which the strategy first reaches `rank <= stop_rank`.

Important: the first five steps are fixed seed words in the GloVe experiments. Therefore, success steps above 5 reflect strategy-driven querying after the shared seed phase.

### Normalized AUC

Rank is lower-is-better, so rank history is converted to a higher-is-better score:

```text
score(rank) = (candidate_count - rank) / (candidate_count - 1)
```

Then the mean score over the query budget is reported as normalized AUC.

Because the candidate count is very large, normalized AUC values can look numerically close even when final ranks differ dramatically. For paper reporting, use normalized AUC as a secondary metric. Success rate, median final best rank, and best-rank curves are more interpretable.

## Main Results: GloVe-60

Source summary:

```text
docs/figures/glove60_benchmark/summary.md
```

Metadata:

```text
candidate_count = 317,730
target_count = 60
run_count_per_strategy = 180
budget = 25
stop_rank = 25
repeats = 3
random_seed = 42
seed_words = animal, tool, nature, emotion, object
```

Result table:

| Strategy | Success@Top-25 | Median Final Best Rank | Median Normalized AUC | Median Success Step |
|---|---:|---:|---:|---:|
| random | 0.017 | 1444.0 | 0.994 | 1 |
| best_neighbor | 0.267 | 179.0 | 0.997 | 13.5 |
| centroid | 0.100 | 226.0 | 0.997 | 13.5 |
| rocchio | 0.383 | 96.5 | 0.997 | 15 |
| pairwise_direction | 0.850 | 7.0 | 0.998 | 18 |
| pairwise_acquisition | 0.900 | 7.0 | 0.997 | 15.5 |

Recommended paper table ordering:

| Method Group | Strategy | Success@Top-25 | Median Rank |
|---|---|---:|---:|
| Naive | Random | 1.7% | 1444.0 |
| Local exploitation | Best Neighbor | 26.7% | 179.0 |
| Positive centroid | Centroid | 10.0% | 226.0 |
| Relevance feedback | Rocchio | 38.3% | 96.5 |
| Proposed | Pairwise Direction | 85.0% | 7.0 |
| Proposed | Pairwise Acquisition | 90.0% | 7.0 |

Main observation:

```text
The pairwise methods strongly outperform all baselines under the same weak ordinal feedback protocol.
```

The most concise claim supported by the data:

```text
On a 317,730-candidate GloVe semantic search benchmark with 60 targets and a 25-query budget, the proposed pairwise acquisition strategy achieved 90.0% success@top-25 and median final rank 7.0, compared with 38.3% and median rank 96.5 for Rocchio, the strongest non-pairwise baseline.
```

## Earlier Supporting Result: GloVe-24

Source summary:

```text
docs/figures/glove24_benchmark/summary.md
```

Configuration:

```text
candidate_count = 317,730
target_count = 24
run_count_per_strategy = 48
budget = 20
stop_rank = 25
repeats = 2
random_seed = 42
seed_words = animal, tool, nature, emotion, object
```

Result table:

| Strategy | Success@Top-25 | Median Final Best Rank | Median Normalized AUC | Median Success Step |
|---|---:|---:|---:|---:|
| random | 0.042 | 1647.5 | 0.994 | 1.0 |
| best_neighbor | 0.208 | 283.0 | 0.996 | 8.0 |
| centroid | 0.083 | 212.5 | 0.996 | 5.5 |
| rocchio | 0.417 | 57.0 | 0.997 | 13.0 |
| pairwise_direction | 0.500 | 38.5 | 0.997 | 14.5 |
| pairwise_acquisition | 0.667 | 16.0 | 0.997 | 14.5 |

Use this as a pilot or development result, not the main paper table.

## Figures

Main GloVe-60 figures:

```text
docs/figures/glove60_benchmark/best_rank_curve.png
docs/figures/glove60_benchmark/success_rate.png
docs/figures/glove60_benchmark/final_rank_distribution.png
```

Recommended use:

- Use `best_rank_curve.png` as the main dynamics figure.
- Use `success_rate.png` as a simple comparison figure.
- Use `final_rank_distribution.png` as a robustness/distribution figure if space allows.

The best-rank curve uses a logarithmic y-axis because rank is lower-is-better and spans several orders of magnitude.

GloVe-24 figures are available under:

```text
docs/figures/glove24_benchmark/
```

Tiny smoke-test figures are available under:

```text
docs/figures/tiny_benchmark/
```

Do not use the tiny smoke-test figures as scientific evidence.

## Interpretation

### Why Pairwise Methods Win

The non-pairwise baselines use only limited structure from the rank feedback:

- Best neighbor follows only the best observed candidate.
- Centroid aggregates positive examples but ignores negative evidence.
- Rocchio uses both good and bad observations but only through coarse means.

The pairwise methods use every observed rank relation. If one observed word has rank 200 and another has rank 2000, that induces a directional preference in embedding space. Over several observations, many such constraints accumulate:

```text
O(t^2) possible pairwise comparisons from t observations
```

This converts sparse weak feedback into a richer signal for learning a latent direction.

Pairwise acquisition further improves search by combining:

- latent direction exploitation;
- proximity to the best current anchor;
- penalty for redundant guesses;
- exploration pressure away from already queried points.

This explains why acquisition has higher success rate than pure pairwise direction in the GloVe-60 run:

```text
pairwise_acquisition: 90.0% success@top-25
pairwise_direction: 85.0% success@top-25
```

### Why Normalized AUC Looks Similar

The candidate set is very large. A rank improvement from 1444 to 7 is substantial for search quality, but under the normalized AUC transformation both ranks are close to the top of a 317,730-item list:

```text
score(1444) approx 0.9955
score(7) approx 0.99998
```

Therefore, normalized AUC is not the most visually discriminative metric here. The paper should emphasize:

- success@top-25;
- median final best rank;
- per-step median best-rank curves;
- final rank distribution.

## Suggested Paper Framing

Potential title direction:

```text
Rank-Guided Semantic Search in Vector Spaces Under Weak Ordinal Feedback
```

Alternative broader title:

```text
Pairwise Rank-Guided Search for Weak-Feedback Optimization in Finite Vector Spaces
```

Suggested abstract claim:

```text
We study search in finite vectorized candidate spaces where the learner receives only ordinal rank feedback for queried candidates. We introduce pairwise rank-guided strategies that convert observed ranks into preference constraints, learn a latent search direction, and use acquisition scoring to balance exploitation and exploration. In a Contexto-style semantic search benchmark over 317,730 GloVe word vectors, the proposed pairwise acquisition method achieves 90.0% success@top-25 over 60 targets within 25 queries, substantially outperforming random search, nearest-neighbor exploitation, centroid feedback, and Rocchio-style relevance feedback.
```

Suggested structure:

1. Introduction
   - Weak-feedback search problem.
   - Motivation from semantic search and future NAS.
   - Contributions: protocol, pairwise direction learner, acquisition strategy, benchmark.
2. Problem Definition
   - Finite candidate set.
   - Embedding function.
   - Hidden target.
   - Ordinal rank feedback.
   - Query budget and success criterion.
3. Methods
   - Baselines.
   - Pairwise constraints.
   - Latent direction learning.
   - Acquisition function.
4. Experimental Setup
   - GloVe data.
   - Dictionary filter.
   - Target set.
   - Seeds, budget, repeats, stop rank.
   - Metrics.
5. Results
   - Main table.
   - Best-rank curve.
   - Success rate chart.
   - Final-rank distribution.
6. Discussion
   - Why pairwise constraints help.
   - Limitations of semantic benchmark.
   - Transfer to NAS as future work.
7. Conclusion
   - Pairwise weak-feedback search is promising and reusable.

## Limitations To Mention

These limitations should be stated honestly:

1. The main benchmark uses a single embedding family, GloVe 50d.
2. Results may change with embedding model, vector dimension, or candidate filtering.
3. The target set is manually selected, although it is fixed and committed.
4. Repeats are currently 3 for the main GloVe-60 run. This is acceptable for a first paper-style result but more repeats would strengthen statistical claims.
5. Random search has stochasticity; deterministic strategies repeat the same behavior across repeats unless the shared protocol changes.
6. The Contexto-style oracle is a simulation based on cosine similarity, not the exact live Contexto game ranking.
7. NAS transfer is a motivation and future direction unless architecture-space experiments are added.
8. Hyperparameters for pairwise acquisition have not yet been ablated.

## Recommended Next Experiments

For a stronger publication, the next agent should consider:

1. Repeat GloVe-60 with 5 or 10 repeats.
2. Expand to 100 targets if runtime is acceptable.
3. Add a seed-count ablation:
   - 0 seed words;
   - 3 seed words;
   - 5 seed words;
   - 10 seed words.
4. Add budget curves:
   - budget 10;
   - budget 15;
   - budget 20;
   - budget 25;
   - budget 30.
5. Add acquisition ablations:
   - direction only;
   - direction + anchor;
   - direction + anchor - redundancy;
   - full acquisition.
6. Test another embedding model if available.
7. Add bootstrap confidence intervals or standard errors for success@top-25 and median rank.

The current codebase can support these experiments with the existing CLI. Some ablations may require exposing strategy hyperparameters through CLI arguments.

## Notes For The Report-Writing Agent

Use the GloVe-60 benchmark as the main experimental result. Use GloVe-24 only as a development/pilot result if needed.

Do not overstate novelty as a full new field. A safer claim is:

```text
The work combines ordinal rank feedback, pairwise preference learning, and acquisition scoring into a reusable weak-feedback search framework, evaluated on a Contexto-style semantic search benchmark.
```

Avoid claiming:

```text
State-of-the-art Contexto solver.
```

unless compared to existing Contexto-specific solvers, which has not been done.

Avoid claiming NAS results unless NAS experiments are actually added.

A good paper contribution statement:

```text
This work contributes a reusable benchmark protocol and implementation for comparing weak-feedback search strategies in vectorized candidate spaces, and shows that pairwise rank-guided methods substantially outperform several simple but relevant baselines on a large semantic-search benchmark.
```

## Compact Result Paragraph For Paper Draft

The following paragraph can be adapted directly:

```text
We evaluated six weak-feedback search strategies on a Contexto-style semantic benchmark using 317,730 filtered GloVe word vectors and 60 fixed target words. Each method received the same five initial seed queries and was allowed a total budget of 25 queries, with success defined as reaching a candidate ranked within the top 25 nearest neighbors of the hidden target. Over 180 runs per strategy, pairwise acquisition achieved the strongest performance, with 90.0% success@top-25 and median final best rank 7.0. The pairwise direction method also performed strongly, reaching 85.0% success@top-25 and median rank 7.0. In contrast, the strongest non-pairwise baseline, Rocchio-style relevance feedback, achieved 38.3% success@top-25 and median rank 96.5, while random search achieved only 1.7% success and median rank 1444.0. These results suggest that converting ordinal rank observations into pairwise preference constraints provides a substantially more informative learning signal than using only nearest-neighbor, centroid, or classical relevance-feedback heuristics.
```

