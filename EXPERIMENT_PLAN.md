# Research Experiment Plan: Rank-Guided Search Benchmark

## Goal

Turn the current Contexto-style solver into a reusable experimental platform for comparing mathematical strategies of weak-feedback search in finite vectorized spaces.

The next conference thesis should not be framed as "a Contexto solver". Contexto-like semantic search should be used as a controlled benchmark for a broader problem:

```text
Given a finite candidate set X, embeddings phi(x), and a hidden target x*,
the algorithm observes only ordinal rank feedback for queried candidates.
The goal is to find candidates close to x* within a limited query budget.
```

This framing should make the work reusable for later Neural Architecture Search research, where candidates are neural architectures and the feedback may be partial, ordinal, noisy, expensive, or preference-based.

## Research Requirements

All implemented methods must follow the same information protocol:

- At step `t`, a strategy may use only the observed history:

```text
H_t = {(x_i, rank_i)}
```

- A strategy must not access the hidden target vector directly.
- A strategy must not use future feedback.
- A strategy may score all unvisited candidates using their embeddings.
- All strategies must run with the same target set, initial seeds, budget, and stop criterion.
- Randomized strategies must support fixed seeds and repeated runs.

The benchmark must be reproducible from CLI commands.

## Current Project Baseline

The project already contains useful building blocks:

- `CandidateStore`: vectorized candidate collection.
- `SimilarityRankOracle`: hidden-target rank simulator.
- `RankGuidedSearchMachine`: current pairwise rank-guided search facade.
- `PairwiseConstraintBuilder`: converts rank observations into pairwise preferences.
- `RankDirectionLearner`: learns a latent search direction from pairwise constraints.
- `score_candidates`: acquisition scoring with direction, best-anchor, redundancy, and exploration components.
- `EvaluationRunner`: runs simulations across multiple targets.
- `ContextoSolver`: word-search wrapper.
- `rgsn.web`: local UI for manual/simulated sessions.

Do not throw these away. Refactor them carefully into a strategy-based experiment pipeline.

## Implementation Plan

### 1. Add A Common Strategy Interface

Create a shared strategy abstraction, for example in:

```text
src/rgsn/strategies.py
```

Suggested shape:

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

Keep the interface small. The simulation runner should not need to know the internal mathematics of a strategy.

### 2. Implement Baseline Strategies

Implement simple baselines first. They are important for a credible comparison.

Required strategies:

- `RandomStrategy`
  - Chooses random unseen candidates.
  - Must support deterministic random seed.

- `BestNeighborStrategy`
  - Finds candidates most similar to the best observed candidate.
  - If there are no observations, falls back to random or a configured seed behavior.

- `CentroidStrategy`
  - Builds a centroid from the top `m` observed candidates by rank.
  - Scores candidates by cosine similarity to this centroid.

- `RocchioStrategy`
  - Uses a relevance-feedback style update:

```text
q = alpha * mean(best observed) - beta * mean(worst observed)
```

  - Then scores candidates by similarity to `q`.

### 3. Implement Pairwise Strategies

Refactor the current rank-guided implementation into strategies:

- `PairwiseDirectionStrategy`
  - Builds pairwise constraints from rank observations.
  - Learns latent direction `q` using `RankDirectionLearner`.
  - Scores candidates by similarity to `q`.

- `PairwiseAcquisitionStrategy`
  - Uses the current acquisition formula:

```text
A(x) =
  alpha * direction_score
  + beta * best_anchor
  - gamma * redundancy
  + delta * exploration
```

  - Exposes configurable weights.

Optional later strategies:

- `RankWeightedCentroidStrategy`
- `EpsilonGreedyBestNeighborStrategy`
- `UncertaintyAwarePairwiseStrategy`
- `PreferenceBayesianStrategy`

Keep optional strategies out of the critical path until the benchmark pipeline is solid.

### 4. Refactor Simulation Around Strategies

Add or update a strategy-driven runner, for example:

```text
src/rgsn/benchmark.py
```

It should support:

- one strategy on one target;
- one strategy on many targets;
- many strategies on the same target set;
- repeated runs for randomized strategies;
- shared budget and stop rank;
- shared seed words;
- trace collection for every run.

Each trace should store:

- strategy name;
- target id;
- run index;
- budget;
- stop rank;
- random seed;
- ordered guesses;
- rank after each guess;
- best rank history;
- success step, if any;
- strategy configuration metadata.

### 5. Add Metrics

Add metrics that are useful for thesis tables and plots:

- `success@T`
- `success@top_k`
- `mean_best_rank@T`
- `median_best_rank@T`
- `mean_success_step`
- `median_success_step`
- `AUC` or normalized area under the `best_rank_history` curve
- per-step mean/median best rank

Important: ranks are lower-is-better, so plots should either use a log scale or transform rank to a higher-is-better score. Be explicit in labels.

### 6. Add Result Export

Add JSON and CSV export.

Suggested files:

```text
results/contexto_benchmark.json
results/contexto_summary.csv
results/contexto_traces.csv
```

JSON should preserve full traces. CSV should be convenient for tables and plotting.

Do not commit large result files unless they are small smoke-test fixtures. Add appropriate ignore rules if needed.

### 7. Add CLI Commands

Add a reproducible CLI, either as a module or console script.

Suggested command:

```powershell
$env:PYTHONPATH="src"
python -m rgsn.benchmark `
  --embeddings data/words.vec `
  --targets data/targets.txt `
  --strategies random,best_neighbor,centroid,rocchio,pairwise,pairwise_acquisition `
  --seed-words road,water,tree `
  --budget 30 `
  --stop-rank 10 `
  --repeats 5 `
  --random-seed 42 `
  --out-json results/contexto_benchmark.json `
  --out-summary-csv results/contexto_summary.csv `
  --out-traces-csv results/contexto_traces.csv
```

Also support a small smoke-test command using `tests/fixtures/tiny_words.vec`.

### 8. Add Plotting

Add a plotting/report script, for example:

```text
src/rgsn/report.py
```

Required plots:

- median best rank over steps by strategy;
- success rate by strategy;
- final best rank distribution by strategy.

Suggested output:

```text
results/plots/best_rank_curve.png
results/plots/success_rate.png
results/plots/final_rank_boxplot.png
```

The thesis likely needs only 1-2 figures, but the code should be able to regenerate them.

### 9. Prepare A Contexto-Style Benchmark Dataset

Create a clear dataset protocol.

Needed inputs:

- embedding file;
- target list;
- optional seed word list;
- candidate filtering rules.

Target selection should avoid trivial or broken cases:

- target must exist in the embedding store;
- target should not be one of the seed words;
- target should have enough meaningful neighbors;
- targets should cover different semantic groups if possible.

For early development, use `tests/fixtures/tiny_words.vec`.

For thesis experiments, use a larger embedding file. Do not assume it is committed to the repository. Document where it should be placed and how the benchmark expects it.

### 10. Tests

Add focused tests for:

- every strategy avoids already seen candidates by default;
- randomized strategy is deterministic with a fixed seed;
- each strategy can produce suggestions after zero, one, and multiple observations;
- benchmark runner applies the same target/seed/budget protocol to all strategies;
- metrics are computed correctly for hand-made traces;
- JSON/CSV export contains required fields;
- CLI smoke test works on the tiny fixture.

Suggested test files:

```text
tests/test_strategies.py
tests/test_benchmark.py
tests/test_metrics.py
tests/test_benchmark_cli.py
```

Keep tests fast and independent from large external embedding files.

## Mathematical Content To Support In The Thesis

The implementation should support the following mathematical story.

### General Search Problem

```text
X = {x_1, ..., x_n},  phi(x_i) in R^d
```

There is a hidden target or hidden preference direction:

```text
x* or q*
```

The algorithm receives ordinal feedback:

```text
r(x_i) < r(x_j) means x_i is preferred to x_j
```

### Pairwise Preference Constraints

Observed ranks induce constraints:

```text
C_t = {(i, j): r(x_i) < r(x_j)}
```

The latent-direction model assumes:

```text
x_i preferred to x_j iff <q, phi(x_i) - phi(x_j)> > 0
```

### Logistic Pairwise Objective

The pairwise direction can be described as minimizing:

```text
L(q) = sum log(1 + exp(-<q, phi(x_i) - phi(x_j)>)) + lambda ||q||^2
```

### Acquisition Rule

The next candidate is selected by:

```text
x_{t+1} = argmax A(x; q_t, H_t)
```

with a general acquisition function:

```text
A(x) =
  alpha * direction(x)
  + beta * anchor_to_best(x)
  - gamma * redundancy(x, H_t)
  + delta * exploration(x, H_t)
```

These formulas should match the code closely enough that the thesis text and implementation tell the same story.

## Thesis Outline Supported By This Work

The next thesis can follow this structure:

1. Problem: search in vectorized candidate spaces under weak ordinal feedback.
2. Mathematical protocol: ranks, pairwise preferences, latent direction, acquisition.
3. Compared strategies: random, nearest-best, centroid, Rocchio, pairwise direction, pairwise acquisition.
4. Benchmark: Contexto-like hidden-target semantic search.
5. Metrics: success rate, best-rank dynamics, steps to top-k, final rank distribution.
6. Results: compare strategy behavior under one shared protocol.
7. Discussion: why this matters for future NAS work.

## Weak Points To Control

Address these explicitly in code and thesis text:

- Contexto is a benchmark, not the final research object.
- Results may depend on the embedding model.
- Strategies must receive equal information.
- Strong baselines are required.
- NAS transfer should be presented as motivation and future direction unless NAS experiments are actually run.
- Random methods need repeated runs.
- Large benchmarks need fixed target lists and fixed seeds.

## Definition Of Done

The project is ready for thesis writing when:

- all required strategies are implemented behind one interface;
- multi-strategy benchmark runs from CLI;
- smoke benchmark passes on the tiny fixture;
- full benchmark can export JSON and CSV;
- metrics are tested;
- at least two thesis-ready plots can be generated;
- README documents how to reproduce the experiment;
- tests pass with:

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```

## Suggested Work Order For The Next Agent

1. Inspect the existing `src/rgsn` modules and tests.
2. Add `SearchStrategy` and baseline strategies.
3. Refactor current pairwise logic into strategy classes.
4. Build the benchmark runner around strategies.
5. Add metrics and exports.
6. Add CLI.
7. Add tests.
8. Add plotting/report generation.
9. Update README with reproducible commands.
10. Run the tiny benchmark and test suite.

