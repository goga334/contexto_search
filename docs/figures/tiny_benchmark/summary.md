# Tiny RGSN Benchmark

## Metadata

- `target_ids`: ['river', 'forest', 'city']
- `seed_ids`: ['road', 'tree', 'water']
- `budget`: 6
- `stop_rank`: 2
- `repeats`: 1
- `random_seed`: 42
- `candidate_count`: 12

## Strategy Summary

| Strategy | Success Rate | Median Best Rank | Median AUC | Median Success Step |
|---|---:|---:|---:|---:|
| best_neighbor | 1.000 | 2 | 0.788 | 4 |
| centroid | 0.667 | 2 | 0.788 | 4.000 |
| pairwise_acquisition | 1.000 | 2 | 0.788 | 4 |
| pairwise_direction | 1.000 | 2 | 0.788 | 4 |
| random | 1.000 | 2 | 0.788 | 4 |
| rocchio | 1.000 | 1 | 0.788 | 5 |
