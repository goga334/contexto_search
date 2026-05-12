# GloVe-24 Weak Feedback Benchmark

## Metadata

- `target_ids`: ['river', 'forest', 'city', 'doctor', 'computer', 'music', 'football', 'airplane', 'ocean', 'mountain', 'coffee', 'bread', 'gold', 'election', 'science', 'hospital', 'kitchen', 'car', 'train', 'book', 'movie', 'phone', 'dog', 'winter']
- `seed_ids`: ['animal', 'tool', 'nature', 'emotion', 'object']
- `budget`: 20
- `stop_rank`: 25
- `repeats`: 2
- `random_seed`: 42
- `candidate_count`: 317730

## Strategy Summary

| Strategy | Success Rate | Median Best Rank | Median AUC | Median Success Step |
|---|---:|---:|---:|---:|
| best_neighbor | 0.208 | 283.000 | 0.996 | 8.000 |
| centroid | 0.083 | 212.500 | 0.996 | 5.500 |
| pairwise_acquisition | 0.667 | 16.000 | 0.997 | 14.500 |
| pairwise_direction | 0.500 | 38.500 | 0.997 | 14.500 |
| random | 0.042 | 1647.500 | 0.994 | 1.000 |
| rocchio | 0.417 | 57.000 | 0.997 | 13.000 |
