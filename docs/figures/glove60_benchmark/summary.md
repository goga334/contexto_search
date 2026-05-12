# GloVe-60 Weak Feedback Benchmark

## Metadata

- `target_ids`: ['river', 'forest', 'city', 'doctor', 'computer', 'music', 'football', 'airplane', 'ocean', 'mountain', 'coffee', 'bread', 'gold', 'election', 'science', 'hospital', 'kitchen', 'car', 'train', 'book', 'movie', 'phone', 'dog', 'winter', 'summer', 'teacher', 'university', 'airport', 'hotel', 'church', 'police', 'lawyer', 'painting', 'market', 'engine', 'garden', 'camera', 'medicine', 'family', 'school', 'village', 'country', 'government', 'internet', 'software', 'hardware', 'television', 'newspaper', 'restaurant', 'bridge', 'island', 'desert', 'volcano', 'lake', 'rain', 'snow', 'fire', 'metal', 'glass', 'plastic']
- `seed_ids`: ['animal', 'tool', 'nature', 'emotion', 'object']
- `budget`: 25
- `stop_rank`: 25
- `repeats`: 3
- `random_seed`: 42
- `candidate_count`: 317730

## Strategy Summary

| Strategy | Success Rate | Median Best Rank | Median AUC | Median Success Step |
|---|---:|---:|---:|---:|
| best_neighbor | 0.267 | 179.000 | 0.997 | 13.500 |
| centroid | 0.100 | 226.000 | 0.997 | 13.500 |
| pairwise_acquisition | 0.900 | 7.000 | 0.997 | 15.500 |
| pairwise_direction | 0.850 | 7.000 | 0.998 | 18 |
| random | 0.017 | 1444.000 | 0.994 | 1 |
| rocchio | 0.383 | 96.500 | 0.997 | 15 |
