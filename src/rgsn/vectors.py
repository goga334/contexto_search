from __future__ import annotations

import math
from collections.abc import Sequence


Vector = list[float]


def as_vector(values: Sequence[float]) -> Vector:
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError("Embedding vectors must not be empty")
    return vector


def check_same_dim(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError(f"Vector dimension mismatch: {len(left)} != {len(right)}")


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    check_same_dim(left, right)
    return sum(a * b for a, b in zip(left, right))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def sub(left: Sequence[float], right: Sequence[float]) -> Vector:
    check_same_dim(left, right)
    return [a - b for a, b in zip(left, right)]


def add_scaled(left: Sequence[float], right: Sequence[float], scale: float) -> Vector:
    check_same_dim(left, right)
    return [a + scale * b for a, b in zip(left, right)]


def normalize(vector: Sequence[float]) -> Vector:
    magnitude = norm(vector)
    if magnitude == 0.0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def mean(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        raise ValueError("Cannot average an empty vector list")
    dim = len(vectors[0])
    values = zeros(dim)
    for vector in vectors:
        check_same_dim(values, vector)
        values = [left + right for left, right in zip(values, vector)]
    return [value / len(vectors) for value in values]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    check_same_dim(left, right)
    denom = norm(left) * norm(right)
    if denom == 0.0:
        return 0.0
    return dot(left, right) / denom


def zeros(dim: int) -> Vector:
    if dim <= 0:
        raise ValueError("Vector dimension must be positive")
    return [0.0 for _ in range(dim)]
