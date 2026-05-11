from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Container

from rgsn.types import Candidate
from rgsn.vectors import Vector, normalize


@dataclass(frozen=True, slots=True)
class CandidateStore:
    """Reusable vector candidate store for word, architecture, or config spaces."""

    candidates: dict[str, Candidate]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("CandidateStore requires at least one candidate")
        dims = {len(candidate.embedding) for candidate in self.candidates.values()}
        if len(dims) != 1:
            raise ValueError(f"All candidates must have the same embedding dimension, got {sorted(dims)}")

    @classmethod
    def from_candidates(cls, candidates: Iterable[Candidate]) -> CandidateStore:
        return cls({candidate.id: candidate for candidate in candidates})

    @classmethod
    def from_mapping(
        cls,
        embeddings: Mapping[str, Sequence[float]],
        *,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
        normalize_embeddings: bool = True,
    ) -> CandidateStore:
        items = []
        metadata = metadata or {}
        for item_id, embedding in embeddings.items():
            vector = [float(value) for value in embedding]
            if normalize_embeddings:
                vector = normalize(vector)
            items.append(Candidate(str(item_id), vector, dict(metadata.get(item_id, {}))))
        return cls.from_candidates(items)

    @classmethod
    def from_text_file(
        cls,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        normalize_embeddings: bool = True,
        lowercase_ids: bool = False,
        allowed_ids: Container[str] | None = None,
        skip_header: bool | None = None,
        max_items: int | None = None,
    ) -> CandidateStore:
        """Load whitespace-separated vectors.

        Supports common text vector files:

        - `word 0.1 0.2 ...`
        - optional word2vec header: `vocab_size dimension`
        """

        source = Path(path)
        embeddings: dict[str, Vector] = {}
        with source.open("r", encoding=encoding) as handle:
            first_line = handle.readline()
            inferred_skip = _looks_like_vector_header(first_line)
            should_skip_first = inferred_skip if skip_header is None else skip_header
            if not should_skip_first:
                _parse_vector_line(first_line, embeddings, lowercase_ids=lowercase_ids, allowed_ids=allowed_ids)

            for line in handle:
                if max_items is not None and len(embeddings) >= max_items:
                    break
                _parse_vector_line(line, embeddings, lowercase_ids=lowercase_ids, allowed_ids=allowed_ids)

        return cls.from_mapping(embeddings, normalize_embeddings=normalize_embeddings)

    @property
    def dimension(self) -> int:
        return len(next(iter(self.candidates.values())).embedding)

    def ids(self) -> list[str]:
        return sorted(self.candidates)

    def values(self) -> list[Candidate]:
        return list(self.candidates.values())

    def get(self, candidate_id: str) -> Candidate:
        try:
            return self.candidates[candidate_id]
        except KeyError as exc:
            raise KeyError(f"Unknown candidate '{candidate_id}'") from exc

    def subset(self, candidate_ids: Iterable[str]) -> CandidateStore:
        return CandidateStore({candidate_id: self.get(candidate_id) for candidate_id in candidate_ids})

    def filter(self, predicate: Callable[[Candidate], bool]) -> CandidateStore:
        return CandidateStore(
            {
                candidate.id: candidate
                for candidate in self.candidates.values()
                if predicate(candidate)
            }
        )


def _looks_like_vector_header(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) != 2:
        return False
    return all(part.isdigit() for part in parts)


def _parse_vector_line(
    line: str,
    embeddings: dict[str, Vector],
    *,
    lowercase_ids: bool,
    allowed_ids: Container[str] | None,
) -> None:
    parts = line.strip().split()
    if not parts:
        return
    if len(parts) < 2:
        raise ValueError(f"Invalid vector line: {line!r}")
    item_id = parts[0].lower() if lowercase_ids else parts[0]
    if allowed_ids is not None and item_id not in allowed_ids:
        return
    try:
        embeddings[item_id] = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"Invalid numeric vector value in line: {line!r}") from exc
