from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rgsn.store import CandidateStore


@dataclass(frozen=True, slots=True)
class DictionaryCoverage:
    dictionary_size: int
    candidate_count: int
    overlap_count: int
    missing_count: int

    @property
    def coverage(self) -> float:
        if self.dictionary_size == 0:
            return 0.0
        return self.overlap_count / self.dictionary_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "dictionary_size": self.dictionary_size,
            "candidate_count": self.candidate_count,
            "overlap_count": self.overlap_count,
            "missing_count": self.missing_count,
            "coverage": self.coverage,
        }


@dataclass(frozen=True, slots=True)
class WordDictionary:
    """Allowed-word inventory for Contexto-style solving.

    The generic RGSN solver works with any candidate IDs. This class is the
    word-domain gate that decides which embedding IDs are valid guesses.
    """

    words: frozenset[str]

    @classmethod
    def from_words(
        cls,
        words: Iterable[str],
        *,
        lowercase: bool = True,
        alpha_only: bool = True,
        min_length: int | None = 2,
        max_length: int | None = None,
    ) -> WordDictionary:
        cleaned = []
        for word in words:
            value = _normalize_word(word, lowercase=lowercase)
            if _accept_word(value, alpha_only=alpha_only, min_length=min_length, max_length=max_length):
                cleaned.append(value)
        return cls(frozenset(cleaned))

    @classmethod
    def from_text_file(
        cls,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        lowercase: bool = True,
        alpha_only: bool = True,
        min_length: int | None = 2,
        max_length: int | None = None,
    ) -> WordDictionary:
        source = Path(path)
        words = []
        with source.open("r", encoding=encoding) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                words.append(stripped.split()[0])
        return cls.from_words(
            words,
            lowercase=lowercase,
            alpha_only=alpha_only,
            min_length=min_length,
            max_length=max_length,
        )

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word in self.words

    def __len__(self) -> int:
        return len(self.words)

    def sorted_words(self) -> list[str]:
        return sorted(self.words)

    def filter_store(self, store: CandidateStore) -> CandidateStore:
        return store.filter(lambda candidate: candidate.id in self.words)

    def missing_from_store(self, store: CandidateStore) -> list[str]:
        return sorted(word for word in self.words if word not in store.candidates)

    def coverage(self, store: CandidateStore) -> DictionaryCoverage:
        overlap_count = sum(1 for word in self.words if word in store.candidates)
        return DictionaryCoverage(
            dictionary_size=len(self.words),
            candidate_count=len(store.candidates),
            overlap_count=overlap_count,
            missing_count=len(self.words) - overlap_count,
        )


def _normalize_word(word: str, *, lowercase: bool) -> str:
    value = word.strip()
    return value.lower() if lowercase else value


def _accept_word(
    word: str,
    *,
    alpha_only: bool,
    min_length: int | None,
    max_length: int | None,
) -> bool:
    if not word:
        return False
    if alpha_only and not word.isalpha():
        return False
    if min_length is not None and len(word) < min_length:
        return False
    if max_length is not None and len(word) > max_length:
        return False
    return True
