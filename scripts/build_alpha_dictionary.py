from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an alphabetic word list from a text embedding file.")
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--allow-unicode", action="store_true")
    args = parser.parse_args()

    words: set[str] = set()
    opener = gzip.open if args.embeddings.suffix == ".gz" else open
    with opener(args.embeddings, mode="rt", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
        if not _looks_like_header(first):
            _add_word(first, words, min_length=args.min_length, ascii_only=not args.allow_unicode)
        for line in handle:
            _add_word(line.strip().split(maxsplit=1), words, min_length=args.min_length, ascii_only=not args.allow_unicode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(words)), encoding="utf-8")
    print(f"Wrote {len(words)} words to {args.output}")


def _looks_like_header(parts: list[str]) -> bool:
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def _add_word(parts: list[str], words: set[str], *, min_length: int, ascii_only: bool) -> None:
    if not parts:
        return
    word = parts[0].lower()
    if len(word) >= min_length and word.isalpha() and (not ascii_only or word.isascii()):
        words.add(word)


if __name__ == "__main__":
    main()
