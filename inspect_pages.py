"""Temporary page inspection for IPA/OCR quality checks. No Chroma/OpenAI."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz

PDF_PATH = Path("data/source/sokolova_practical_phonetics.pdf")
PAGES = (36, 43, 44, 46, 47, 49, 58, 59, 64, 79)
PREVIEW_CHARS = 8000
# Keywords / symbols to highlight (substring match, case-insensitive for Latin words).
MATCH_PATTERNS = (
    "articulation",
    "tongue",
    "lips",
    "vowel",
    "consonant",
    "θ",
    "t",
    "ɔ",
    "[9]",
    "[v]",
)


def line_matches(line: str) -> bool:
    lowered = line.lower()
    for pattern in MATCH_PATTERNS:
        if pattern.lower() in lowered:
            # Avoid matching every line just because of letter "t":
            # keep "t" only when it appears as a phoneme-like token.
            if pattern == "t":
                if re.search(r"(?<![A-Za-z])t(?![A-Za-z])|\[t\]|/t/", line, re.IGNORECASE):
                    return True
                continue
            return True
    return False


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF не найден: {PDF_PATH.resolve()}")

    document = fitz.open(PDF_PATH)
    try:
        for page_number in PAGES:
            index = page_number - 1
            if index < 0 or index >= document.page_count:
                print(f"=== Страница {page_number}: вне диапазона ===\n")
                continue

            page = document[index]
            text = page.get_text("text") or ""
            preview = text[:PREVIEW_CHARS]

            print("=" * 72)
            print(f"Страница: {page_number}")
            print("-" * 72)
            print(f"Первые {PREVIEW_CHARS} символов:")
            print(preview)
            print("-" * 72)
            print("Строки с совпадениями (articulation/tongue/lips/vowel/consonant/θ/t/ɔ/[9]/[v]):")
            matched_any = False
            for line in text.splitlines():
                if line_matches(line):
                    print(line)
                    matched_any = True
            if not matched_any:
                print("(совпадений не найдено)")
            print()
    finally:
        document.close()


if __name__ == "__main__":
    main()
