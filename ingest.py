"""Extract PDF pages, clean text, and build chunk records with metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf as fitz

PDF_PATH = Path("data/source/sokolova_practical_phonetics.pdf")
CHUNKS_PATH = Path("data/processed/chunks.json")
CHUNK_SIZE = 1200
OVERLAP = 200
FIGURE_RE = re.compile(
    r"\b(?:fig\.|figure)\s*(?:\d+|[IVXLC]+)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Minimize whitespace cleanup without removing IPA symbols."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """Split one page text into overlapping chunks. Never mixes pages."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def page_has_figure(text: str) -> bool:
    """True if page text contains a Fig./Figure caption with a number."""
    return bool(FIGURE_RE.search(text))


def build_chunk_records(pdf_path: Path) -> tuple[list[dict], dict]:
    """Build chunk records and collect summary stats."""
    document = fitz.open(pdf_path)
    source_file = pdf_path.name
    page_count = document.page_count
    records: list[dict] = []
    total_chars = 0
    pages_with_text = 0
    figure_pages: list[int] = []

    try:
        for index, page in enumerate(document):
            page_number = index + 1
            raw_text = page.get_text("text") or ""
            has_figure = page_has_figure(raw_text)
            cleaned = clean_text(raw_text)

            if has_figure:
                figure_pages.append(page_number)
            if cleaned:
                pages_with_text += 1
                total_chars += len(cleaned)

            page_chunks = split_into_chunks(cleaned, CHUNK_SIZE, OVERLAP)
            for chunk_index, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"p{page_number:03d}_c{chunk_index:03d}"
                records.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source_file": source_file,
                            "page_number": page_number,
                            "chunk_id": chunk_id,
                            "has_figure": has_figure,
                        },
                    }
                )
    finally:
        document.close()

    stats = {
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "total_chars": total_chars,
        "figure_pages": figure_pages,
    }
    return records, stats


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF не найден: {PDF_PATH.resolve()}")

    records, stats = build_chunk_records(PDF_PATH)

    if not records:
        raise RuntimeError(
            "После обработки не создано ни одного чанка. "
            "Проверьте содержимое PDF и параметры chunking."
        )

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    print(f"Число страниц: {stats['page_count']}")
    print(f"Число страниц с непустым текстом: {stats['pages_with_text']}")
    print(f"Общее число символов: {stats['total_chars']}")
    print(f"Число созданных чанков: {len(records)}")
    print(f"Номера страниц, где has_figure=True: {stats['figure_pages']}")
    print(f"Страницы с Figure-маркером: {stats['figure_pages']}")
    print(f"Количество страниц с Figure-маркером: {len(stats['figure_pages'])}")

    example = records[0]
    print("\nПример чанка:")
    print(f"id: {example['id']}")
    print("metadata:")
    print(json.dumps(example["metadata"], ensure_ascii=False, indent=2))
    print("text (первые 700 символов):")
    print(example["text"][:700])
    print(f"\nСохранено: {CHUNKS_PATH.resolve()}")


if __name__ == "__main__":
    main()
