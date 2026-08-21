"""Index chunk records from ingest into a local ChromaDB collection."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "phonetics_chunks"
BATCH_SIZE = 100


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Файл чанков не найден: {chunks_path.resolve()}. "
            "Сначала запустите ingest.py."
        )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError(
            f"JSON пустой: {chunks_path.resolve()}. Нечего индексировать."
        )
    return chunks


def index_chunks(chunks: list[dict]) -> None:
    """Replace collection phonetics_chunks and index all chunk records."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    count = collection.count()
    if count != len(chunks):
        raise RuntimeError(
            f"Число записей в коллекции ({count}) не равно числу чанков ({len(chunks)})."
        )

    print(f"Путь к базе: {CHROMA_DIR.resolve()}")
    print(f"Имя коллекции: {COLLECTION_NAME}")
    print(f"Число записей collection.count(): {count}")
    print(f"Первые 3 id добавленных чанков: {ids[:3]}")


def main() -> None:
    chunks = load_chunks()
    index_chunks(chunks)


if __name__ == "__main__":
    main()
