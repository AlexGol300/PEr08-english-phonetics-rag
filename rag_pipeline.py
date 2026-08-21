"""RAG pipeline: parse IPA input, retrieve Chroma contexts, generate answers."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "phonetics_chunks"
CACHE_DB_PATH = Path("data/cache.sqlite3")

# Explicit British RP phoneme inventory for longest-match parsing.
# Source list must stay unique; matching always uses SORTED_PHONEMES.
_RP_PHONEMES_RAW: tuple[str, ...] = (
    "tʃ",
    "dʒ",
    "iː",
    "ɜː",
    "ɔː",
    "uː",
    "ɑː",
    "eɪ",
    "aɪ",
    "ɔɪ",
    "əʊ",
    "aʊ",
    "ɪə",
    "eə",
    "ʊə",
    "p",
    "b",
    "m",
    "w",
    "f",
    "v",
    "θ",
    "ð",
    "t",
    "d",
    "s",
    "z",
    "n",
    "l",
    "r",
    "ʃ",
    "ʒ",
    "j",
    "k",
    "g",
    "ŋ",
    "h",
    "ɪ",
    "e",
    "æ",
    "ʌ",
    "ɒ",
    "ə",
)

# Unique RP set (dict.fromkeys preserves order, drops duplicates).
RP_PHONEMES: tuple[str, ...] = tuple(dict.fromkeys(_RP_PHONEMES_RAW))
if len(RP_PHONEMES) != len(_RP_PHONEMES_RAW):
    raise RuntimeError("RP_PHONEMES содержит дубликаты — исправьте исходный список.")

# Critical: longer phonemes must be tried before shorter ones.
SORTED_PHONEMES: tuple[str, ...] = tuple(
    sorted(RP_PHONEMES, key=len, reverse=True)
)

# MVP phonemes validated against PDF pages after retrieval diagnostics (TZ v1.3).
SUPPORTED_PHONEMES: dict[str, dict[str, object]] = {
    "p": {"place": "bilabial", "validated_pages": [36]},
    "f": {"place": "labiodental", "validated_pages": [44]},
    "v": {"place": "labiodental", "validated_pages": [44, 79]},
    "z": {"place": "alveolar", "validated_pages": [47]},
}

STRESS_MARKS = ("ˈ", "ˌ")

IPA_INTERNAL_SPACE_ERROR = (
    "Ошибка формата IPA: внутри транскрипции не должно быть пробелов.\n"
    "Используйте, например: five | /faɪv/"
)

PLACE_ORDER: tuple[str, ...] = (
    "bilabial",
    "labiodental",
    "dental",
    "alveolar",
    "postalveolar",
    "palatal",
    "velar",
    "glottal",
)

# Static technical place classification (not taken from the PDF).
PHONEME_PLACE: dict[str, str] = {
    # bilabial
    "p": "bilabial",
    "b": "bilabial",
    "m": "bilabial",
    "w": "bilabial",
    # labiodental
    "f": "labiodental",
    "v": "labiodental",
    # dental
    "θ": "dental",
    "ð": "dental",
    # alveolar
    "t": "alveolar",
    "d": "alveolar",
    "s": "alveolar",
    "z": "alveolar",
    "n": "alveolar",
    "l": "alveolar",
    "r": "alveolar",
    # postalveolar
    "ʃ": "postalveolar",
    "ʒ": "postalveolar",
    "tʃ": "postalveolar",
    "dʒ": "postalveolar",
    # palatal
    "j": "palatal",
    # velar
    "k": "velar",
    "g": "velar",
    "ŋ": "velar",
    # glottal
    "h": "glottal",
    # vowels
    "iː": "vowel",
    "ɜː": "vowel",
    "ɔː": "vowel",
    "uː": "vowel",
    "ɑː": "vowel",
    "eɪ": "vowel",
    "aɪ": "vowel",
    "ɔɪ": "vowel",
    "əʊ": "vowel",
    "aʊ": "vowel",
    "ɪə": "vowel",
    "eə": "vowel",
    "ʊə": "vowel",
    "ɪ": "vowel",
    "e": "vowel",
    "æ": "vowel",
    "ʌ": "vowel",
    "ɒ": "vowel",
    "ə": "vowel",
}

PLACE_RANK: dict[str, int] = {place: index for index, place in enumerate(PLACE_ORDER)}

SYSTEM_PROMPT = """Ты — ассистент по практической фонетике английского языка.
Рабочий стандарт произношения: British RP.

Твоя задача — сформировать подробный учебный ответ на русском языке
только на основе переданного контекста из PDF-учебника.

Тебе передаются:
- слово пользователя;
- IPA-транскрипция, предоставленная пользователем;
- конкретная фонема, для которой уже подтверждено,
  что она входит в валидированный набор MVP;
- место образования фонемы, если оно определено статическим кодом;
- найденные текстовые чанки на английском, прошедшие проверку
  критерия валидации retrieval;
- номер страницы для каждого чанка;
- признак has_figure для каждого чанка.

Если тебе передан явный сигнал «фонема не поддерживается»
или «контекст недостаточен», ты формируешь только отказ
по соответствующему шаблону ниже и не создаёшь артикуляционное
описание.

КРИТИЧЕСКОЕ ПРАВИЛО:
Не используй знания о фонетике, которых нет в переданном контексте.
Даже если факт кажется тебе очевидным, типичным или общеизвестным,
не добавляй его без прямого основания в чанках.

ПРАВИЛА ДОСТОВЕРНОСТИ:

1. Используй только факты, которые явно содержатся в переданных чанках.

2. Не дополняй ответ общими знаниями о фонетике.

3. Не делай выводы по аналогии.
Например, нельзя писать «как обычно для dental-звуков»,
если такой факт не сформулирован в переданном контексте.

4. Не используй фонетические характеристики — «глухой», «звонкий»,
«взрывной», «щелевой», «округлённый», «напряжённый» и подобные —
если именно они не встречаются явно в переданном контексте.

5. Не исправляй IPA-транскрипцию пользователя.

6. Не создавай и не угадывай номера страниц.
Используй только page_number, переданные в контексте.

7. Не описывай содержание рисунков.
Если has_figure=true у использованного чанка, добавь только:
«см. рис. на стр. N».

8. Не утверждай, что рисунок объясняет конкретный факт артикуляции,
если это не сказано в текстовом контексте.

9. Не создавай информацию о языке, губах, голосе, шуме или воздушной
струе, если такой информации нет в чанках.

10. Если для конкретного поля данных нет, напиши:
«в источнике не описано».
Не заменяй это общими знаниями и не используй другие формулировки
вроде «не указано» или «нет данных».

Если в контексте есть раздел сравнения с русскими звуками
(Comparison with the Russian ...), передай его смысл точно,
не меняя направление утверждения. Не пиши, что аналогов нет,
если источник говорит, что они есть, и наоборот.
Если раздел сравнения не попал в переданный контекст,
не упоминай сравнение с русским языком вообще.
Если раздел Comparison with the Russian обрывается, неполный
или не содержит законченного утверждения, не упоминай сравнение
с русскими звуками вообще. Не восстанавливай и не додумывай
пропущенную часть.

ПРАВИЛА ПОЛНОТЫ:

11. Используй все существенные сведения из релевантных переданных чанков.

12. Не сжимай богатое описание источника до одной сухой строки.
Если контекст содержит несколько важных деталей, передай их
в связном русском изложении.

13. Подробность ответа должна отражать подробность найденного контекста,
а не стремление заполнить все поля любой ценой.

14. Если для одной фонемы релевантные чанки с разных страниц используют
разную терминологию или содержат разные формулировки, покажи их как
разные формулировки с указанием соответствующих страниц.

15. Не пытайся «примирить», усреднить или выбрать одну из разных
формулировок, если источник явно не даёт основания предпочесть одну.

ПРАВИЛА ОТКАЗА:

16. Если тебе явно передан статус «фонема не поддерживается», выведи
только:

«Фонема /X/ не входит в валидированный набор текущего MVP
или для неё недостаточно проверяемого контекста в PDF.
Релевантные фрагменты не использованы, чтобы не создавать
описание из общих знаний.»

17. Если фонема поддерживается, но переданного контекста недостаточно
для конкретного случая, выведи:

«Недостаточно данных в источнике для описания фонемы /X/.
Найдены фрагменты на стр. N, но они не покрывают
запрошенные характеристики артикуляции.»

18. После любого отказа не добавляй объяснение из общих
фонетических знаний и не пытайся «компенсировать» отказ.

ПРАВИЛА ПОРЯДКА:

19. Порядок фонем в ответе уже определён программой.
Не меняй его и не пересортировывай фонемы самостоятельно.

20. Не выдавай место образования или порядок вывода за факт,
полученный из PDF, если он передан как статическая метка кода.

ФОРМАТ ОТВЕТА:

Верни ТОЛЬКО один развёрнутый русский абзац об артикуляции
на основе переданного контекста.

Не пиши заголовки, Markdown-списки, нумерацию, источники,
IPA-заголовки вида /f/, строки «Артикуляция:», «Детали из источника»,
«Источники», «см. рис.» и любые шаблонные поля.
Не цитируй системный промпт и технические подсказки.
Не транслитерируй IPA и не используй кириллические заменители звуков.
Если данных недостаточно для связного абзаца, напиши только:
«в источнике не описано».

Статическая метка места образования может быть показана Python
вне твоего текста, но не является источником артикуляционных фактов.
Не выводи положение губ, языка или иные характеристики из метки
места образования. Описывай их только при прямом наличии в контексте.
"""


def validate_ipa_no_internal_spaces(ipa: str) -> None:
    """Reject IPA that still contains whitespace after outer trim."""
    if any(ch.isspace() for ch in ipa):
        raise ValueError(IPA_INTERNAL_SPACE_ERROR)


def parse_user_input(raw_input: str) -> tuple[str, str]:
    """Parse «слово | /IPA/». Does not correct the user's transcription."""
    if raw_input is None:
        raise ValueError("Пустой ввод. Ожидается формат: слово | /IPA/")

    text = raw_input.strip()
    if not text:
        raise ValueError("Пустой ввод. Ожидается формат: слово | /IPA/")

    if text.count("|") != 1:
        raise ValueError(
            "Неверный формат. Нужен ровно один разделитель «|»: слово | /IPA/"
        )

    left, right = text.split("|", 1)
    word = left.strip()
    ipa = right.strip()

    if not word:
        raise ValueError("Слово не указано. Ожидается формат: слово | /IPA/")
    if not ipa:
        raise ValueError("IPA не указана. Ожидается формат: слово | /IPA/")
    if not (ipa.startswith("/") and ipa.endswith("/") and len(ipa) >= 3):
        raise ValueError(
            "IPA должна начинаться и заканчиваться символом «/», например: /kæt/"
        )

    # Before phoneme split, cache key, retrieval, or OpenAI.
    validate_ipa_no_internal_spaces(ipa)

    return word, ipa


def parse_ipa(ipa: str) -> list[str]:
    """Split IPA into phonemes using longest-match over the RP inventory."""
    if ipa is None:
        raise ValueError("Пустая IPA-строка.")

    text = ipa.strip()
    validate_ipa_no_internal_spaces(text)

    if text.startswith("/") and text.endswith("/") and len(text) >= 2:
        text = text[1:-1]

    for mark in STRESS_MARKS:
        text = text.replace(mark, "")

    if not text:
        raise ValueError("После удаления ударения IPA пуста.")

    phonemes: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        matched = None
        for candidate in SORTED_PHONEMES:
            end = index + len(candidate)
            if text[index:end] == candidate:
                matched = candidate
                break

        if matched is None:
            fragment = text[index : index + 1]
            raise ValueError(f"Не удалось распознать фрагмент IPA: {fragment}")

        phonemes.append(matched)
        index += len(matched)

    return phonemes


def get_place_and_order(phoneme: str) -> tuple[str, int]:
    """Return (place, sort_key) for static technical classification."""
    place = PHONEME_PLACE.get(phoneme, "unclassified")
    if place == "vowel":
        # Vowels after all consonant places; relative IPA order applied later.
        return place, len(PLACE_ORDER)
    if place == "unclassified":
        return place, len(PLACE_ORDER) + 1
    return place, PLACE_RANK[place]


def is_supported(phoneme: str) -> bool:
    """True if phoneme is in the validated MVP set (TZ v1.3)."""
    return phoneme in SUPPORTED_PHONEMES


def unsupported_phoneme_message(phoneme: str) -> str:
    return (
        f"Фонема /{phoneme}/ не входит в валидированный набор текущего MVP\n"
        "или для неё недостаточно проверяемого контекста в PDF.\n"
        "Релевантные фрагменты не использованы, чтобы не создавать\n"
        "описание из общих знаний."
    )


def sort_phonemes_for_answer(phonemes: list[str]) -> list[str]:
    """Consonants by place order; vowels next in IPA order; unclassified last."""
    consonants: list[tuple[int, int, str]] = []
    vowels: list[tuple[int, str]] = []
    unclassified: list[tuple[int, str]] = []

    for ipa_index, phoneme in enumerate(phonemes):
        place, place_rank = get_place_and_order(phoneme)
        if place == "vowel":
            vowels.append((ipa_index, phoneme))
        elif place == "unclassified":
            unclassified.append((ipa_index, phoneme))
        else:
            consonants.append((place_rank, ipa_index, phoneme))

    consonants.sort(key=lambda item: (item[0], item[1]))
    # Keep original IPA order among vowels / unclassified.
    vowels.sort(key=lambda item: item[0])
    unclassified.sort(key=lambda item: item[0])

    ordered = [item[2] for item in consonants]
    ordered.extend(item[1] for item in vowels)
    ordered.extend(item[1] for item in unclassified)
    return ordered


def search_contexts(phoneme: str, top_k: int = 4) -> list[dict]:
    """Retrieve top_k Chroma chunks for one phoneme without mutating the DB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="phonetics_chunks")

    query_text = f"English phoneme /{phoneme}/ articulation tongue lips voice airflow"
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns nested lists: documents[0], metadatas[0], distances[0].
    documents_nested = results.get("documents")
    metadatas_nested = results.get("metadatas")
    distances_nested = results.get("distances")
    if (
        not documents_nested
        or not isinstance(documents_nested, list)
        or not documents_nested[0]
    ):
        return []

    documents = documents_nested[0]
    metadatas = (
        metadatas_nested[0]
        if metadatas_nested and isinstance(metadatas_nested, list)
        else [{}] * len(documents)
    )
    distances = (
        distances_nested[0]
        if distances_nested and isinstance(distances_nested, list)
        else [None] * len(documents)
    )

    contexts: list[dict] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        contexts.append(
            {
                "text": text or "",
                "page_number": metadata.get("page_number"),
                "has_figure": bool(metadata.get("has_figure", False)),
                # Always present for diagnostics / filtering.
                "distance": float(distance) if distance is not None else float("inf"),
            }
        )
    return contexts


def filter_validated_chunks(
    phoneme: str,
    search_results: list[dict],
) -> list[dict]:
    """Keep only retrieval hits from the phoneme's validated_pages."""
    validated_pages = set(SUPPORTED_PHONEMES[phoneme]["validated_pages"])  # type: ignore[arg-type]
    return [
        item
        for item in search_results
        if item.get("page_number") in validated_pages
    ]


def source_page_numbers(chunks: list[dict]) -> list[int]:
    """Unique page numbers ascending from filtered chunks."""
    pages = {
        int(item["page_number"])
        for item in chunks
        if item.get("page_number") is not None
    }
    return sorted(pages)


def figure_page_numbers(chunks: list[dict]) -> list[int]:
    """Unique ascending pages among filtered chunks with has_figure=True."""
    pages = {
        int(item["page_number"])
        for item in chunks
        if item.get("has_figure") and item.get("page_number") is not None
    }
    return sorted(pages)


def format_supported_phoneme_block(
    phoneme: str,
    place: str,
    llm_paragraph: str,
    chunks: list[dict],
) -> str:
    """Assemble one phoneme block in Python (header/sources never from LLM)."""
    paragraph = (llm_paragraph or "").strip()
    pages = source_page_numbers(chunks)
    sources = (
        "Источники: " + ", ".join(f"стр. {page}" for page in pages) + "."
        if pages
        else "Источники: стр. —."
    )
    lines = [
        f"/{phoneme}/ — {place}",
        "",
        "Артикуляция:",
        paragraph,
        "",
        sources,
    ]
    for page in figure_page_numbers(chunks):
        lines.append(f"см. рис. на стр. {page}")
    return "\n".join(lines)


def build_context(
    search_results: list[dict],
    phoneme: str | None = None,
) -> str:
    """Build a compact context string from retrieved documents only.

    For SUPPORTED_PHONEMES, keep only chunks from the phoneme's validated_pages.
    Intentionally omits distance so the LLM does not see ranking scores.
    """
    results = list(search_results)
    if phoneme is not None and is_supported(phoneme):
        results = filter_validated_chunks(phoneme, results)
        if not results:
            validated_pages = SUPPORTED_PHONEMES[phoneme]["validated_pages"]
            return (
                "недостаточно контекста: "
                f"для /{phoneme}/ в результате retrieval нет чанков "
                f"со страниц {validated_pages}."
            )

    if not results:
        return "Контекст не найден."

    parts: list[str] = []
    for index, item in enumerate(results[:4], start=1):
        page_number = item.get("page_number")
        has_figure = bool(item.get("has_figure", False))
        text = (item.get("text") or "")[:1800]
        parts.append(
            f"[Чанк {index}]\n"
            f"page_number: {page_number}\n"
            f"has_figure: {has_figure}\n"
            f"text:\n{text}"
        )
    return "\n\n".join(parts)


def context_page_numbers(context: str) -> list[int]:
    """Extract unique ascending page_number values from a build_context string."""
    pages = {
        int(match.group(1))
        for match in re.finditer(r"^page_number:\s*(\d+)\s*$", context, re.MULTILINE)
    }
    return sorted(pages)


def format_retrieval_diagnostics(phoneme: str, search_results: list[dict]) -> str:
    """Format retrieval hits for safe local diagnostics (includes distance)."""
    lines = [f"phoneme: /{phoneme}/", f"hits: {len(search_results)}"]
    if not search_results:
        lines.append("(нет результатов)")
        return "\n".join(lines)

    for index, item in enumerate(search_results, start=1):
        distance = item.get("distance", float("inf"))
        try:
            distance_str = f"{float(distance):.4f}"
        except (TypeError, ValueError):
            distance_str = "n/a"
        preview = (item.get("text") or "")[:180].replace("\n", " ")
        lines.append(
            f"[{index}] page_number={item.get('page_number')} "
            f"has_figure={bool(item.get('has_figure', False))} "
            f"distance={distance_str}\n"
            f"    text: {preview}"
        )
    return "\n".join(lines)


def diagnose_retrieval(phoneme: str, top_k: int = 4) -> list[dict]:
    """Safe diagnostic mode: search only, print hits, never call OpenAI."""
    results = search_contexts(phoneme, top_k=top_k)
    print(format_retrieval_diagnostics(phoneme, results))
    print()
    return results


def print_full_context(phoneme: str, top_k: int = 4) -> str:
    """Safe diagnostic: print the exact context string that would go to the LLM."""
    results = search_contexts(phoneme, top_k=top_k)
    context = build_context(results, phoneme=phoneme)
    print(f"=== full context for /{phoneme}/ ===")
    print(context)
    print(f"=== end context (/ {phoneme}/), chars={len(context)} ===")
    return context


def _require_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() in {
        "",
        "ваш_реальный_ключ_здесь",
        "your_openai_api_key_here",
    }:
        raise RuntimeError(
            "OPENAI_API_KEY не найден. Создайте локальный файл .env по образцу .env.example"
        )
    return OpenAI(api_key=api_key)


def normalize_cache_key(word: str, ipa: str) -> str:
    """Build a stable cache key after input parsing."""
    return f"{word.strip().lower()}|{ipa.strip()}"


def _ensure_cache_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            response_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()


def get_cached_response(cache_key: str) -> str | None:
    """Return cached response_text for key, or None on miss."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_DB_PATH) as connection:
        _ensure_cache_schema(connection)
        row = connection.execute(
            "SELECT response_text FROM response_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    return str(row[0])


def save_cached_response(cache_key: str, response_text: str) -> None:
    """Persist a successfully generated response."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(CACHE_DB_PATH) as connection:
        _ensure_cache_schema(connection)
        connection.execute(
            """
            INSERT OR REPLACE INTO response_cache (cache_key, response_text, created_at)
            VALUES (?, ?, ?)
            """,
            (cache_key, response_text, created_at),
        )
        connection.commit()


def generate_answer(word: str, ipa: str, phonemes: list[str]) -> str:
    """Build one full answer for a word: single header + per-phoneme blocks.

    Sorting and support checks happen in Python before any LLM call.
    Header, place line, and sources are assembled in Python; LLM returns
    only one articulation paragraph.
    Successful final answers are stored in SQLite and reused on cache hit.
    """
    cache_key = normalize_cache_key(word, ipa)
    cached = get_cached_response(cache_key)
    if cached is not None:
        print(f"cache hit -> {cache_key}")
        return cached

    print(f"cache miss -> {cache_key}")

    ordered = sort_phonemes_for_answer(phonemes)

    parts: list[str] = [
        f"Слово: {word}",
        f"IPA: {ipa}",
        f"Распознанные фонемы: {', '.join(f'/{p}/' for p in phonemes)}",
        "",
    ]

    client: OpenAI | None = None

    for phoneme in ordered:
        if not is_supported(phoneme):
            parts.append(unsupported_phoneme_message(phoneme))
            parts.append("")
            continue

        place = str(
            SUPPORTED_PHONEMES[phoneme].get("place")
            or get_place_and_order(phoneme)[0]
        )
        search_results = search_contexts(phoneme, top_k=4)
        filtered_chunks = filter_validated_chunks(phoneme, search_results)
        context = build_context(search_results, phoneme=phoneme)

        if not filtered_chunks or context.startswith("недостаточно контекста"):
            pages = sorted(
                {
                    item.get("page_number")
                    for item in search_results
                    if item.get("page_number") is not None
                }
            )
            pages_str = ", ".join(f"стр. {p}" for p in pages) if pages else "стр. —"
            parts.append(
                f"Недостаточно данных в источнике для описания фонемы /{phoneme}/.\n"
                f"Найдены фрагменты на {pages_str}, но они не покрывают\n"
                "запрошенные характеристики артикуляции."
            )
            parts.append("")
            continue

        if client is None:
            client = _require_openai_client()

        user_prompt = (
            f"Фонема: /{phoneme}/\n"
            f"Контекст:\n{context}\n\n"
            "Напиши только один связный русский абзац об артикуляции "
            "этой фонемы по контексту. Без заголовков, списков, источников и IPA. "
            "Не используй место образования как источник фактов."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        paragraph = (response.choices[0].message.content or "").strip()
        parts.append(
            format_supported_phoneme_block(
                phoneme=phoneme,
                place=place,
                llm_paragraph=paragraph,
                chunks=filtered_chunks,
            )
        )
        parts.append("")

    response_text = "\n".join(parts).strip() + "\n"
    save_cached_response(cache_key, response_text)
    return response_text


if __name__ == "__main__":
    # Safe local checks / diagnostics only — no OpenAI / generate_answer calls.
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "diagnose":
        phonemes = sys.argv[2:] or ["θ", "t", "ɔː"]
        for phoneme in phonemes:
            diagnose_retrieval(phoneme)
        raise SystemExit(0)

    assert len(RP_PHONEMES) == len(set(RP_PHONEMES)), "RP_PHONEMES must be unique"
    assert SORTED_PHONEMES == tuple(
        sorted(RP_PHONEMES, key=len, reverse=True)
    ), "SORTED_PHONEMES must be length-descending"

    assert parse_ipa("/tʃiː/") == ["tʃ", "iː"]
    assert parse_ipa("/dʒɔː/") == ["dʒ", "ɔː"]
    assert parse_ipa("/ˈθɔːt/") == ["θ", "ɔː", "t"]

    thought_sorted = sort_phonemes_for_answer(parse_ipa("/ˈθɔːt/"))
    assert thought_sorted == ["θ", "t", "ɔː"], thought_sorted

    assert is_supported("v") is True
    assert is_supported("b") is False
    assert is_supported("s") is False
    assert is_supported("θ") is False

    five_sorted = sort_phonemes_for_answer(["f", "aɪ", "v"])
    assert five_sorted == ["f", "v", "aɪ"], five_sorted

    context_f = build_context(search_contexts("f"), phoneme="f")
    context_v = build_context(search_contexts("v"), phoneme="v")
    assert context_page_numbers(context_f) == [44], context_page_numbers(context_f)
    assert context_page_numbers(context_v) == [44, 79], context_page_numbers(context_v)
    assert not context_f.startswith("недостаточно контекста"), context_f[:200]
    assert not context_v.startswith("недостаточно контекста"), context_v[:200]

    demo_block = format_supported_phoneme_block(
        phoneme="f",
        place="labiodental",
        llm_paragraph="Тестовый абзац артикуляции.",
        chunks=[{"page_number": 44, "has_figure": False}],
    )
    assert demo_block.splitlines()[0] == "/f/ — labiodental", demo_block.splitlines()[0]

    assert normalize_cache_key(" Five ", " /faɪv/ ") == "five|/faɪv/"

    assert parse_ipa("/faɪv/") == ["f", "aɪ", "v"]
    try:
        parse_user_input("five | /f aɪ v/")
        raise AssertionError("expected ValueError for IPA with internal spaces")
    except ValueError as exc:
        assert "пробелов" in str(exc), exc
        assert "Ошибка формата IPA" in str(exc), exc

    print("Safe checks OK")
    print("parse_ipa('/ˈθɔːt/') ->", parse_ipa("/ˈθɔːt/"))
    print("sort for /ˈθɔːt/ ->", thought_sorted)
    print("is_supported: v=True, b/s/θ=False")
    print("sort for ['f', 'aɪ', 'v'] ->", five_sorted)
    print("filtered context pages f ->", context_page_numbers(context_f))
    print("filtered context pages v ->", context_page_numbers(context_v))
    print("demo block first line ->", demo_block.splitlines()[0])
    print(demo_block)
    print("cache key check ->", normalize_cache_key(" Five ", " /faɪv/ "))
