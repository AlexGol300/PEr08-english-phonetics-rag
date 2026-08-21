"""CLI entry point for the English phonetics RAG assistant."""

from __future__ import annotations

import sys

from rag_pipeline import generate_answer, parse_ipa, parse_user_input


def process_request(raw_input: str) -> int:
    """
    Validate input, then optionally call the RAG+LLM pipeline.

    Returns 0 on success, 1 on validation error.
    Does not call OpenAI when format or IPA validation fails.
    """
    try:
        word, ipa = parse_user_input(raw_input)
    except ValueError as exc:
        print(f"Ошибка формата: {exc}")
        print("Введите транскрипцию в формате: слово | /IPA/")
        return 1

    try:
        phonemes = parse_ipa(ipa)
    except ValueError as exc:
        print(f"Ошибка IPA: {exc}")
        print("Введите транскрипцию в формате: слово | /IPA/")
        return 1

    answer = generate_answer(word, ipa, phonemes)
    print(answer)
    return 0


def main() -> int:
    print("Ассистент по практической фонетике (British RP)")
    print("Формат ввода: слово | /IPA/")
    print("Пример: thought | /θɔːt/")
    print("Для выхода введите exit или quit.\n")

    # Non-interactive one-shot: python app.py "thought /θɔːt/"
    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:]).strip()
        return process_request(raw)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            return 0

        process_request(raw)


if __name__ == "__main__":
    raise SystemExit(main())
