from pathlib import Path
import fitz

pdf_path = Path("data/source/sokolova_practical_phonetics.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF не найден: {pdf_path.resolve()}")

document = fitz.open(pdf_path)
pages_text = [page.get_text("text").strip() for page in document]
nonempty_pages = [index + 1 for index, text in enumerate(pages_text) if text]

print(f"PDF: {pdf_path.resolve()}")
print(f"Страниц: {len(document)}")
print(f"Страниц с текстом: {len(nonempty_pages)}")
print(f"Всего символов: {sum(len(text) for text in pages_text)}")

for page_number in (1, min(10, len(document))):
    text = pages_text[page_number - 1]
    print(f"\n--- Страница {page_number}: первые 800 символов ---")
    print(text[:800] if text else "[ТЕКСТ НЕ ИЗВЛЕЧЁН]")

document.close()
