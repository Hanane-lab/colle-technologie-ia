import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import pdfplumber
import re


def extract_text_from_pdf(file_stream) -> str:
    text_parts = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def clean_text(text: str):
    # Supprimer les caractères parasites (carrés, �, □, unicode illisibles)
    text = re.sub(r"[■□�◆◇▶◀▪▫✖✘❌✓✔✚✖️🔸🔹🔺🔻⬜⬛]", " ", text)

    # Supprimer les dates automatiques (ex : 02/10/2025 ou 2/1/24)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text)

    # Supprimer les numéros de page (ex : "Page 2", "2/25", "p.3", "3 p")
    text = re.sub(r"\bpage\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)
    text = re.sub(r"\bp\.\s*\d+\b", " ", text, flags=re.IGNORECASE)

    # Supprimer les %
    text = text.replace("%", " ")

    # Supprimer les doubles espaces et nettoyer
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_chunker(text: str, max_chars: int = 3500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return chunks
