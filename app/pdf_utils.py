import io
import pdfplumber
import re
import nltk
nltk.download('punkt', quiet=True)




def extract_text_from_pdf(file_stream) -> str:
    """Extrait tout le texte d'un fichier PDF (file_stream: BytesIO ou file-like).
    Retourne une chaîne de caractères brute (texte concatené page-par-page).
    """
    text_parts = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)




def clean_text(text: str) -> str:
    t = text
    # Supprime les en-têtes/pieds de page numériques courants (approx.)
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)
    # Remplace plusieurs espaces/newlines par un seul espace
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    return t




def sentence_chunker(text: str, max_chars: int = 3500):
    #Découpe le texte en chunks en se basant sur la segmentation en phrases.
    #max_chars est un proxy pour les tokens (pratique pour BART/T5 avec ~1024 tokens).
    #Retourne une liste de chunks.

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