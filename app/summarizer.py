import nltk
nltk.download("punkt", quiet=True)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ============================================================
#  EXTRACTIF BERT (MiniLM) — VERSION AMÉLIORÉE
# ============================================================
class BertExtractiveSummarizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def summarize(self, text, ratio=0.2):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return text

        # Encodage en petits batches pour réduire la RAM
        sent_emb = self.model.encode(
            sentences,
            batch_size=4,           # 🔥 très important
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Idem pour le document complet
        doc_emb = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True
        )[0].reshape(1, -1)

        sims = cosine_similarity(sent_emb, doc_emb).reshape(-1)

        k = max(1, int(len(sentences) * ratio))
        top_idx = sorted(np.argsort(sims)[-k:])
        selected = [sentences[i] for i in top_idx]

        return " ".join([clean_partial(s) for s in selected])


# ============================================================
#  CHARGEMENT DES MODELES
# ============================================================
def load_summarizer(model_name: str):
    """
    Chargement robuste du modèle choisi.
    - bert-extractive = SentenceTransformer (extractive)
    - t5-base = modèle seq2seq (abstractive)
    """

    # MODE EXTRACTIF
    if model_name == "bert-extractive":
        model = BertExtractiveSummarizer()
        tokenizer = None
        return model, tokenizer

    # MODE T5 (Abstractive)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        local_files_only=False
    )

    # Force CPU (évite les erreurs CUDA/OOM)
    model.to("cpu")

    summarizer_pipe = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU obligatoire
        framework="pt"
    )

    return summarizer_pipe, tokenizer



# ============================================================
#   NETTOYAGE DES PETITS RESUMÉS
# ============================================================
def clean_partial(text):
    # Supprimer les caractères parasites
    text = re.sub(r"[■□�◆◇▶◀▪▫✖✘❌✓✔✚✖️🔸🔹🔺🔻⬜⬛]", " ", text)

    # Supprimer les %
    text = text.replace("%", "")

    # Compresser les espaces
    text = re.sub(r"\s+", " ", text)

    # Ajouter un saut de ligne AVANT chaque puce
    text = re.sub(r"[•\-]\s*", r"\n- ", text)

    # Supprimer la première ligne vide potentielle
    text = text.lstrip()

    # Optionnel : ajouter une ligne vide entre les points
    text = re.sub(r"\n- ", r"\n\n- ", text)

    return text.strip()




# ============================================================
#   RESUME D’UN CHUNK
# ============================================================
def summarize_chunk(chunk, model):

    # --- BERT EXTRACTIF ---
    if isinstance(model, BertExtractiveSummarizer):
        return clean_partial(model.summarize(chunk, ratio=0.25))

    # --- T5 ABSTRACTION ---
    prompt = (
        "Résume le texte suivant sous forme structurée et lisible, "
        "avec des sections, des titres et des puces. "
        "Retourne un résumé long et organisé :\n\n"
        + chunk
    )

    out = model(prompt)[0]["summary_text"]   # <-- FIX ICI
    return clean_partial(out.strip())


# ============================================================
#   RESUME HIERARCHIQUE AMÉLIORÉ
# ============================================================
from pdf_utils import sentence_chunker


def hierarchical_summary(text, model, tokenizer=None):

    # Chunks
    chunks = sentence_chunker(text, max_chars=3000)
    partial = [summarize_chunk(c, model) for c in chunks]
    partial = [clean_partial(p) for p in partial]

    # ===========================
    #   RESUME MOYEN STRUCTURÉ
    # ===========================
    medium_prompt = (
        "Tu es un expert en pédagogie informatique.\n"
        "Réécris un résumé clair, structuré et professionnel à partir des résumés suivants.\n"
        "Respecte EXACTEMENT la structure suivante :\n\n"
        "1. Introduction (maximum 3 phrases)\n"
        "2. Concepts principaux : sections clairement séparées\n"
        "   - Définition\n"
        "   - Objectif\n"
        "   - Caractéristiques essentielles\n"
        "3. Synthèse finale (2 phrases)\n\n"
        "Règles strictes :\n"
        "- phrases courtes\n"
        "- aucune redondance\n"
        "- vocabulaire simple\n"
        "- ne pas inventer d’informations\n"
        "- pas d'exemples\n\n"
        "Voici les résumés :\n\n"
        + "\n\n".join(partial)
    )

    if isinstance(model, BertExtractiveSummarizer):
        medium = clean_partial(model.summarize(" ".join(partial), ratio=0.25))
    else:
        medium = clean_partial(model(medium_prompt)[0]["summary_text"])

    # ===========================
    #   RESUME COURT
    # ===========================
    short_prompt = (
        "Résume le texte suivant en 6 phrases courtes et essentielles.\n"
        "Garde uniquement les notions principales du chapitre UML.\n\n"
        + medium
    )

    short = (
        model(short_prompt)[0]["summary_text"]
        if not isinstance(model, BertExtractiveSummarizer)
        else model.summarize(medium, ratio=0.15)
    )

    short = clean_partial(short)

    # ===========================
    #   RESUME LONG (POINTS CLÉS)
    # ===========================
    long_prompt = (
        "Transforme les résumés suivants en une liste de points clés.\n"
        "Chaque point doit être :\n"
        "- une seule phrase\n"
        "- un concept important\n"
        "- sans exemples\n"
        "- sans redondance\n\n"
        "Liste :\n\n"
        + "\n\n".join(partial)
    )

    if isinstance(model, BertExtractiveSummarizer):
        long = "\n".join([f"• {p}" for p in partial])
    else:
        long = clean_partial(model(short_prompt)[0]["summary_text"])

    return {
        "short": short.strip(),
        "medium": medium.strip(),
        "long": long.strip(),
    }
