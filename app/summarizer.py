from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache
import math




@lru_cache(maxsize=2)
def load_summarizer(model_name: str = "facebook/bart-large-cnn"):
    #Charge et retourne un pipeline de summarization et son tokenizer.
    #lru_cache évite de recharger le modèle à chaque appel (important en déploiement).
    
    # charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
    return pipe, tokenizer




def summarize_chunks(chunks, pipe, tokenizer, max_length=256, min_length=64):
    summaries = []
    # pipeline peut gérer une liste mais on veut contrôler la longueur
    for c in chunks:
        # découper s'il est trop long pour le modèle en tokens (sécurité)
        inputs = tokenizer(c, return_tensors='pt', truncation=True)
        # appeler pipeline
        s = pipe(c, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(s[0]["summary_text"].strip())
    return summaries




def hierarchical_summary(text, pipe, tokenizer):
    """Processus en deux niveaux : résumer les chunks, puis résumer la concaténation.
    Retourne un dict avec 'short', 'medium', 'long'.
    """
    from app.pdf_utils import sentence_chunker


    # 1) Chunk
    chunks = sentence_chunker(text, max_chars=3500)


    # 2) Résumer chaque chunk (résumé détaillé par chunk)
    chunk_summaries = summarize_chunks(chunks, pipe, tokenizer, max_length=200, min_length=60)


    # 3) Résumé moyen : concaténer les résumés des chunks et résumer à nouveau
    concatenated = "\n".join(chunk_summaries)
    medium = pipe(concatenated, max_length=300, min_length=120, do_sample=False)[0]["summary_text"].strip()


    # 4) Résumé court : résumer le résumé moyen
    short = pipe(medium, max_length=120, min_length=30, do_sample=False)[0]["summary_text"].strip()


    # 5) Résumé long : la concaténation des résumés de chunks (réorganisable en bullets)
    long = "\n\n".join([f"- {s}" for s in chunk_summaries])


    return {"short": short, "medium": medium, "long": long}