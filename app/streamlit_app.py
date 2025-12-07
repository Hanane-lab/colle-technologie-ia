import streamlit as st
from pdf_utils import extract_text_from_pdf, clean_text
from summarizer import load_summarizer, hierarchical_summary
from io import BytesIO

st.set_page_config(page_title="Résumeur de Cours", layout='wide')

st.title("Résumé automatique de cours — Prêt à déployer 🚀")
st.markdown("Upload un PDF de cours et obtiens 3 niveaux de résumé (court / moyen / long).")

# --- INITIALISATION SESSION STATE ---
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None

if "pipe" not in st.session_state:
    st.session_state["pipe"] = None

if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Choisir le modèle", [
        "bert-extractive", 
        "t5-base"
    ])

    max_chars = st.slider("Approx. max caractères par chunk", 2000, 8000, 3500, step=500)
    run_button = st.button("Charger modèle et résumer")

# --- UPLOAD ---
uploaded_file = st.file_uploader("Choisir un PDF", type=['pdf'])

if uploaded_file is not None:

    file_bytes = BytesIO(uploaded_file.read())

    with st.spinner("Extraction du texte..."):
        raw_text = extract_text_from_pdf(file_bytes)
        cleaned = clean_text(raw_text)

    st.info(f"Texte extrait — longueur: {len(cleaned)} caractères")

    # --- BOUTON CLIQUÉ ---
    if run_button:
        with st.spinner("Chargement du modèle (peut prendre du temps)…"):
            pipe, tokenizer = load_summarizer(model_name)
            st.session_state["pipe"] = pipe
            st.session_state["tokenizer"] = tokenizer

        with st.spinner("Génération des résumés..."):
            summaries = hierarchical_summary(cleaned, pipe, tokenizer)
            st.session_state["summaries"] = summaries

    # --- AFFICHAGE SI DES RÉSUMÉS EXISTENT ---
    if st.session_state["summaries"] is not None:
        summaries = st.session_state["summaries"]

        st.subheader("Résumé court (abstract)")
        st.write(summaries['short'])

        st.subheader("Résumé moyen")
        st.write(summaries['medium'])

        st.subheader("Résumé long (points clés)")
        st.write(summaries['long'])

        # Download text file
        def make_download(text, filename="resume.txt"):
            return BytesIO(text.encode('utf-8'))

        st.download_button(
            "Télécharger le résumé (TXT)",
            data=make_download(summaries['medium'].strip()),
            file_name='resume.txt'
        )

        st.success("Résumé généré ✔")

else:
    st.write("Aucun PDF chargé — upload un PDF pour commencer.")
