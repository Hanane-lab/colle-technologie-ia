import streamlit as st
from app.pdf_utils import extract_text_from_pdf, clean_text
from app.summarizer import load_summarizer, hierarchical_summary


st.set_page_config(page_title="Résumeur de Cours", layout='wide')


st.title("Résumé automatique de cours — Prêt à déployer 🚀")


st.markdown("Upload un PDF de cours et obtiens 3 niveaux de résumé (court / moyen / long).")


with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Choisir le modèle", [
    "facebook/bart-large-cnn",
    "t5-base",
    # "google/led-base-16384" # optionnel si vous déployez GPU/plus de mémoire
    ])
    max_chars = st.slider("Approx. max caractères par chunk", 2000, 8000, 3500, step=500)
    run_button = st.button("Charger modèle et résumer")


uploaded_file = st.file_uploader("Choisir un PDF", type=['pdf'])


if uploaded_file is not None:
    # Convert to BytesIO
    file_bytes = BytesIO(uploaded_file.read())
    with st.spinner("Extraction du texte..."):
        raw_text = extract_text_from_pdf(file_bytes)
        cleaned = clean_text(raw_text)


    st.info("Texte extrait — longueur: {} caractères".format(len(cleaned)))


    if run_button:
        with st.spinner("Chargement du modèle (peut prendre du temps la première fois)…"):
            pipe, tokenizer = load_summarizer(model_name)


        with st.spinner("Génération des résumés..."):
            summaries = hierarchical_summary(cleaned, pipe, tokenizer)


    st.subheader("Résumé court (abstract)")
    st.write(summaries['short'])


    st.subheader("Résumé moyen")
    st.write(summaries['medium'])


    st.subheader("Résumé long (points clés)")
    st.write(summaries['long'])


    # Download as text
    def make_download(text, filename="resume.txt"):
        return BytesIO(text.encode('utf-8'))


    st.download_button("Télécharger le résumé (TXT)", data=make_download(summaries['medium'].strip()), file_name='resume.txt')


    st.success("Fini — tu peux maintenant télécharger ou copier les résumés.")


else:
    st.write("Aucun PDF chargé — upload un PDF pour commencer.")