"""
Demo Streamlit — la vitrine du projet (J22).

C'est ce que vous montrez en entretien ou a un client. Elle doit afficher
la reponse ET les passages sources, sinon on ne voit pas ce qui fait la
valeur du systeme : la tracabilite.

Lancer :  streamlit run app_streamlit.py
"""
import streamlit as st

st.set_page_config(page_title="RAG From Zero", page_icon="🔍", layout="wide")
st.title("RAG From Zero — assistant documentaire ancre")
st.caption("Recherche hybride (vectoriel + BM25) → reranking cross-encoder → generation citee")

with st.sidebar:
    st.header("Parametres")
    top_k = st.slider("Passages transmis au LLM", 1, 10, 5)
    use_rerank = st.checkbox("Activer le reranking", value=True)
    use_hybrid = st.checkbox("Recherche hybride", value=True)
    st.divider()
    st.markdown("**Astuce demo** : decochez le reranking et reposez la meme "
                "question pour montrer la difference de qualite en direct.")

question = st.text_input("Votre question", placeholder="Ex : quelles sont les conditions de rupture du contrat ?")

if st.button("Interroger", type="primary") and question:
    # TODO (J22) — appeler l'API /query ou le pipeline directement
    st.warning("A brancher sur votre pipeline (Arc 8).")
    # Structure d'affichage attendue :
    #   st.markdown(reponse)
    #   for i, src in enumerate(sources, 1):
    #       with st.expander(f"[{i}] {src['doc_id']} — score {src['score']:.3f}"):
    #           st.write(src["text"])
    #   st.caption(f"Latence : {latence} ms")
