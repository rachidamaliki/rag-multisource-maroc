"""
Interface de test du RAG.

    streamlit run app_streamlit.py

Trois onglets :
  1. Tester  — poser une question (libre ou tiree du golden dataset), choisir
               la methode de recherche, voir les passages et la reponse citee.
               Si la question vient du golden dataset, les bons passages sont
               marques : on VOIT si la recherche a reussi.
  2. Evaluation — les resultats chiffres par configuration et par type.
  3. Golden dataset — les 50 questions et leurs preuves.
"""
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import GOLDEN_DIR, INDEX_DIR, REPORTS
from src.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.sources import balanced_merge

st.set_page_config(page_title="RAG Code du travail", page_icon="⚖️", layout="wide")

METHODES = {
    "RRF (hybride par rangs)": "rrf",
    "Vectoriel seul": "vectoriel",
    "BM25 seul": "bm25",
    "Ponderee [0,5 ; 1]": "ponderee",
    "Ponderee + quota par langue": "ponderee+quota",
}


# ---------------------------------------------------------------- chargement
def index_disponibles() -> list[str]:
    if not INDEX_DIR.exists():
        return []
    return sorted([d.name for d in INDEX_DIR.iterdir() if (d / "hnsw.faiss").exists()],
                  key=lambda n: (n.endswith("_v1"), n))


@st.cache_resource(show_spinner="Chargement de l'index…")
def charger_index(nom: str):
    from src.vectorstore import FaissHNSWStore
    d = INDEX_DIR / nom
    chunks = {json.loads(l)["id"]: json.loads(l) for l in (d / "chunks.jsonl").open(encoding="utf-8")}
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    return chunks, FaissHNSWStore.load(d), pickle.loads((d / "bm25.pkl").read_bytes()), meta


@st.cache_resource(show_spinner="Chargement du modele d'embedding (bge-m3)…")
def charger_embedder():
    from src.embeddings import Embedder
    return Embedder("bge-m3")


@st.cache_resource
def charger_llm(modele: str):
    from src.llm import LLMClient
    return LLMClient(model=modele)


@st.cache_data
def charger_golden() -> list[dict]:
    f = GOLDEN_DIR / "golden.jsonl"
    return [json.loads(l) for l in f.open(encoding="utf-8")] if f.exists() else []


# ---------------------------------------------------------------- recherche
def rechercher(question, methode, chunks, store, bm25, top_k, vivier):
    t0 = time.perf_counter()
    qv = charger_embedder().embed_queries([question])[0]
    t_emb = time.perf_counter() - t0

    lv = store.search(qv, k=vivier, ef_search=128)
    lb = bm25.search(question, k=vivier)
    if methode == "vectoriel":
        res = lv
    elif methode == "bm25":
        res = lb
    elif methode == "rrf":
        res = reciprocal_rank_fusion([lv, lb], top_n=vivier)
    else:
        res = weighted_score_fusion([lv, lb], poids=[0.5, 1.0], top_n=vivier)
        if methode == "ponderee+quota":
            groupes = {}
            for r in res:
                groupes.setdefault(chunks[r.chunk_id]["lang"], []).append(r)
            res = balanced_merge(groupes, total_k=vivier, min_per_group=2)
    return res[:top_k], {"embedding_ms": t_emb * 1000, "recherche_ms": (time.perf_counter() - t0 - t_emb) * 1000}


# ---------------------------------------------------------------- barre laterale
st.sidebar.title("⚖️ Reglages")
indexes = index_disponibles()
if not indexes:
    st.error("Aucun index trouve dans data/index/. Lancer scripts/01_build_index.py.")
    st.stop()

nom_index = st.sidebar.selectbox("Index", indexes,
                                 help="structural_bge-m3 = corrige ; _v1 = avant correction des renvois")
libelle = st.sidebar.radio("Methode de recherche", list(METHODES))
methode = METHODES[libelle]
top_k = st.sidebar.slider("Passages affiches / envoyes au LLM", 1, 10, 5)
vivier = st.sidebar.slider("Vivier par moteur", 5, 60, 20,
                           help="Mesure : un vivier trop large degrade RRF (95 % -> 50 % en top-5)")
generer = st.sidebar.toggle("Generer une reponse (Groq)", value=True)
modele = st.sidebar.selectbox("Modele LLM", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"], disabled=not generer)

chunks, store, bm25, meta = charger_index(nom_index)
st.sidebar.caption(f"{len(chunks)} chunks · decoupage {meta.get('chunker', '?')} · "
                   f"embeddings depuis `{meta.get('embed_field', 'text_norm')}`")

golden = charger_golden()
st.title("RAG Code du travail marocain")
onglet_test, onglet_eval, onglet_golden = st.tabs(["🔎 Tester", "📊 Evaluation", "📋 Golden dataset"])

# ---------------------------------------------------------------- onglet 1
with onglet_test:
    mode = st.radio("Source de la question", ["Question libre", "Question du golden dataset"], horizontal=True)

    attendu = None
    if mode == "Question libre":
        question = st.text_input("Votre question", placeholder="Ex : Combien de jours de conge paye par mois ?")
    else:
        if not golden:
            st.warning("golden.jsonl introuvable : lancer scripts/04_build_golden.py")
            st.stop()
        types = ["tous"] + sorted({g["type"] for g in golden})
        filtre = st.selectbox("Type", types)
        choix = [g for g in golden if filtre == "tous" or g["type"] == filtre]
        g = st.selectbox("Question", choix, format_func=lambda g: f"{g['qid']} · {g['question']}")
        question, attendu = g["question"], g
        c1, c2 = st.columns(2)
        c1.markdown(f"**Reponse attendue :** {g['expected_answer']}")
        if g["answerable"]:
            c2.markdown(f"**Article :** {g['article_ref']}  \n**Preuve :** « {g['evidence']} »")
        else:
            c2.markdown("**Question piege** — le systeme doit refuser.  \n" + (g.get("trap") or ""))

    if st.button("Lancer", type="primary", disabled=not question):
        resultats, temps = rechercher(question, methode, chunks, store, bm25, top_k, vivier)
        pertinents = set(attendu["relevant_chunk_ids"]) if attendu and attendu["answerable"] else set()

        st.subheader("Passages retrouves")
        if pertinents:
            rangs = [r.rank + 1 for r in resultats if r.chunk_id in pertinents]
            if rangs:
                st.success(f"✅ Bon article trouve — premiere occurrence en position {rangs[0]} "
                           f"(MRR = {1 / rangs[0]:.2f})")
            else:
                st.error(f"❌ Aucun passage de {attendu['article_ref']} dans le top-{top_k}")
        st.caption(f"embedding {temps['embedding_ms']:.0f} ms · recherche {temps['recherche_ms']:.0f} ms")

        passages = []
        for i, r in enumerate(resultats, start=1):
            c = chunks[r.chunk_id]
            passages.append(c)
            marque = "✅ " if r.chunk_id in pertinents else ""
            titre = f"{marque}[{i}] {c.get('unit_ref') or 'sans reference'} · {c['lang'].upper()} · score {r.score:.4f}"
            with st.expander(titre, expanded=(i == 1)):
                st.caption(c["doc_id"])
                st.write(c["text"])

        if generer:
            st.subheader("Reponse generee")
            from src.generation import generate
            with st.spinner("Generation…"):
                t = time.perf_counter()
                try:
                    out = generate(question, passages, charger_llm(modele))
                except Exception as e:
                    st.error(f"Echec de l'appel LLM : {e}")
                    st.stop()
                duree = (time.perf_counter() - t) * 1000
            st.markdown(out["answer"] or "_reponse vide_")
            ctrl = out["citations"]
            a, b, c3 = st.columns(3)
            a.metric("Refus", "oui" if out["refused"] else "non")
            b.metric("Citations valides", len(ctrl["valides"]))
            c3.metric("Citations inventees", len(ctrl["hallucinees"]))
            if ctrl["hallucinees"]:
                st.error(f"Citations vers des passages inexistants : {ctrl['hallucinees']}")
            elif ctrl["sans_citation"]:
                st.warning("La reponse affirme sans citer de source.")
            if attendu and not attendu["answerable"]:
                if out["refused"]:
                    st.success("✅ Question piege : le systeme a bien refuse")
                else:
                    st.error("❌ Question piege : le systeme a repondu au lieu de refuser (hallucination probable)")
            st.caption(f"generation {duree:.0f} ms · modele {modele}")
            with st.expander("Voir le prompt envoye"):
                st.code(out["prompt"], language="text")

# ---------------------------------------------------------------- onglet 2
with onglet_eval:
    f = REPORTS / "golden_retrieval.csv"
    if not f.exists():
        st.info("Pas encore de resultats : lancer scripts/05_evaluate_golden.py")
    else:
        df = pd.read_csv(f)
        st.caption("hit@k = au moins un bon passage dans le top-k · MRR = 1 / position du premier bon passage")
        t = st.selectbox("Type de question", df["type"].unique())
        vue = df[df["type"] == t][["configuration", "n_requetes", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "diagnostic"]]
        st.dataframe(vue.style.highlight_max(subset=["hit@1", "hit@5", "hit@10", "mrr"], color="#c8e6c9"),
                     hide_index=True, use_container_width=True)
        st.bar_chart(vue.set_index("configuration")[["hit@1", "hit@5", "hit@10"]])

# ---------------------------------------------------------------- onglet 3
with onglet_golden:
    if golden:
        df = pd.DataFrame([{"qid": g["qid"], "type": g["type"], "langue": g["lang"], "question": g["question"],
                            "article": g.get("article_ref", "—"), "preuve": g.get("evidence", "—")} for g in golden])
        st.caption(f"{len(df)} questions · chaque preuve est verifiee mot pour mot par scripts/04_build_golden.py")
        st.dataframe(df, hide_index=True, use_container_width=True)
