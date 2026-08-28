# RAG From Zero — recherche documentaire ancree, construite depuis les primitives

> Assistant reglementaire **multi-source** et bilingue **francais / arabe**, implemente sans
> LangChain ni LlamaIndex : chaque composant — similarite cosinus, BM25, fusion de rangs,
> metriques de retrieval — est ecrit a la main et **mesure**.

**Corpus :** Code du travail marocain + CGNC + circulaires DGI (sources publiques)
**Statut :** en cours — voir [GUIDE.md](GUIDE.md) pour le plan de travail.

## Le probleme traite

Une PME marocaine ne consulte pas *un* texte de reference, mais plusieurs : le Code du travail,
le plan comptable, et les circulaires fiscales qui les actualisent. Repondre correctement exige
donc de **croiser plusieurs sources**, de savoir **laquelle fait foi** quand elles divergent, et
de **citer** systematiquement d'ou vient l'information.

Ce sont trois problemes que les demonstrations RAG mono-document n'abordent jamais, et les
trois raisons pour lesquelles elles ne survivent pas a un usage reel.

---

## Pourquoi ce projet

La plupart des implementations RAG reposent sur un framework qui prend une
dizaine de decisions techniques a votre place — taille des chunks, modele
d'embedding, parametres de l'index, strategie de fusion — sans jamais les
exposer. Quand la qualite se degrade, il devient impossible de diagnostiquer.

Ici, chaque decision est prise explicitement et **justifiee par une mesure**
sur un jeu d'evaluation annote a la main.

## Resultats

> Section a remplir au fil du projet — c'est elle que lit un recruteur.
> Les chiffres bruts sont journalises dans [RESULTS.md](RESULTS.md).

| Configuration | Recall@5 | MRR | Faithfulness | Latence p50 |
|---|---|---|---|---|
| Baseline (chunks fixes + vectoriel) | — | — | — | — |
| + chunking structurel (articles preserves) | — | — | — | — |
| + recherche hybride (RRF) | — | — | — | — |
| + fusion equilibree entre sources | — | — | — | — |
| + reranking cross-encoder | — | — | — | — |

**Par source** (la moyenne globale masque les ecarts) :

| Source | Recall@5 | Faithfulness | Remarque |
|---|---|---|---|
| Code du travail | — | — | |
| CGNC | — | — | |
| Circulaires DGI | — | — | |

**Par type de question** :

| Type | Recall@5 | Ce qui est teste |
|---|---|---|
| `semantic` | — | comprehension |
| `exact_match` | — | apport de BM25 |
| `cross_source` | — | fusion equilibree entre sources |
| `conflict` | — | regle d'autorite (loi > doctrine, recent > ancien) |
| `unanswerable` | — | taux de refus correct |

**Figures :**
- `reports/figures/arc3_recall_vs_latency.png` — cout reel de la recherche approximative
- `reports/figures/arc4_hybrid_comparison.png` — ou le vectoriel seul echoue
- `reports/figures/arc5_precision_vs_latency.png` — arbitrage precision / latence

## Architecture

```
question  (+ filtre optionnel : source, langue, autorite)
   |
   +-> embedding (bge-m3, multilingue) --> index HNSW  --+
   +-> tokenisation FR/AR              --> index BM25  --+
                                                         |
                              fusion RRF + quota par source (top-30)
                                                         |
                                     cross-encoder rerank (top-5)
                                                         |
                              resolution de conflit (autorite, puis date)
                                                         |
                              generation citee [1] [2] + refus si insuffisant
                                                         |
                              verification des citations
```

Les deux etages propres au multi-source — le **quota par source** (sans lui, la source la plus
volumineuse ecrase les autres) et la **resolution de conflit** — sont dans
[`src/sources.py`](src/sources.py).

## Stack

100 % gratuit, aucun service payant.

| Role | Choix | Pourquoi |
|---|---|---|
| Embeddings | `BAAI/bge-m3` | multilingue FR/AR, local, gratuit |
| Index vectoriel | FAISS HNSW / pgvector | local ; pgvector pour le filtrage SQL |
| Lexical | BM25 implemente a la main | references exactes (articles, codes) |
| Reranking | `bge-reranker-v2-m3` | cross-encoder, gratuit |
| Generation | Groq (Llama 3.3 70B) ou Gemini | quota gratuit |
| API / demo | FastAPI + Streamlit | |

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # puis renseigner la cle Groq ou Gemini
```

Base vectorielle optionnelle (Arc 3) :

```bash
docker run -d --name pgvector -p 5432:5432 \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=ragpass -e POSTGRES_DB=ragzero \
  pgvector/pgvector:pg16
```

## Utilisation

```bash
python scripts/00_ingest.py --priority 1                     # source 1 (J3)
python scripts/00_ingest.py --priority 2                     # sources 2 et 3 (J9)
python scripts/01_build_index.py --chunker structural --model bge-m3
python scripts/02_evaluate.py --config all --output reports/master_table.csv
uvicorn src.api:app --reload                                 # API
streamlit run app_streamlit.py                               # demo
pytest -q                                                    # tests de non-regression
```

## Structure

```
src/
  geometry.py      Arc 0  similarite cosinus, kNN exhaustif (numpy pur)
  chunking.py      Arc 1  4 strategies de decoupage comparables
  embeddings.py    Arc 2  couche multi-modeles + cache disque
  vectorstore.py   Arc 3  brute force / FAISS-HNSW / pgvector
  lexical.py       Arc 4  BM25 depuis la formule
  fusion.py        Arc 4  Reciprocal Rank Fusion
  rerank.py        Arc 5  cross-encoder
  sources.py       multi-source : filtrage, quota par source, conflits
  metrics.py       Arc 6  recall@k, MRR, NDCG
  judge.py         Arc 6  faithfulness, LLM-as-judge calibre
  generation.py    Arc 7  prompt ancre, verification des citations, refus
  pipeline.py      assemblage complet
  api.py           Arc 8  FastAPI, cache, monitoring, derive
```

## Licence

MIT — corpus exclu (textes publics, non redistribues).
