# GUIDE — les étapes, expliquées

Ce fichier est votre feuille de route quotidienne. Pour chaque jour :
**ce que vous faites**, **pourquoi ça compte**, et **comment savoir que c'est fini**.

> Règle d'or : un `git commit` par jour, et `RESULTS.md` mis à jour dès qu'un chiffre sort.
> Un résultat non noté est un résultat perdu.

---

## ÉTAPE 0 — Mise en route (30 minutes, aujourd'hui)

```bash
cd C:\Users\oo\rag_fromscratch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
git init
git add .
git commit -m "chore: structure initiale du projet"
```

Créez ensuite une clé gratuite sur https://console.groq.com et collez-la dans `.env`. Vérifiez :

```bash
python -c "from src.config import settings; print(settings.llm_provider, bool(settings.groq_api_key))"
```

**Une seule décision à prendre maintenant, et à ne plus jamais rediscuter : votre corpus.**

Le projet est **multi-source** — c'est ce qui le distingue d'un exercice. Trois sources
publiques marocaines, déclarées dans `src/config.py` :

| Source | Dossier | Autorité | Quand |
|---|---|---|---|
| Code du travail | `data/raw/code_travail/` | 3 (loi) | **J3** |
| CGNC (plan comptable) | `data/raw/cgnc/` | 3 (norme) | J9 |
| Circulaires / notes DGI | `data/raw/dgi_circulaires/` | 2 (doctrine) | J9 |

**Aujourd'hui, téléchargez uniquement le Code du travail.** Les deux autres arriveront au J9,
une fois le pipeline fonctionnel de bout en bout. Ingérer trois formats le même jour est la
meilleure façon de perdre trois jours.

Pourquoi verrouiller ce choix : changer de corpus en cours de route invalide votre golden
dataset et tous vos benchmarks. C'est la première cause d'abandon des projets d'un mois.

---

# SEMAINE 1 — Les fondations

## J1 — La géométrie (`src/geometry.py`)

**Ce que vous faites** : implémenter `cosine_similarity`, `normalize` et `euclidean_distance`
en numpy pur. Puis embedder 5 phrases avec `Embedder` et tracer la matrice de similarité en heatmap.

**Pourquoi** : tout le RAG repose sur une seule idée — « proche dans l'espace = proche par le
sens ». Si cette idée reste abstraite, tout le reste sera de la magie. Le cosinus mesure un
**angle** : deux textes de longueurs très différentes mais de même sens pointent dans la même
direction, alors que leur distance euclidienne serait grande.

**Fini quand** : `pytest tests/test_metrics.py::test_cosine_matches_sklearn` passe.

## J2 — Casser les embeddings + baseline kNN

**Ce que vous faites** : embedder « la banque a refusé mon crédit » et « je me suis assise sur
la banque au bord du fleuve ». Regarder leur similarité. Refaire en arabe. Puis UMAP 2D sur
50 phrases de sujets différents. Enfin, **boss fight** : `brute_force_knn` sur 100 phrases.

**Pourquoi** : vous découvrez que les embeddings ne « comprennent » pas — ils moyennent. Cette
limite explique pourquoi BM25 restera nécessaire (Arc 4). Et le kNN exhaustif devient votre
**vérité terrain** : sans lui, impossible de prouver au J10 ce que HNSW vous fait perdre.

**Fini quand** : le top-3 de 10 requêtes est vérifié manuellement et a du sens.

## J3 — Le corpus, source 1 seulement (`scripts/00_ingest.py`)

```bash
python scripts/00_ingest.py --priority 1     # code_travail uniquement
```

**Ce que vous faites** : PDF → texte propre, **pour la seule source `code_travail`**. Adapter
`clean()` (en-têtes, pieds de page, mots coupés, normalisation arabe : alef ا/أ/إ, ya ي/ى,
diacritiques), puis `detect_lang()` et `detect_source()`.

**Pourquoi une seule source aujourd'hui** : l'ingestion est la journée la plus risquée du
planning. Un pipeline qui marche sur une source se généralise en quelques heures ; un pipeline
qui ne marche sur aucune ne se généralise jamais. Les métadonnées (`source_id`, `authority`,
`unit`) sont écrites dès maintenant — ce sont elles qui rendront possibles le filtrage (J9), la
fusion équilibrée (J12) et l'arbitrage des conflits (J17).

**Fini quand** : `data/corpus/code_travail__*.json` existe, chaque document porte son
`source_id` et son `authority`, et vous avez lu 3 pages au hasard sans y trouver de pollution.

## J4 — Chunking 1 et 2 (`src/chunking.py`)

**Ce que vous faites** : `chunk_fixed` (baseline) et `chunk_by_sentence`.

**Pourquoi** : la baseline n'est pas là pour être bonne, elle est là pour être **battue avec
des chiffres**. Sans point de comparaison, « ma méthode est meilleure » n'est qu'une opinion.

**Attention** : le découpage de phrases en arabe ne suit pas les règles du français (pas de
majuscules, ponctuation ؟ et ،). Écrivez votre propre regex — c'est le but de l'exercice.

## J5 — Chunking 3 et 4 + fragmentation

**Ce que vous faites** : `chunk_semantic` (couper là où la similarité entre phrases
consécutives chute) et `chunk_structural` (**boss fight** : ne jamais couper un article de loi
ni un tableau). Puis `context_fragmentation_rate`.

**Pourquoi** : c'est le pire échec silencieux d'un RAG. Si l'article 145 est coupé en deux,
aucun des deux morceaux ne répond à la question, et le système échoue sans qu'aucune erreur ne
s'affiche. Sur un corpus juridique, le chunker structurel devrait gagner nettement — et vous
allez le prouver.

## J6 — La couche embedding (`src/embeddings.py`)

**Ce que vous faites** : le cache est déjà écrit ; à vous d'implémenter `benchmark_models` et
le **boss fight multilingue** — 20 concepts en FR et en AR, interroger dans une langue,
retrouver dans l'autre.

**Pourquoi** : c'est le résultat le plus original de tout votre projet. Presque personne ne
mesure la robustesse cross-lingue FR/AR, et c'est exactement ce dont a besoin n'importe quelle
entreprise marocaine. Vérifiez aussi le piège des préfixes (`query:` / `passage:`) : les
inverser fait chuter le recall sans provoquer la moindre erreur.

---

# SEMAINE 2 — Mesurer, puis chercher

> Le document original place l'évaluation à la fin. **On l'avance ici**, parce que sans
> métriques tous les choix suivants sont à l'aveugle. On construit la balance avant de peser.

## J7 — Golden dataset, partie 1 (30 questions)

**Ce que vous faites** : lire des passages, écrire **avec vos mots** la question à laquelle ils
répondent, noter l'identifiant du chunk. Voir `data/golden/README.md` pour la répartition.

**Pourquoi** : c'est le seul travail du projet qu'aucun outil ne peut faire à votre place, et
celui qui rend tout le reste crédible.

**Piège** : ne recopiez pas le vocabulaire du passage dans la question. Sinon vous testez la
recherche par mots-clés, pas la compréhension — et vos scores seront artificiellement excellents.

**Fini quand** : 30 lignes dans `data/golden/golden.jsonl`. Comptez 2 heures.

## J8 — Les métriques (`src/metrics.py`)

**Ce que vous faites** : `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`, depuis les
formules. Puis scorer les 4 chunkings du J4-J5.

**Pourquoi** : `recall@k` répond à « le bon passage est-il remonté ? » — si non, aucun LLM ne
peut rattraper. `MRR` répond à « à quelle position ? ». On peut avoir un bon recall et un
mauvais MRR : les deux sont nécessaires.

**Fini quand** : `pytest -q` passe, et RESULTS.md contient votre **premier tableau réel**.
C'est le premier moment où le projet devient sérieux.

## J9 — L'index + les 2 sources restantes (`src/vectorstore.py`, `src/sources.py`)

```bash
python scripts/00_ingest.py --priority 2     # cgnc + dgi_circulaires
```

**Ce que vous faites** : d'abord ingérer les deux sources restantes (le HTML des circulaires
demande `html_to_pages`). Puis `BruteForceStore`, `FaissHNSWStore`, et **`build_filter()`** —
le filtrage par source, langue et autorité.

**Le point technique à ne pas rater** : le filtre doit s'appliquer **pendant** la recherche,
pas après. Si vous récupérez le top-30 global puis jetez ce qui n'est pas de la bonne source,
il peut ne rien rester. Avec pgvector c'est une clause `WHERE` dans la même requête ; avec
FAISS, un `IDSelector`.

**Mesure à faire (checklist Arc 3)** : le surcoût du filtrage. Résultat contre-intuitif à
attendre — filtrer sur une **petite** source peut *ralentir* HNSW, parce que le graphe doit
explorer bien plus loin pour trouver assez de candidats valides. Notez-le : c'est le genre
d'observation qui impressionne en entretien.

**Pourquoi** : HNSW construit un graphe en couches — on atterrit grossièrement dans la bonne
région, puis on affine. On échange de la précision contre de la vitesse. Dessinez le graphe sur
papier avant de faire confiance à la librairie : c'est la question d'entretien classique.

## J10 — BOSS FIGHT : exact vs approximatif

**Ce que vous faites** : mêmes requêtes sur l'index exact et sur HNSW, en variant `ef_search`
(16, 32, 64, 128, 256). Tracer recall@10 vs latence.

**Pourquoi** : la valeur par défaut d'`ef_search` n'est presque jamais la bonne, et un
`ef_search` trop bas détruit le recall **silencieusement**. Savoir où est le coude de la courbe
pour VOS données est une compétence d'ingénieure, pas d'utilisatrice de librairie.

**Fini quand** : `reports/figures/arc3_recall_vs_latency.png` existe, avec le point retenu
annoté. **Première figure de portfolio.**

## J11 — BM25 (`src/lexical.py`)

**Ce que vous faites** : tokeniseur FR/AR, puis BM25 depuis la formule. Vérifier vos scores
contre `rank_bm25`. Trouver **le cas d'échec** : une requête du type « article 145 » que le
vectoriel rate et que BM25 trouve.

**Pourquoi** : ce cas d'échec est le cœur de l'arc suivant. Il justifie à lui seul l'existence
de l'hybride — et c'est une démonstration très convaincante face à un client (« votre recherche
sémantique seule ne retrouvera pas vos références de comptes »).

## J12 — Fusion RRF (`src/fusion.py`)

**Ce que vous faites** : `reciprocal_rank_fusion` à la main, puis `weighted_score_fusion`.
Comparer les deux sur vos requêtes `semantic` et `exact_match` séparément.

**Pourquoi** : additionner un score cosinus (0 à 1) et un score BM25 (0 à 30) n'a aucun sens —
les échelles sont incomparables. La RRF ignore les scores et ne garde que les **rangs**. C'est
15 lignes de code, et c'est typiquement ce qu'un framework fait sans que vous sachiez que ça existe.

**Puis, spécifique au multi-source** : implémenter `balanced_merge()` dans
[`src/sources.py`](src/sources.py). Problème réel à corriger : si le Code du travail fait
3 000 chunks et les circulaires 200, le top-30 sera presque toujours 100 % Code du travail —
non parce qu'il est plus pertinent, mais parce qu'il est **plus gros**. C'est un biais de
volume. La parade : garantir un quota minimal par source avant la fusion.

**Fini quand** : tableau vectoriel / BM25 / hybride / hybride équilibré, ventilé par type de
requête — et la fusion équilibrée doit gagner nettement sur les questions `cross_source`.
**Deuxième artefact de portfolio.**

---

# SEMAINE 3 — Précision et ancrage

## J13 — Reranking (`src/rerank.py`)

**Ce que vous faites** : top-30 hybride → top-5 reranké par cross-encoder.

**Pourquoi** (question d'entretien quasi certaine) : le bi-encodeur encode question et document
**séparément** — les documents sont pré-calculés, donc c'est instantané, mais le modèle ne les a
jamais vus ensemble. Le cross-encodeur les lit **ensemble** : bien plus précis, mais il faut un
passage de modèle par paire. D'où les deux étages : ratisser large et vite, puis affiner sur peu.

## J14 — BOSS FIGHT : précision vs latence

**Ce que vous faites** : precision@5 vs latence bout-en-bout pour des pools de 10 / 30 / 100
candidats. Puis **écrire un paragraphe** justifiant votre choix comme devant un chef de produit.

**Pourquoi** : savoir arbitrer avec des chiffres (« +180 ms pour +2 %, pas rentable en
interactif ») est précisément ce qui distingue un profil qu'on embauche.
**Troisième figure de portfolio.**

## J15 — Golden dataset, partie 2 (+30 questions)

**Ce que vous faites** : monter à 60 questions. Les 3 sources étant en place depuis le J9,
c'est maintenant que vous écrivez les catégories les plus intéressantes : **10 `cross_source`**
(la réponse exige deux sources), **6 `conflict`** (deux sources se contredisent), les questions
**en arabe**, et les **14 non répondables**.

**Pourquoi** : les `cross_source` et `conflict` sont quasi absentes des projets RAG publics.
Ce sont elles qui donnent du sens à `balanced_merge()` et à `resolve_conflict()` — sans elles,
ces deux fonctions ne sont que du code non validé. Et les non répondables sont la matière
première du J18 : un système qui répond toujours n'est pas un bon système, c'est un système
dangereux.

## J16 — Le juge LLM (`src/judge.py`)

**Ce que vous faites** : implémenter `judge_faithfulness`, puis **obligatoirement**
`calibrate_judge` sur 10 exemples que vous avez notés à la main.

**Pourquoi** : un juge LLM sans grille ni calibrage est **pire que pas de métrique** — il donne
une fausse confiance. Le chiffre d'accord juge/humain doit apparaître dans votre README : c'est
lui qui rend toutes vos autres métriques de génération crédibles.

## J17 — Génération citée (`src/generation.py`)

**Ce que vous faites** : `build_context` (passages numérotés), `verify_citations`, et le test
de l'ordre des passages.

**Pourquoi** : un bon retrieval ne garantit pas une bonne réponse. Le modèle peut recevoir le
passage parfait et l'ignorer. L'effet « lost in the middle » est réel : les LLM traitent mieux
le début et la fin du contexte. À mesurer, pas à supposer.

**Spécifique au multi-source** : implémenter `resolve_conflict()`. Règle explicite — autorité
décroissante (loi > doctrine > guide), puis date la plus récente. Et surtout, la réponse
générée doit **mentionner le conflit** : « selon la circulaire de 2025 [2], qui actualise le
guide de 2019 [5] ». Le prompt doit aussi afficher la source de chaque passage numéroté, sinon
l'utilisateur ne peut pas juger de la fiabilité de ce qu'il lit. Validez sur vos 6 questions
`conflict`.

## J18 — BOSS FIGHT : le test de refus

**Ce que vous faites** : passer vos 15 questions non répondables dans le pipeline. Critère :
**au plus 1 ou 2 hallucinations**. Sinon, renforcez la logique de refus et recommencez.

**Pourquoi** : c'est le test que presque personne ne fait, et le premier que posera un client
sérieux. Sur du juridique ou du comptable, une réponse inventée est un risque réel — savoir
dire « je ne sais pas » est une fonctionnalité, pas un échec.

---

# SEMAINE 4 — Production et vitrine

## J19 — LE TABLEAU MAÎTRE (`scripts/02_evaluate.py`)

**Ce que vous faites** : boucler sur toutes les combinaisons (4 chunkings × 4 retrievals —
vectoriel / BM25 / hybride / hybride équilibré — × 2 reranking) et produire une seule page :
recall@5, MRR, NDCG@10, faithfulness, latence, coût.

**Puis `per_source_report()`** : ventiler toutes les métriques **par source**. C'est souvent la
ligne la plus révélatrice du projet — un modèle d'embedding peut exceller sur le juridique et
s'effondrer sur les tableaux comptables du CGNC. La moyenne globale cache ce genre de problème ;
la ventilation le rend visible, et vous permet de dire quelque chose que personne d'autre ne
peut dire sur son projet.

**Pourquoi** : c'est **l'artefact numéro 1 du projet**. Il résume un mois de travail en un objet
qu'un recruteur comprend en 30 secondes, et qu'aucun tutoriel ne produit. C'est la différence
entre « j'ai suivi un tuto » et « j'ai mené une étude ».

## J20 — L'API (`src/api.py`)

**Ce que vous faites** : `/query` = embed → hybride → rerank → générer → citer. Charger les
modèles **une seule fois** au démarrage.

**Pourquoi** : un notebook n'est pas un système. Une API, si.

## J21 — Robustesse

**Ce que vous faites** : cache de réponses, log du coût et de la latence par étage, `/stats`,
`/drift`, et surtout les **replis** : si le reranker tombe, renvoyer les résultats hybrides en
signalant le mode dégradé.

**Pourquoi** : c'est ce qui sépare une démo d'un service. « Que se passe-t-il si le vector store
ne répond plus ? » est une question d'entretien système très fréquente — et vous aurez une
réponse concrète.

## J22 — La démo (`app_streamlit.py`)

**Ce que vous faites** : question → réponse citée + passages sources affichables. Ajoutez
l'interrupteur reranking on/off.

**Pourquoi** : en entretien comme devant un client, pouvoir désactiver le reranking en direct et
montrer la dégradation vaut mieux que dix minutes d'explication.

## J23 — Le README

**Ce que vous faites** : remplir la section Résultats, intégrer les 3 figures et le tableau
maître, documenter **chaque décision avec le chiffre qui la justifie**.

**Pourquoi** : c'est ce que les gens lisent réellement. Un projet excellent avec un README vide
est un projet invisible. Formule cible pour chaque choix : « j'ai retenu X plutôt que Y parce
que _chiffre_ ».

## J24 — Publication

**Ce que vous faites** : nettoyer le dépôt (vérifier qu'aucune clé n'est commitée), publier sur
GitHub, enregistrer une vidéo de démo de 2 minutes, mettre à jour CV + LinkedIn +
**profil Indeed**.

---

## Si vous prenez du retard

Sacrifiez dans cet ordre :

1. la 3ᵉ source (`dgi_circulaires`) — gardez `code_travail` + `cgnc`, le multi-source
   fonctionne dès deux sources
2. pgvector (gardez FAISS seul)
3. le 3ᵉ modèle d'embedding
4. le chunking sémantique
5. le streaming
6. l'interface Streamlit

**Ne descendez jamais sous deux sources** : c'est ce qui donne son intérêt aux questions
`cross_source` et `conflict`, donc à la moitié de la valeur ajoutée du projet.

**Ne sacrifiez JAMAIS** : le golden dataset, les métriques, le tableau maître, le README.
Ce sont eux qui portent 100 % de la valeur du projet.
