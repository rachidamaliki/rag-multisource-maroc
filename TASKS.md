# TÂCHES — la liste unique

Un seul fichier à suivre. Cochez au fur et à mesure.
Les explications détaillées de chaque jour sont dans [GUIDE.md](GUIDE.md), mais **si vous ne
lisez qu'un fichier, c'est celui-ci.**

Légende : 🔴 = bloquant · ⏱️ = durée estimée · 📁 = fichier à modifier

---

## AVANT DE COMMENCER

- [x] Dépôt GitHub créé et public
- [x] Python 3.12 + librairies installées
- [ ] 🔴 **Clé Groq** — https://console.groq.com → Sign up → API Keys → coller dans `.env` ⏱️ 2 min
- [ ] 🔴 **Corpus** — PDF du Code du travail marocain dans `data/raw/code_travail/` ⏱️ 10 min

Vérifier que tout est prêt : `venv\Scripts\python scripts\check_env.py`

---

## SEMAINE 1 — Les fondations

### J1 · Similarité cosinus 📁 `src/geometry.py` ⏱️ 3 h
- [ ] `normalize()` — diviser un vecteur par sa norme
- [ ] `cosine_similarity()` — produit scalaire des vecteurs normalisés
- [ ] `euclidean_distance()` — pour comparer avec le cosinus
- [ ] Embedder 5 phrases, afficher la matrice de similarité en heatmap
- [ ] ✅ Validation : `pytest tests/test_metrics.py::test_cosine_matches_sklearn`

### J2 · Recherche exhaustive 📁 `src/geometry.py` ⏱️ 3 h
- [ ] Comparer « banque » finance vs rivière → observer la polysémie
- [ ] Refaire le test en arabe
- [ ] Projeter 50 phrases en 2D avec UMAP
- [ ] `brute_force_knn()` — les k plus proches voisins, sans librairie
- [ ] ✅ Validation : vérifier à la main le top-3 de 10 requêtes

### J3 · Nettoyer le corpus 📁 `scripts/00_ingest.py` ⏱️ 3 h
- [ ] `clean()` — enlever en-têtes, pieds de page, recoller les mots coupés
- [ ] `clean()` — normaliser l'arabe (alef, ya, diacritiques)
- [ ] `detect_lang()` — fr ou ar selon les caractères Unicode
- [ ] `detect_source()` — déduire la source du nom de dossier
- [ ] ✅ Validation : `python scripts/00_ingest.py --priority 1` produit du JSON propre

### J4 · Découpage, méthodes 1 et 2 📁 `src/chunking.py` ⏱️ 3 h
- [ ] `chunk_fixed()` — fenêtres de N tokens avec recouvrement (la baseline)
- [ ] `chunk_by_sentence()` — ne jamais couper au milieu d'une phrase
- [ ] ✅ Validation : afficher 5 chunks de chaque méthode et les lire

### J5 · Découpage, méthodes 3 et 4 📁 `src/chunking.py` ⏱️ 3 h
- [ ] `chunk_semantic()` — couper là où le sens change
- [ ] `chunk_structural()` — ne jamais couper un article de loi
- [ ] `context_fragmentation_rate()` — mesurer les réponses coupées en deux
- [ ] ✅ Validation : le chunker structurel garde l'article 145 entier

### J6 · Comparer les modèles 📁 `src/embeddings.py` ⏱️ 3 h
- [ ] `benchmark_models()` — comparer 3 modèles d'embedding
- [ ] Test croisé FR ↔ AR : 20 concepts, interroger dans une langue, trouver dans l'autre
- [ ] ✅ Validation : un tableau comparatif dans `RESULTS.md`

---

## SEMAINE 2 — Mesurer, puis chercher

### J7 · Jeu de test, partie 1 📁 `data/golden/golden.jsonl` ⏱️ 2 h
- [ ] Écrire 30 questions + le passage qui y répond (annotation manuelle)
- [ ] ✅ Validation : 30 lignes JSON valides

### J8 · Les métriques 📁 `src/metrics.py` ⏱️ 3 h
- [ ] `recall_at_k()` — le bon passage est-il dans le top-k ?
- [ ] `precision_at_k()`
- [ ] `mrr()` — à quelle position arrive le premier bon résultat ?
- [ ] `ndcg_at_k()` — pertinence graduée avec amortissement
- [ ] `evaluate_retrieval()` — agréger sur tout le jeu de test
- [ ] ✅ Validation : `pytest -q` passe + premier tableau réel dans `RESULTS.md`

### J9 · L'index + les 2 autres sources 📁 `src/vectorstore.py`, `src/sources.py` ⏱️ 3 h
- [ ] Ingérer `cgnc` et `dgi_circulaires` (`--priority 2`)
- [ ] `html_to_pages()` — extraire le HTML des circulaires
- [ ] `BruteForceStore.search()` — la référence exacte
- [ ] `FaissHNSWStore.build()` et `.search()` — l'index rapide
- [ ] `build_filter()` — filtrer par source pendant la recherche
- [ ] ✅ Validation : mêmes requêtes sur les deux index, résultats comparés

### J10 · Exact vs approximatif 📁 `src/vectorstore.py` ⏱️ 3 h
- [ ] `recall_loss_curve()` — faire varier `ef_search` (16 → 256)
- [ ] Tracer recall@10 en fonction de la latence
- [ ] ✅ Validation : 📊 `reports/figures/arc3_recall_vs_latency.png`

### J11 · BM25 📁 `src/lexical.py` ⏱️ 3 h
- [ ] `tokenize()` — découper en mots, FR et AR
- [ ] `BM25.__init__()` et `.search()` — depuis la formule
- [ ] Trouver une requête que le vectoriel rate et que BM25 trouve
- [ ] ✅ Validation : vos scores correspondent à ceux de `rank_bm25`

### J12 · Fusion 📁 `src/fusion.py`, `src/sources.py` ⏱️ 3 h
- [ ] `reciprocal_rank_fusion()` — fusionner par rangs, pas par scores
- [ ] `weighted_score_fusion()` — l'alternative, à comparer
- [ ] `balanced_merge()` — quota par source (sinon la plus grosse écrase tout)
- [ ] ✅ Validation : 📊 tableau vectoriel / BM25 / hybride par type de question

---

## SEMAINE 3 — Précision et ancrage

### J13 · Reranking 📁 `src/rerank.py` ⏱️ 3 h
- [ ] `CrossEncoderReranker.rerank()` — top-30 → top-5
- [ ] Mesurer le gain de précision
- [ ] ✅ Validation : la précision@5 augmente vs sans reranking

### J14 · Précision vs latence 📁 `src/rerank.py` ⏱️ 3 h
- [ ] `precision_latency_curve()` — pools de 10 / 30 / 100 candidats
- [ ] Écrire un paragraphe justifiant votre choix
- [ ] ✅ Validation : 📊 `reports/figures/arc5_precision_vs_latency.png`

### J15 · Jeu de test, partie 2 📁 `data/golden/golden.jsonl` ⏱️ 2 h
- [ ] +10 questions `cross_source` (réponse dans 2 sources)
- [ ] +6 questions `conflict` (2 sources se contredisent)
- [ ] +14 questions sans réponse possible
- [ ] ✅ Validation : 60 questions au total

### J16 · Juge LLM 📁 `src/judge.py` ⏱️ 3 h
- [ ] `judge_faithfulness()` — la réponse est-elle fidèle aux sources ?
- [ ] `ANSWER_RELEVANCE_PROMPT` — écrire la grille
- [ ] `calibrate_judge()` — 🔴 vérifier l'accord avec 10 notes manuelles
- [ ] ✅ Validation : le taux d'accord juge/humain est noté dans `RESULTS.md`

### J17 · Réponses citées 📁 `src/generation.py`, `src/sources.py` ⏱️ 3 h
- [ ] `build_context()` — numéroter les passages [1] [2]
- [ ] `verify_citations()` — détecter les citations inventées
- [ ] `resolve_conflict()` — autorité d'abord, date ensuite
- [ ] Tester l'ordre des passages (effet « lost in the middle »)
- [ ] ✅ Validation : zéro citation hallucinée sur 20 questions

### J18 · Savoir dire « je ne sais pas » 📁 `src/generation.py` ⏱️ 3 h
- [ ] `has_refused()`
- [ ] `adversarial_refusal_test()` — 15 questions impossibles
- [ ] ✅ Validation : au plus 2 hallucinations sur 15

---

## SEMAINE 4 — Production et vitrine

### J19 · Le tableau maître 📁 `scripts/02_evaluate.py`, `src/sources.py` ⏱️ 3 h
- [ ] Boucler sur toutes les combinaisons chunking × retrieval × reranking
- [ ] `per_source_report()` — ventiler les métriques par source
- [ ] ✅ Validation : 🏆 `reports/master_table.csv`

### J20 · L'API 📁 `src/api.py` ⏱️ 3 h
- [ ] `startup()` — charger les modèles une seule fois
- [ ] `query()` — le pipeline complet
- [ ] `RAGPipeline.retrieve()` et `.answer()` 📁 `src/pipeline.py`
- [ ] ✅ Validation : `uvicorn src.api:app` répond sur `/query`

### J21 · Robustesse 📁 `src/api.py` ⏱️ 3 h
- [ ] Cache de réponses
- [ ] `stats()` — latence, coût, taux de cache
- [ ] `drift()` — détecter les questions inhabituelles
- [ ] Replis si un composant tombe
- [ ] ✅ Validation : couper le reranker → l'API répond quand même

### J22 · La démo 📁 `app_streamlit.py` ⏱️ 3 h
- [ ] Brancher l'interface sur l'API
- [ ] Afficher les passages sources sous la réponse
- [ ] ✅ Validation : `streamlit run app_streamlit.py`

### J23 · Le README 📁 `README.md` ⏱️ 3 h
- [ ] Remplir tous les tableaux de résultats
- [ ] Intégrer les 3 figures
- [ ] Justifier chaque décision par un chiffre
- [ ] ✅ Validation : 🏆 un README qu'un recruteur comprend en 2 minutes

### J24 · Publication ⏱️ 3 h
- [ ] Vérifier qu'aucune clé n'est commitée
- [ ] Vidéo de démo de 2 minutes
- [ ] Mettre à jour CV + LinkedIn + Indeed
- [ ] ✅ Validation : le lien du dépôt est sur votre CV

---

## Récapitulatif

| | |
|---|---|
| Fonctions à écrire | **115** |
| Questions à annoter | 60 |
| Durée | 24 jours × 3 h = **72 h** |
| Livrables finaux | 3 figures + 1 tableau maître + 1 API + 1 démo |
