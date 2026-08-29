# TÂCHES — la liste unique

Un seul fichier à suivre. Cochez au fur et à mesure.

**Plan recalibré** pour un profil qui maîtrise déjà embeddings, ETL, appels LLM et déploiement :
la semaine 1 est compressée de 6 jours à 4, et les heures gagnées sont réinvesties sur
l'évaluation — la partie qui manque réellement au CV.

Légende : 📁 fichier · ⏱️ durée · 🎓 concept nouveau pour vous · ⚡ compressé (déjà acquis)

> Chaque jour, avant de coder : demandez **« explique le concept du Jn »**.
> Les explications longues sont dans [GUIDE.md](GUIDE.md) (numérotation d'origine, voir
> la correspondance indiquée sur chaque jour).

---

## AVANT DE COMMENCER

- [x] Dépôt GitHub créé et public
- [x] Python 3.12 + librairies installées
- [x] Clé Groq configurée, appel LLM testé (`openai/gpt-oss-120b`)
- [x] Corpus : 4 PDF du Code du travail (2 FR, 2 AR) — voir [data/raw/README.md](data/raw/README.md)

Vérifier : `venv\Scripts\python scripts\check_env.py`

---

## SEMAINE 1 — Fondations (compressée)

### ✅ J1 · Géométrie ⚡ 📁 `src/geometry.py` — *fait*
*(ex-GUIDE J1 + J2)*
- [x] `normalize()`, `cosine_similarity()`, `euclidean_distance()`
- [x] `brute_force_knn()` — la vérité terrain du projet
- [x] Contrat de forme `(n,d) × (m,d) → (n,m)` validé
- [x] ✅ `pytest tests/test_metrics.py::test_cosine_matches_sklearn` passe

### J2 · Nettoyer le corpus 🎓 📁 `scripts/00_ingest.py` ⏱️ 3 h
*(ex-GUIDE J3)* — **concept : normalisation de l'arabe, un problème que le CV ne couvre pas**
- [ ] `clean()` — en-têtes, pieds de page, mots coupés en fin de ligne
- [ ] `clean()` — **ligature lam-alef** (`المقاوالت` → `المقاولات`), alef ا/أ/إ, ya ي/ى, diacritiques
- [ ] `clean()` — espaces parasites intra-mots (20,8 % de fragmentation mesurée)
- [ ] `detect_lang()` — fr / ar par plage Unicode
- [ ] `detect_source()` — depuis le nom de dossier
- [ ] ✅ `python scripts/00_ingest.py --priority 1` produit du JSON propre

### J3 · Les 4 découpages ⚡ 📁 `src/chunking.py` ⏱️ 4 h
*(ex-GUIDE J4 + J5)*
- [ ] `chunk_fixed()` — fenêtres de tokens avec recouvrement (baseline)
- [ ] `chunk_by_sentence()` — jamais de coupure en milieu de phrase (regex FR **et** AR)
- [ ] `chunk_semantic()` — couper où la similarité entre phrases chute
- [ ] `chunk_structural()` 🎓 — l'article de loi est atomique, jamais coupé
- [ ] `context_fragmentation_rate()` 🎓 — mesurer les réponses coupées en deux
- [ ] ✅ L'article 145 reste entier avec le chunker structurel

### J4 · Comparer les modèles 📁 `src/embeddings.py` ⏱️ 3 h
*(ex-GUIDE J6)* — **concept : robustesse cross-lingue, votre différenciation**
- [ ] `benchmark_models()` — 3 modèles sur qualité, latence, taille
- [ ] 🎓 Test croisé FR ↔ AR : interroger en français, retrouver le passage arabe
- [ ] Vérifier le piège des préfixes `query:` / `passage:`
- [ ] ✅ Tableau comparatif dans `RESULTS.md`

---

## SEMAINE 2 — Mesurer, puis chercher

### J5 · Golden dataset, partie 1 🎓 📁 `data/golden/golden.jsonl` ⏱️ **3 h** *(+1 h)*
*(ex-GUIDE J7)* — **concept : ce qui rend tout le reste démontrable**
- [ ] 30 questions + le passage attendu, annotées à la main
- [ ] Formuler avec **vos mots**, jamais le vocabulaire du passage
- [ ] ✅ 30 lignes JSON valides

### J6 · Les métriques 🎓 📁 `src/metrics.py` ⏱️ 3 h
*(ex-GUIDE J8)* — **concept : recall vs MRR vs NDCG, et ce que chacune ne voit pas**
- [ ] `recall_at_k()`, `precision_at_k()`
- [ ] `mrr()` — position du premier bon résultat
- [ ] `ndcg_at_k()` — pertinence graduée, amortissement logarithmique
- [ ] `evaluate_retrieval()` — agrégation sur tout le jeu de test
- [ ] ✅ `pytest -q` passe + premier tableau réel dans `RESULTS.md`

### J7 · Index + sources 2 et 3 🎓 📁 `src/vectorstore.py`, `src/sources.py` ⏱️ 3 h
*(ex-GUIDE J9)* — **concept : HNSW, et pourquoi filtrer pendant ≠ filtrer après**
- [ ] Ingérer `cgnc` et `dgi_circulaires` (`--priority 2`) + `html_to_pages()`
- [ ] `BruteForceStore.search()` — réutilise le J1
- [ ] `FaissHNSWStore.build()` / `.search()`
- [ ] `build_filter()` — filtrer par source **dans** la requête
- [ ] ✅ Mêmes requêtes sur les deux index, écarts constatés

### J8 · Exact vs approximatif 🎓 📁 `src/vectorstore.py` ⏱️ 3 h
*(ex-GUIDE J10)* — **concept : le compromis recall / latence, et où est le coude**
- [ ] `recall_loss_curve()` — `ef_search` de 16 à 256
- [ ] Mesurer aussi le surcoût du filtrage par source
- [ ] ✅ 📊 `reports/figures/arc3_recall_vs_latency.png`

### J9 · BM25 🎓 📁 `src/lexical.py` ⏱️ 3 h
*(ex-GUIDE J11)* — **concept : saturation TF, IDF, et pourquoi le vectoriel rate « article 145 »**
- [ ] `tokenize()` — FR et AR
- [ ] `BM25` depuis la formule (k1, b), vérifié contre `rank_bm25`
- [ ] Trouver le cas d'échec du vectoriel
- [ ] ✅ Vos scores correspondent à la référence

### J10 · Fusion 🎓 📁 `src/fusion.py`, `src/sources.py` ⏱️ 3 h
*(ex-GUIDE J12)* — **concept : pourquoi fusionner des rangs et pas des scores**
- [ ] `reciprocal_rank_fusion()`
- [ ] `weighted_score_fusion()` — l'alternative, à comparer
- [ ] `balanced_merge()` — quota par source (biais de volume)
- [ ] ✅ 📊 Tableau vectoriel / BM25 / hybride par type de question

---

## SEMAINE 3 — Précision et ancrage

### J11 · Reranking 🎓 📁 `src/rerank.py` ⏱️ 3 h
*(ex-GUIDE J13)* — **concept : bi-encodeur O(1) vs cross-encodeur O(n)**
- [ ] `CrossEncoderReranker.rerank()` — top-30 → top-5
- [ ] ✅ La précision@5 augmente vs sans reranking

### J12 · Précision vs latence 🎓 📁 `src/rerank.py` ⏱️ 3 h
*(ex-GUIDE J14)* — **concept : arbitrer avec des chiffres, pas des impressions**
- [ ] `precision_latency_curve()` — pools de 10 / 30 / 100
- [ ] Rédiger la justification du choix retenu
- [ ] ✅ 📊 `reports/figures/arc5_precision_vs_latency.png`

### J13 · Golden dataset, partie 2 🎓 📁 `data/golden/golden.jsonl` ⏱️ **4 h** *(+2 h)*
*(ex-GUIDE J15)* — **concept : questions multi-sources et questions piégées**
- [ ] +10 `cross_source` — la réponse exige deux sources
- [ ] +6 `conflict` — deux sources se contredisent
- [ ] +14 `unanswerable` — le corpus ne peut pas répondre
- [ ] ✅ 60 questions au total

### J14 · Juge LLM 🎓 📁 `src/judge.py` ⏱️ **5 h** *(+2 h)*
*(ex-GUIDE J16)* — **concept : faithfulness, et pourquoi un juge non calibré est nuisible**
- [ ] `judge_faithfulness()` — décomposer la réponse en affirmations
- [ ] `ANSWER_RELEVANCE_PROMPT` — écrire la grille
- [ ] `calibrate_judge()` — accord avec 10 notes manuelles **(obligatoire)**
- [ ] ✅ Le taux d'accord juge/humain est publié dans `RESULTS.md`

### J15 · Réponses citées 🎓 📁 `src/generation.py`, `src/sources.py` ⏱️ 3 h
*(ex-GUIDE J17)* — **concept : ancrage, lost-in-the-middle, hiérarchie d'autorité**
- [ ] `build_context()` — passages numérotés, source affichée
- [ ] `verify_citations()` — détecter les citations inventées
- [ ] `resolve_conflict()` — autorité, puis date, **et le signaler**
- [ ] Tester l'ordre des passages
- [ ] ✅ Zéro citation hallucinée sur 20 questions

### J16 · Savoir dire « je ne sais pas » 🎓 📁 `src/generation.py` ⏱️ 3 h
*(ex-GUIDE J18)* — **concept : le refus comme fonctionnalité, pas comme échec**
- [ ] `has_refused()`, `adversarial_refusal_test()`
- [ ] ✅ Au plus 2 hallucinations sur 15 questions impossibles

---

## SEMAINE 4 — Production et vitrine

### J17 · Le tableau maître 🎓 📁 `scripts/02_evaluate.py` ⏱️ **5 h** *(+2 h)*
*(ex-GUIDE J19)* — **🏆 l'artefact n°1 du projet**
- [ ] Boucler sur chunking × retrieval × reranking
- [ ] `per_source_report()` — ventilation par source
- [ ] ✅ `reports/master_table.csv` + tableau dans le README

### J18 · L'API 📁 `src/api.py`, `src/pipeline.py` ⏱️ 3 h
*(ex-GUIDE J20)*
- [ ] `RAGPipeline.retrieve()` et `.answer()`
- [ ] `startup()` — charger les modèles une seule fois
- [ ] `query()` — pipeline complet
- [ ] ✅ `uvicorn src.api:app` répond sur `/query`

### J19 · Robustesse 🎓 📁 `src/api.py` ⏱️ 3 h
*(ex-GUIDE J21)* — **concept : dégradation gracieuse et dérive**
- [ ] Cache de réponses, `stats()`, `drift()`
- [ ] Replis si un composant tombe
- [ ] ✅ Couper le reranker → l'API répond quand même

### J20 · La démo ⚡ 📁 `app_streamlit.py` ⏱️ 2 h
*(ex-GUIDE J22)*
- [ ] Interface branchée, passages sources affichés, interrupteur reranking
- [ ] ✅ `streamlit run app_streamlit.py`

### J21 · Le README 📁 `README.md` ⏱️ 3 h
*(ex-GUIDE J23)*
- [ ] Tous les tableaux remplis, 3 figures intégrées
- [ ] Chaque décision justifiée par un chiffre
- [ ] ✅ 🏆 Un README compris en 2 minutes

### J22 · Publication ⏱️ 3 h
*(ex-GUIDE J24)*
- [ ] **Révoquer la clé Groq passée par le chat, en recréer une**
- [ ] Vérifier qu'aucun secret n'est commité
- [ ] Vidéo de démo de 2 minutes
- [ ] CV + LinkedIn + Indeed mis à jour
- [ ] ✅ Le lien du dépôt est sur votre CV

---

## Récapitulatif

| | Plan initial | Plan recalibré |
|---|---|---|
| Jours | 24 | **22** |
| Heures | 72 h | 72 h |
| Heures sur l'évaluation (golden, juge, tableau maître) | 11 h | **17 h** |
| Fonctions restant à écrire | 115 | **111** |

Le mois reste tenu ; l'effort se déplace de ce que vous savez déjà vers ce qui manque au CV.
