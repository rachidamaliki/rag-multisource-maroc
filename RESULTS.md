# RESULTS — journal des chiffres

> Remplissez ce fichier AU FUR ET A MESURE. Au J19 vous ne vous souviendrez
> plus des resultats du J8. Ce fichier alimente directement le README final.

## Arc 0 — Geometrie
- Corpus de test : ___ phrases
- Accord kNN maison vs sklearn : ___
- Observation sur la polysemie (banque/rivière) : ___

## Arc 1 — Chunking
| Strategie | Nb chunks | Tokens moy. | Variance | Fragmentation | Cout embedding |
|---|---|---|---|---|---|
| fixed | | | | | |
| sentence | | | | | |
| semantic | | | | | |
| structural | | | | | |

## Arc 2 — Modeles d'embedding
| Modele | Dim | Recall@5 FR | Recall@5 AR | Cross-lingue | Latence/1k chunks |
|---|---|---|---|---|---|
| bge-m3 | 1024 | | | | |
| multilingual-e5-base | 768 | | | | |
| paraphrase-MiniLM | 384 | | | | |

**Modele retenu et pourquoi :** ___

## Arc 3 — HNSW
| ef_search | Recall@10 vs exact | Latence p50 (ms) | Latence p95 |
|---|---|---|---|
| 16 | | | |
| 32 | | | |
| 64 | | | |
| 128 | | | |
| 256 | | | |

**Coude de la courbe :** ef_search = ___
**Surcout du filtrage par metadonnees :** ___ ms

## Arc 4 — Hybride
| Methode | semantic | exact_match | cross_source | Global |
|---|---|---|---|---|
| Vectoriel seul | | | | |
| BM25 seul | | | | |
| Hybride RRF | | | | |
| Hybride pondere | | | | |
| Hybride + quota par source | | | | |

**Biais de volume mesure** : part de chaque source dans le top-30 AVANT quota
(vs part attendue) : ___

**Cas d'echec trouve :** ___

## Arc 5 — Reranking
| Pool reranke | Precision@5 | Latence totale (ms) |
|---|---|---|
| 10 | | |
| 30 | | |
| 100 | | |

**Configuration retenue + justification (1 paragraphe) :** ___

## Arc 6 — Evaluation
- Taille du golden dataset : ___ questions (___ fr / ___ ar / ___ non repondables)
- **Accord juge LLM vs annotation manuelle : ___** (obligatoire)

## Arc 7 — Ancrage
- Citations hallucinees : ___ / ___
- Test de refus adversarial : ___ hallucinations sur 15
- Effet de l'ordre des passages (lost-in-the-middle) : ___

## Multi-source
### Corpus
| Source | Nb documents | Nb chunks | Langues | Autorite |
|---|---|---|---|---|
| code_travail | | | fr/ar | 3 |
| cgnc | | | fr | 3 |
| dgi_circulaires | | | fr/ar | 2 |

### Filtrage (Arc 3)
| Filtre | Recall@10 | Latence p50 | Surcout vs sans filtre |
|---|---|---|---|
| aucun | | | — |
| source = code_travail | | | |
| source = dgi_circulaires (petite source) | | | |

**Observation sur le surcout du filtrage :** ___

### Ventilation par source (Arc 6)
| Source | Recall@5 | MRR | Faithfulness |
|---|---|---|---|
| code_travail | | | |
| cgnc | | | |
| dgi_circulaires | | | |

**Source la plus difficile et pourquoi :** ___

### Conflits (Arc 7)
- Questions `conflict` : ___ / 6 arbitrees correctement
- Conflits explicitement signales dans la reponse : ___ / ___

## Arc 8 — Production
- Latence bout-en-bout p50 / p95 : ___ / ___ ms
- Taux de cache : ___ %
- Tokens moyens par requete : ___
