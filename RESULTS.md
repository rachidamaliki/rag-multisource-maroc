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

## Index construit  [structural_bge-m3]
- corpus : 4 documents, 586 pages, 2 langues
- decoupage structurel : 2 968 chunks, 185 tokens moy., 2 404 avec reference d'article
- embeddings bge-m3 : 2 968 x 1 024, 2 534 s sur CPU (~42 min, une seule fois grace au cache)
- BM25 : 15 278 termes distincts, 45 tokens/chunk apres retrait des mots vides

## PREMIERE MESURE — vectoriel vs BM25 sur des identifiants exacts
Protocole : 20 numeros d'articles reellement presents dans le corpus, requete
« article N », verite terrain gratuite via le champ `unit_ref` des chunks.

| Methode | top-1 | top-5 |
|---|---|---|
| Vectoriel (bge-m3) | **10 %** | **10 %** |
| BM25 | **95 %** | **95 %** |

Exemples d'echec du vectoriel :
  « article 231 » -> renvoie l'article 219
  « article 396 » -> renvoie l'article 376
  « dahir 1.03.194 » -> aucun resultat pertinent

Interpretation : pour le modele d'embedding, « article 231 » et « article 219 »
occupent presque le meme point de l'espace — ce sont tous deux « un article de
loi avec un numero ». Le numero lui-meme ne porte quasiment aucun signal.
BM25, lui, cherche le token `231` et le trouve.

Le fait que top-1 = top-5 pour les deux methodes est revelateur : elargir la
recherche n'aide pas le vectoriel. Ce n'est pas un probleme de classement,
c'est un probleme de representation.

>>> C'est la justification chiffree de la recherche HYBRIDE. <<<

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
