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

## FUSION — l'hybride n'est pas gratuitement meilleur

### RRF verifie a la main
Exemple de reference (vectoriel : A 0,58 / B 0,54 / C 0,50 ; BM25 : D 15,5 / A 15,3 / B 9,1)
  RRF -> A, B, D, C  (scores conformes au calcul manuel a 1e-12)
  somme naive des scores bruts -> D devance B : l'echelle BM25 (0-30) ecrase le cosinus (0-1)
10 tests unitaires dans tests/test_fusion.py.

### Requetes « article N » (identifiant exact), 20 requetes
| Methode | top-1 | top-5 |
|---|---|---|
| Vectoriel seul | 10 % | 10 % |
| BM25 seul | **95 %** | **95 %** |
| Hybride RRF (poids egaux, vivier 10) | 20 % | 95 % |
| Hybride pondere (poids egaux) | 15 % | 95 % |

>>> L'hybride DEGRADE le top-1 : de 95 % a 20 %. <<<

Cause mesuree : le bon chunk n'est trouve que par BM25, il recoit donc 1/61 = 0,0164.
Les chunks trouves par les DEUX moteurs cumulent ~0,025 a 0,033 et le devancent.
Or, sur ce type de requete, le vectoriel ne propose que des articles plausibles mais
faux (219 pour 231). Le consensus est donc systematiquement TROMPEUR.

### Effet de la taille du vivier de candidats (RRF, poids egaux)
| Vivier / moteur | top-1 | top-5 | bon chunk present |
|---|---|---|---|
| 5 | 10 % | 95 % | 95 % |
| 10 | 20 % | 95 % | 95 % |
| 20 | 25 % | 95 % | 95 % |
| 40 | 40 % | 65 % | 95 % |
| 60 | 40 % | 50 % | 95 % |
| 120 | 40 % | 50 % | 100 % |

La colonne « bon chunk present » reste a 95 % : ce n'est PAS un probleme de recall
mais de CLASSEMENT. Plus le vivier vectoriel s'elargit, plus il injecte de faux
candidats que BM25 confirme quelque part.
Consequence pratique : la taille du vivier est un parametre de qualite, pas un
simple reglage de performance.

### Balayage des poids [vectoriel, BM25]
| Fusion | Poids | top-1 | top-5 |
|---|---|---|---|
| RRF | [1,0 ; 1,0] | 40 % | 50 % |
| RRF | [0,5 ; 1,0] | 50 % | 55 % |
| RRF | [0,25 ; 1,0] | 75 % | 95 % |
| RRF | [0,1 ; 1,0] | 85 % | 95 % |
| RRF | [0,0 ; 1,0] | 95 % | 95 % |
| Ponderee | [1,0 ; 1,0] | 45 % | 95 % |
| **Ponderee** | **[0,5 ; 1,0]** | **95 %** | **95 %** |
| Ponderee | [0,25 ; 1,0] | 95 % | 95 % |

La fusion PONDEREE resiste bien mieux que RRF : la normalisation min-max donne 1,0
au premier de BM25, qui reste donc en tete. Sur ce type de requete, ponderee [0,5 ; 1]
egale BM25 seul en top-1 tout en conservant l'apport du vectoriel.

## RECHERCHE VECTORIELLE PAR DIRECTION LINGUISTIQUE
Verite terrain gratuite : le numero d'article est partage entre editions et entre langues.

| Direction | recall@1 | recall@5 | Lecture |
|---|---|---|---|
| FR -> FR (deux editions francaises) | 96 % | 96 % | **plafond** : methode et index sains |
| AR -> AR (deux editions arabes) | 56 % | 68 % | nettement degrade |
| FR -> AR (cross-lingue) | — | 15 % | anormal pour bge-m3 |
| FR -> AR, BM25 | — | 0 % | attendu : aucun token partage |

FR->FR a 96 % prouve que ni la methode ni l'index ne sont en cause. La degradation
est cote arabe, et elle s'aggrave en cross-lingue.

Hypothese en cours de test : on embedde le champ `text_norm`, c'est-a-dire de
l'arabe VOLONTAIREMENT non standard (conflation lam-alef). BM25 s'en moque — il
compare des chaines — mais le modele d'embedding n'a jamais vu cette orthographe
a l'entrainement. La normalisation qui AIDE BM25 pourrait NUIRE au vectoriel.

## DIAGNOSTIC DU CROSS-LINGUE — la metrique etait cassee, pas le modele

Le 15 % initial en FR->AR etait un ARTEFACT DE MESURE. Re-mesure proprement
sur 40 requetes en recherche exacte : 45 %.

### Ce qui occupe reellement le top-5 (index mixte, requete francaise, 40 requetes)
| Contenu du top-5 | Part |
|---|---|
| chunks FR d'AUTRES articles | **73 %** |
| chunks FR du bon article | 19 % |
| chunks AR du bon article (la cible comptee) | **7 %** |
| chunks AR d'autres articles | 1 % |

La requete etant en francais, les chunks francais gagnent par proximite de
LANGUE et non par pertinence. La version arabe du bon article n'est pas
introuvable : elle est EVINCEE. On comptait donc un echec alors que le systeme
retrouvait le bon article dans 92 % des cas, simplement dans l'autre langue.

### HNSW est innocente
| Recherche | recall@5 FR->AR |
|---|---|
| exacte (brute force) | 45 % |
| HNSW ef_search=64 | 45 % |
| HNSW ef_search=128 | 45 % |
| HNSW ef_search=512 | 45 % |

Aucune difference : l'approximation n'y est pour rien. Utile a savoir avant de
passer des heures a regler l'index.

### Le correctif : quota par langue
| Politique de selection du top-5 | cible AR | bon article (toute langue) |
|---|---|---|
| aucune (top-5 brut) | 45 % | 92 % |
| **quota 3 FR + 2 AR** | **90 %** | **100 %** |
| filtre strict langue = AR | 95 % | 95 % |

Le quota DOUBLE le recall cross-lingue sans rien perdre — le bon article passe
meme de 92 % a 100 %, parce qu'on cesse de gaspiller cinq places sur une seule
langue. Le filtre strict fait un point de mieux sur la cible arabe mais exige
de connaitre la langue voulue a l'avance ; le quota ne suppose rien.

### Le quota ne coute rien sur les requetes par identifiant
| Configuration | top-1 | top-5 |
|---|---|---|
| BM25 seul | 95 % | 95 % |
| ponderee [0,5 ; 1] sans quota | 95 % | 95 % |
| ponderee [0,5 ; 1] + quota 2 par langue | 95 % | 95 % |

Conclusion : quota par langue adopte par defaut. Meme mecanisme applicable au
biais de VOLUME entre sources (une source de 3 000 chunks evince une source de
200 pour exactement la meme raison). Implemente dans src/sources.py
(balanced_merge), 9 tests unitaires dans tests/test_sources.py.

## LES METRIQUES — et un piege decouvert en les validant

Implementees depuis les formules dans src/metrics.py : recall@k, precision@k,
hit@k, MRR, NDCG@k, plus evaluate_retrieval() et compare_runs().
24 tests unitaires.

### Le piege : recall@1 est PLAFONNE par le nombre de passages pertinents
Premiere execution sur les requetes « article N » (4,2 passages pertinents par
requete, car le meme article existe en 2 editions FR et 2 editions AR) :

| Configuration | recall@1 | recall@10 | MRR | Diagnostic automatique (1re version) |
|---|---|---|---|---|
| BM25 | 0,30 | 0,93 | **0,95** | « classement — un reranker devrait aider » |

Diagnostic FAUX. Le MRR de 0,95 dit que le premier resultat etait le bon dans
95 % des cas : il n'y a aucun probleme de classement. L'ecart venait d'un
plafond mecanique — avec 4,2 passages pertinents, recall@1 ne peut pas
depasser 1/4,2 = 0,24, meme avec un systeme parfait.

    >>> recall@k n'est comparable entre requetes que si elles ont un nombre
    >>> similaire de passages pertinents. hit@k et MRR le sont toujours.

Correctifs apportes :
  - ajout de hit@k (1 si au moins un bon passage dans le top-k) — souvent la
    metrique la plus parlante pour du RAG : le LLM n'a besoin que d'UN bon passage
  - le diagnostic automatique se base desormais sur hit@1 vs hit@kmax
  - evaluate_retrieval() expose `pertinents_par_requete`, sans quoi le recall
    n'est pas interpretable

### Apres correction, sur les memes donnees
| Configuration | hit@1 | hit@5 | MRR | recall@10 | Diagnostic |
|---|---|---|---|---|---|
| BM25 | 0,95 | 0,95 | 0,95 | 0,93 | sain |
| Ponderee [0,5 ; 1] | 0,95 | 0,95 | 0,95 | 0,92 | sain |
| Vectoriel | 0,10 | 0,10 | 0,11 | 0,08 | representation |

hit@1 = 0,95 retrouve exactement la mesure initiale faite a la main (95 %) :
le harnais et la mesure manuelle concordent.

### Le diagnostic automatique et ce qu'il prescrit
| Signature | Cause | Levier |
|---|---|---|
| hit@kmax bas, hit@1 ~ hit@kmax | representation | changer de mecanisme (ex. ajouter BM25) |
| hit@1 bas, hit@kmax haut | classement | un reranker |
| hit@1 ~ hit@kmax, les deux hauts | sain | rien a faire |

Le vectoriel sur « article N » est le cas « representation » : 10 % en top-1
comme en top-5. Aucun reglage d'ef_search n'aurait aide — il fallait BM25.
Savoir lire cet ecart evite des heures de reglage inutile.

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
