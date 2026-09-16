# Audit du golden dataset

50 questions — 12 `exact_match`, 19 `semantic`, 7 `semantic_ar`, 12 `unanswerable`

- controles bloquants en echec : **0**
- alertes de recopie (seuil 50%) : **0**

| qid | type | article | preuve | recopie | langues pertinentes |
|---|---|---|---|---|---|
| s01 | semantic | art. 13 | OK | 45% | ar+fr |
| s02 | semantic | art. 14 | OK | 0% | ar+fr |
| s03 | semantic | art. 14 | OK | 14% | ar+fr |
| s04 | semantic | art. 16 | OK | 14% | fr |
| s05 | semantic | art. 34 | OK | 0% | fr |
| s06 | semantic | art. 40 | OK | 11% | fr |
| s07 | semantic | art. 61 | OK | 0% | fr |
| s08 | semantic | art. 52 | OK | 14% | ar+fr |
| s09 | semantic | art. 53 | OK | 29% | ar+fr |
| s10 | semantic | art. 62 | OK | 0% | ar+fr |
| s11 | semantic | art. 72 | OK | 0% | ar+fr |
| s12 | semantic | art. 143 | OK | 25% | ar+fr |
| s13 | semantic | art. 152 | OK | 20% | fr |
| s14 | semantic | art. 153 | OK | 14% | fr |
| s15 | semantic | art. 161 | OK | 12% | ar+fr |
| s16 | semantic | art. 184 | OK | 17% | ar+fr |
| s17 | semantic | art. 201 | OK | 17% | ar+fr |
| s18 | semantic | art. 205 | OK | 0% | ar+fr |
| s19 | semantic | art. 231 | OK | 38% | ar+fr |
| e01 | exact_match | art. 13 | OK | 50% | ar+fr |
| e02 | exact_match | art. 14 | OK | 50% | ar+fr |
| e03 | exact_match | art. 16 | OK | 33% | fr |
| e04 | exact_match | art. 34 | OK | 100% | fr |
| e05 | exact_match | art. 40 | OK | 50% | fr |
| e06 | exact_match | art. 52 | OK | 50% | ar+fr |
| e07 | exact_match | art. 61 | OK | 25% | fr |
| e08 | exact_match | art. 72 | OK | 50% | ar+fr |
| e09 | exact_match | art. 143 | OK | 100% | ar+fr |
| e10 | exact_match | art. 152 | OK | 50% | fr |
| e11 | exact_match | art. 184 | OK | 50% | ar+fr |
| e12 | exact_match | art. 205 | OK | 50% | ar+fr |
| a01 | semantic_ar | art. 143 | OK | n/a | ar+fr |
| a02 | semantic_ar | art. 184 | OK | n/a | ar+fr |
| a03 | semantic_ar | art. 205 | OK | n/a | ar+fr |
| a04 | semantic_ar | art. 231 | OK | n/a | ar+fr |
| a05 | semantic_ar | art. 161 | OK | n/a | ar+fr |
| a06 | semantic_ar | art. 201 | OK | n/a | ar+fr |
| a07 | semantic_ar | art. 72 | OK | n/a | ar+fr |
| u01 | unanswerable | - | absent | - | - |
| u02 | unanswerable | - | absent | - | - |
| u03 | unanswerable | - | absent | - | - |
| u04 | unanswerable | - | absent | - | - |
| u05 | unanswerable | - | absent | - | - |
| u06 | unanswerable | - | absent | - | - |
| u07 | unanswerable | - | absent | - | - |
| u08 | unanswerable | - | absent | - | - |
| u09 | unanswerable | - | absent | - | - |
| u10 | unanswerable | - | absent | - | - |
| u11 | unanswerable | - | absent | - | - |
| u12 | unanswerable | - | absent | - | - |

## Limites connues

- Types `cross_source` et `conflict` absents : ils exigent le CGNC et les circulaires DGI, pas encore ingeres.
- Les preuves sont verifiees sur la version francaise consolidee ; les chunks arabes du meme article sont comptes pertinents par correspondance de numero.
- Validation par preuve textuelle, non par un juriste : la reponse est garantie presente dans le texte, pas son interpretation juridique.
