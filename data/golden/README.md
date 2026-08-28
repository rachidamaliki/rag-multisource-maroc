# Golden dataset — le cœur du projet

## Pourquoi c'est le fichier le plus important du dépôt

Sans ce fichier, aucune affirmation du projet n'est vérifiable. Avec lui, chaque décision
d'architecture devient défendable avec un chiffre. C'est aussi la seule partie du projet
qu'on ne peut pas accélérer : elle demande de la lecture et du jugement humain.

## Objectif : 60 questions minimum

Le corpus étant **multi-source**, deux catégories s'ajoutent par rapport à un RAG classique —
et ce sont les deux qui font la valeur du projet.

| Type | Nombre | Ce que ça teste |
|---|---|---|
| `semantic` | 18 | questions reformulées, sans mot-clé commun avec le texte → le vectoriel |
| `exact_match` | 12 | numéros d'articles, rubriques, montants → BM25 |
| `cross_source` | 10 | **la réponse exige 2 sources différentes** → fusion équilibrée entre sources |
| `conflict` | 6 | **deux sources se contredisent** → règle d'autorité (loi > doctrine, récent > ancien) |
| `unanswerable` | 14 | le corpus ne peut PAS répondre → logique de refus (Arc 7) |

Ajoutez des questions en arabe dans chaque catégorie : c'est votre différenciation, et le seul
moyen de valider le multilingue.

### Les deux types qui font la différence

**`cross_source`** — exemple : *« Dans quelle rubrique comptable enregistrer une indemnité de
licenciement, et sur quelle base légale ? »* La réponse est dans le CGNC **et** dans le Code du
travail. Une recherche naïve remontera 5 passages de la source la plus volumineuse et zéro de
l'autre. C'est exactement le biais de volume que corrige `balanced_merge()` dans
[`src/sources.py`](../../src/sources.py) — et vous pourrez le **mesurer**.

**`conflict`** — exemple : une circulaire DGI de 2025 actualise un traitement décrit dans un
document plus ancien. Le système doit retenir la source la plus récente **et le signaler** dans
sa réponse. Un système qui tranche en silence est plus dangereux qu'un système qui hésite à
voix haute.

Ces deux catégories sont quasi absentes des projets RAG publics. Elles suffisent à distinguer
le vôtre.

## Champs

- `qid` : identifiant unique
- `question` : la question telle qu'un utilisateur réel la poserait
- `lang` : `fr` | `ar`
- `type` : voir tableau ci-dessus
- `source_expected` : la ou les sources qui doivent être mobilisées (liste vide si non répondable)
- `relevant_chunk_ids` : LES passages qui contiennent la réponse
- `expected_answer` : la réponse de référence (pour le juge LLM)
- `answerable` : booléen
- `authority_expected` : pour les `conflict` uniquement — la source qui doit l'emporter

## Méthode d'annotation (étalée : 30 le J7, 30 le J15)

1. Ouvrez un document, lisez un passage, et écrivez la question à laquelle il répond —
   **avec vos mots**, pas ceux du texte. Recopier le vocabulaire du passage rend la tâche
   artificiellement facile et fausse vos métriques.
2. Notez l'identifiant du chunk correspondant et la source.
3. Pour les `cross_source` : cherchez volontairement les points de contact entre vos sources
   (un même sujet traité sous deux angles — juridique et comptable, par exemple).
4. Pour les `unanswerable` : inventez des questions plausibles mais absentes du corpus
   (sujets voisins, articles inexistants).

Comptez 3 à 4 minutes par question — davantage pour les `cross_source`, qui demandent de
lire deux documents. 60 questions ≈ 4 heures, d'où l'étalement sur deux journées.

## Planning des sources

- **J7** : 30 questions sur `code_travail` seul (la seule source ingérée à ce stade)
- **J15** : 30 questions supplémentaires, dont **toutes** les `cross_source` et `conflict`,
  une fois les 3 sources en place
