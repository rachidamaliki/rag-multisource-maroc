# Sources du corpus

> Les PDF eux-memes ne sont **pas** versionnes (`.gitignore`) : ce sont des textes officiels
> publics, republiables mais volumineux. Ce fichier documente leur provenance pour que
> n'importe qui puisse reconstituer le corpus a l'identique.

## `code_travail/` — Code du travail marocain (loi 65-99)

Telecharge le 2026-08-29.

| Fichier | Source | Langue | Pages | Etat |
|---|---|---|---|---|
| `code_travail_fr_consolide_2011.pdf` | casainvest.ma | FR | 206 | ✅ **reference** — version consolidee au 26/10/2011 |
| `code_travail_fr_bo_2004.pdf` | webapps.ilo.org (OIT) | FR | 119 | ✅ Bulletin Officiel n° 5210 d'origine |
| `code_travail_ar_miepeec.pdf` | miepeec.gov.ma (ministere) | AR | — | ✅ **reference arabe** — extraction la plus propre |
| `code_travail_ar_justice_2021.pdf` | adala.justice.gov.ma | AR | 156 | ⚠️ version 2021, extraction plus bruitee |

### Diagnostic d'extraction (fait avant le J3)

Tous les fichiers sont des **PDF natifs** : le texte est extractible directement, aucun OCR
n'est necessaire. C'est une journee de travail economisee.

**Cote francais** — propre. 588 articles distincts detectes automatiquement par le motif
`Article \d+`, sur les 589 que compte la loi. Le chunker structurel du J5 aura une unite
atomique nette sur laquelle s'appuyer.

**Cote arabe** — utilisable, mais bruite. Deux problemes mesures, a traiter dans `clean()` au J3 :

1. **Ligature lam-alef decomposee.** `المقاولات` ressort en `المقاوالت`, `الاستغلالات` en
   `االستغالالت`. C'est le probleme classique des PDF arabes : la ligature `لا` est encodee
   comme deux caracteres separes. Sans normalisation, ces mots ne seront **jamais** retrouves
   par BM25 (J11) et seront mal embeddes (J6).
2. **Espaces parasites a l'interieur des mots.** Taux de fragmentation mesure
   (proportion de mots arabes de 2 lettres ou moins, qui sont presque toujours des morceaux) :

   | Fichier | Fragmentation |
   |---|---|
   | `code_travail_ar_miepeec.pdf` | **20,8 %** |
   | `code_travail_ar_justice_2021.pdf` | 39,6 % |

   D'ou le choix de la version du ministere comme reference arabe.

3. **Reperage des articles en arabe.** Le motif `المادة \d+` ne detecte qu'une partie des
   articles (153 sur la version justice, 31 sur miepeec) alors que le mot `المادة` apparait
   341 fois. Cause : l'ordre RTL place parfois le numero avant le mot a l'extraction.
   `extract_unit_refs()` (J5) devra gerer les deux ordres.

### Ce que ca implique pour le projet

Ces trois defauts ne sont pas des accidents a contourner : ce sont **les donnees reelles**.
Un RAG arabe qui ne normalise pas la ligature lam-alef a un recall silencieusement degrade,
sans qu'aucune erreur ne se declenche. Le mesurer avant/apres normalisation est un resultat
publiable dans le README final — et c'est le genre de detail que personne ne traite dans les
projets RAG anglophones.

## `cgnc/` — Code General de Normalisation Comptable

A telecharger au J9.

## `dgi_circulaires/` — Circulaires et notes de la DGI

A telecharger au J9. Format mixte PDF + HTML, avec des dates de publication : c'est cette
source qui alimente les questions de type `conflict` (une circulaire recente qui actualise
un texte plus ancien).
