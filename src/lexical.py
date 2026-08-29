"""
BM25 — la recherche par mots exacts, implementee depuis la formule.

POURQUOI ON EN A BESOIN ALORS QU'ON A DES EMBEDDINGS
-----------------------------------------------------
Requete : "article 145".
Le vectoriel transforme la requete en un point de l'espace et cherche des
passages "semantiquement proches". Or "article 145", "article 146" et
"article 43" occupent quasiment le meme point : ce sont tous "un article de
loi avec un numero". Le vectoriel renverra un article au hasard parmi des
dizaines.

BM25 cherche le token `145`. Il le trouve.

    Le vectoriel comprend le sens mais floute les details.
    BM25 ignore le sens mais ne se trompe jamais sur un identifiant.

Sur un corpus juridique et comptable — numeros d'articles, codes de comptes,
references de circulaires — c'est decisif. C'est toute la raison d'etre de la
recherche hybride.

LA FORMULE
----------
    score(D, Q) = somme_q  IDF(q) * ( f(q,D) * (k1+1) )
                            / ( f(q,D) + k1 * (1 - b + b*|D|/avgdl) )

    IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )

Trois idees, chacune corrigeant un defaut de la methode naive :

  1. SATURATION (k1 ~ 1.5) — un mot present 10 fois n'est pas 10 fois plus
     pertinent que present 1 fois. La contribution plafonne :
         1 occurrence  -> 1.00      5 occurrences -> 1.92
         2 occurrences -> 1.43     50 occurrences -> 2.42

  2. RARETE (IDF) — "de" est dans tous les documents, il ne distingue rien
     (poids ~ 0). "preavis" est dans 3 % des documents, il est tres
     discriminant (poids eleve). Les mots vides s'annulent d'eux-memes,
     sans liste de stop-words.

  3. LONGUEUR (b ~ 0.75) — un document de 5 000 mots contient mecaniquement
     plus d'occurrences qu'un document de 100 mots. On corrige par la
     longueur relative au document moyen.

DEPENDANCE CRITIQUE A LA NORMALISATION
--------------------------------------
BM25 compare des CHAINES, pas des sens. Si le corpus contient `المقاوالت` et
la requete `المقاولات`, le score est exactement 0 — le mot n'existe pas pour
lui. C'est binaire. D'ou l'importance de src/arabic.py : on indexe et on
interroge toujours la version normalisee (`text_norm`).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .arabic import normalize_for_search

# Un "mot" = lettres latines (avec accents), lettres arabes, ou chiffres.
# Les chiffres sont conserves tels quels : "145" est le token le plus utile
# du corpus juridique.
MOT = re.compile(r"[a-zA-ZÀ-ÿ]+|[؀-ۿ]+|\d+")

# Mots vides. Note : BM25 les neutralise deja par l'IDF (ils apparaissent
# partout, donc leur poids tombe a ~0). Les retirer reduit surtout la taille
# de l'index et le temps de calcul.
STOPWORDS_FR = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "a", "au",
    "aux", "en", "dans", "par", "pour", "sur", "sous", "avec", "sans", "ce",
    "cet", "cette", "ces", "qui", "que", "quoi", "dont", "est", "sont", "etre",
    "il", "elle", "ils", "elles", "son", "sa", "ses", "leur", "leurs", "ne",
    "pas", "plus", "se", "y", "d", "l", "s", "n", "c", "j", "m", "t",
}
STOPWORDS_AR = {
    "من", "في", "علي", "الي", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "ما", "لا", "ان", "او", "كل", "بعد", "قبل", "عند", "هو", "هي", "به", "له",
}
STOPWORDS = STOPWORDS_FR | STOPWORDS_AR


def tokenize(text: str, normalize: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Texte -> liste de tokens comparables.

    normalize=True applique la normalisation arabe et le passage en minuscules.
    A appliquer IDENTIQUEMENT au corpus et aux requetes, sans exception :
    c'est le principe de coherence. Une asymetrie ici casse la recherche
    sans lever la moindre erreur.
    """
    if normalize:
        text = normalize_for_search(text)
    toks = [t.lower() for t in MOT.findall(text)]
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


class BM25:
    """Implementation depuis la formule. `rank_bm25` sert uniquement de
    reference de verification, jamais en production ici."""

    # Deux variantes d'IDF coexistent dans la litterature :
    #
    #   "robertson" (BM25 original, 1994) :
    #       ln( (N - n + 0.5) / (n + 0.5) )
    #       Probleme : devient NEGATIF pour un mot present dans plus de la
    #       moitie du corpus. Un mot tres frequent peut alors FAIRE BAISSER
    #       le score d'un document qui le contient — contre-intuitif.
    #
    #   "lucene" (variante moderne, celle de Lucene/Elasticsearch) :
    #       ln( (N - n + 0.5) / (n + 0.5) + 1 )
    #       Le +1 garantit un IDF toujours positif.
    #
    # On retient "lucene" par defaut, mais la variante reste selectionnable :
    # c'est ce qui permet de verifier l'implementation contre rank_bm25, qui
    # utilise "robertson". Les deux donnent le MEME CLASSEMENT ; seules les
    # valeurs absolues des scores different.
    def __init__(self, corpus_tokens: list[list[str]], chunk_ids: list[str] | None = None,
                 k1: float = 1.5, b: float = 0.75, idf_variant: str = "lucene"):
        self.k1, self.b, self.idf_variant = k1, b, idf_variant
        self.docs = corpus_tokens
        self.chunk_ids = chunk_ids or [str(i) for i in range(len(corpus_tokens))]
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

        # index inverse : mot -> {doc_index: frequence}
        # C'est la structure qui rend la recherche rapide : au lieu de
        # parcourir tous les documents, on ne visite que ceux contenant
        # au moins un mot de la requete.
        self.inverse: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(corpus_tokens):
            for mot, freq in Counter(doc).items():
                self.inverse.setdefault(mot, {})[i] = freq

        # IDF pre-calcule : il ne depend que du corpus, pas de la requete.
        plus_un = 1 if idf_variant == "lucene" else 0
        self.idf = {
            mot: math.log((self.N - len(postings) + 0.5) / (len(postings) + 0.5) + plus_un)
            for mot, postings in self.inverse.items()
        }

    def score_doc(self, tokens_requete: list[str], i: int) -> float:
        s = 0.0
        norm_len = 1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-9)
        for mot in tokens_requete:
            postings = self.inverse.get(mot)
            if not postings or i not in postings:
                continue
            f = postings[i]
            s += self.idf[mot] * (f * (self.k1 + 1)) / (f + self.k1 * norm_len)
        return s

    def search(self, query: str, k: int = 10, allowed_ids: set[str] | None = None):
        from .vectorstore import SearchResult

        q = tokenize(query)
        candidats: set[int] = set()
        for mot in q:
            candidats |= set(self.inverse.get(mot, {}))

        scores = [(i, self.score_doc(q, i)) for i in candidats]
        scores = [(i, s) for i, s in scores if s > 0]
        if allowed_ids is not None:
            scores = [(i, s) for i, s in scores if self.chunk_ids[i] in allowed_ids]
        scores.sort(key=lambda x: -x[1])

        return [SearchResult(self.chunk_ids[i], float(s), r)
                for r, (i, s) in enumerate(scores[:k])]

    def __len__(self):
        return self.N
