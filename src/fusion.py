"""
Fusion de resultats : combiner plusieurs listes de recherche en une seule.

LE PROBLEME
-----------
Le vectoriel et BM25 renvoient chacun une liste classee, mais leurs scores
n'ont AUCUNE unite commune :

    vectoriel : 0,58  0,54  0,50      (cosinus, borne dans [-1, 1])
    BM25      : 15,5  15,3   9,1      (non borne, depend du corpus)

Additionner ces scores revient a additionner des metres et des grammes.
BM25 ecrase mecaniquement le vectoriel, qui ne sert alors plus a rien.

DEUX SOLUTIONS
--------------
1. RRF — ignorer les scores, ne garder que les RANGS.
   Un rang est comparable entre listes : "premier" veut dire la meme chose
   partout. Aucun reglage necessaire, aucune normalisation.

2. Fusion ponderee — normaliser les scores (min-max) puis ponderer.
   Plus expressif (on peut privilegier une voie), mais il faut choisir le
   poids, et la normalisation min-max depend du lot de resultats : les memes
   documents peuvent recevoir des scores differents selon ce qui les entoure.

On implemente les DEUX, parce que le choix se mesure, il ne se suppose pas.
"""
from __future__ import annotations

from collections import defaultdict

from .vectorstore import SearchResult

# Constante de l'article original (Cormack et al., 2009).
# Son role : aplatir les ecarts entre rangs.
#   sans k (1/rang)     : rang 1 = 1,000  rang 2 = 0,500  rang 3 = 0,333
#   avec k=60           : rang 1 = 0,0164 rang 2 = 0,0161 rang 3 = 0,0159
# Avec k=60, ce qui compte est d'etre PRESENT dans plusieurs listes, pas
# d'etre premier dans une seule — sinon la tete d'une liste dominerait tout
# et on retomberait dans le probleme de depart.
RRF_K = 60


def reciprocal_rank_fusion(listes: list[list[SearchResult]], k: int = RRF_K,
                           top_n: int = 30, poids: list[float] | None = None) -> list[SearchResult]:
    """Fusion par rangs.

        score(doc) = somme_sur_listes  poids_liste / (k + rang)

    Les rangs sont comptes a partir de 1 (un rang 0 donnerait un poids
    artificiellement eleve au premier element).

    Propriete centrale : un document trouve par PLUSIEURS moteurs, meme sans
    etre premier, bat un document trouve par un SEUL moteur en premiere
    position. Quand deux methodes tres differentes sont d'accord, c'est
    generalement le bon resultat.

    `poids` permet de ponderer les listes (ex. [1.0, 0.5] pour donner deux
    fois moins d'importance a BM25). Par defaut, toutes egales.
    """
    poids = poids or [1.0] * len(listes)
    if len(poids) != len(listes):
        raise ValueError("poids et listes doivent avoir la meme longueur")

    cumul: dict[str, float] = defaultdict(float)
    for liste, w in zip(listes, poids):
        for res in liste:
            cumul[res.chunk_id] += w / (k + res.rank + 1)

    classe = sorted(cumul.items(), key=lambda x: -x[1])
    return [SearchResult(cid, score, rang) for rang, (cid, score) in enumerate(classe[:top_n])]


def _minmax(valeurs: list[float]) -> list[float]:
    """Ramene une liste de scores dans [0, 1].

    Limite a connaitre : la normalisation depend du LOT. Le meme document
    peut obtenir 1,0 dans un lot et 0,3 dans un autre, selon ses voisins.
    Les scores fusionnes ne sont donc pas comparables d'une requete a l'autre
    — c'est le defaut principal de cette approche face a RRF.
    """
    if not valeurs:
        return []
    lo, hi = min(valeurs), max(valeurs)
    if hi - lo < 1e-12:
        return [1.0] * len(valeurs)
    return [(v - lo) / (hi - lo) for v in valeurs]


def weighted_score_fusion(listes: list[list[SearchResult]], poids: list[float] | None = None,
                          top_n: int = 30) -> list[SearchResult]:
    """Fusion par scores normalises puis ponderes.

    L'alternative a RRF. A comparer sur les deux types de requetes :
      - questions de sens        -> le vectoriel devrait porter le resultat
      - identifiants exacts      -> BM25 devrait porter le resultat
    Une methode qui gagne sur un type et perd sur l'autre n'est pas un bon
    choix par defaut : c'est exactement ce que le tableau comparatif revelera.
    """
    poids = poids or [1.0] * len(listes)
    cumul: dict[str, float] = defaultdict(float)
    for liste, w in zip(listes, poids):
        norm = _minmax([r.score for r in liste])
        for res, s in zip(liste, norm):
            cumul[res.chunk_id] += w * s

    classe = sorted(cumul.items(), key=lambda x: -x[1])
    return [SearchResult(cid, score, rang) for rang, (cid, score) in enumerate(classe[:top_n])]


def hybrid_search(query: str, query_vec, store, bm25, top_k: int = 30,
                  ef_search: int = 128, methode: str = "rrf",
                  poids: list[float] | None = None,
                  allowed_ids: set[str] | None = None) -> list[SearchResult]:
    """Recherche hybride complete : les deux moteurs, puis fusion.

    Point important : chaque moteur est interroge sur un top-k ELARGI
    (2 x top_k). Fusionner deux listes de 30 pour en garder 30 laisse peu de
    marge au consensus — un document doit pouvoir figurer dans les deux
    listes pour que RRF joue son role.
    """
    k_large = top_k * 2
    l_vec = store.search(query_vec, k=k_large, ef_search=ef_search, allowed_ids=allowed_ids)
    l_bm = bm25.search(query, k=k_large, allowed_ids=allowed_ids)

    if methode == "rrf":
        return reciprocal_rank_fusion([l_vec, l_bm], top_n=top_k, poids=poids)
    if methode == "weighted":
        return weighted_score_fusion([l_vec, l_bm], poids=poids, top_n=top_k)
    if methode == "vector":
        return l_vec[:top_k]
    if methode == "bm25":
        return l_bm[:top_k]
    raise ValueError(f"methode inconnue : {methode}")
