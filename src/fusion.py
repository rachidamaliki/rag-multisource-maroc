"""
ARC 4 (partie 2) — La fusion des resultats.

Concept a acquerir :
  Vous avez deux listes de resultats : une du vectoriel (scores 0 a 1),
  une de BM25 (scores 0 a 30, sans borne). Les additionner n'a AUCUN sens :
  les echelles sont incomparables. C'est le piege documente dans le roadmap.

  La RRF (Reciprocal Rank Fusion) resout ca elegamment : on ignore les
  scores et on ne garde que les RANGS.

      RRF(doc) = somme_sur_listes  1 / (k + rang_du_doc_dans_la_liste)

  avec k ~ 60. Un document classe 1er dans une liste et 50e dans l'autre
  bat un document classe 10e partout. Simple, robuste, sans calibrage.
"""
from __future__ import annotations


def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60, top_n: int = 30) -> list[tuple[str, float]]:
    """
    TODO (J12) — a ecrire a la main. C'est 15 lignes, et c'est exactement
    le genre de chose qu'un framework fait pour vous sans que vous sachiez
    qu'elle existe.

    result_lists : plusieurs listes d'identifiants de chunks, chacune deja triee
    retourne : [(chunk_id, score_rrf), ...] trie par score decroissant
    """
    raise NotImplementedError("Arc 4")


def weighted_score_fusion(vector_results, bm25_results, alpha: float = 0.5, top_n: int = 30):
    """
    TODO (J12) — l'alternative : normaliser les scores (min-max) puis
    ponderer. A comparer avec la RRF sur vos deux types de requetes.

    Question a laquelle vous devrez repondre en entretien :
    laquelle generalise le mieux, et pourquoi ?
    """
    raise NotImplementedError("Arc 4")
