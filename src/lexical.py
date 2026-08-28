"""
ARC 4 (partie 1) — La recherche lexicale : BM25.

Concept a acquerir :
  La recherche vectorielle comprend le SENS mais floute les details.
  Demandez "article 145" : le vectoriel vous renverra des articles qui
  "parlent de la meme chose", pas forcement le 145. BM25 compte les mots
  exacts — il ne comprend rien, mais il ne se trompe pas sur un identifiant.

  Sur un corpus juridique ou comptable (references d'articles, codes
  comptables, numeros de comptes), c'est decisif. C'est LA raison pour
  laquelle l'hybride existe.

Formule BM25 (a implementer, pas a copier d'une librairie) :
    score(D, Q) = somme_sur_termes  IDF(qi) * ( f(qi,D) * (k1+1) )
                                    / ( f(qi,D) + k1 * (1 - b + b*|D|/avgdl) )
    IDF(qi) = ln( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )

  k1 (~1.5) controle la saturation : le 10e "impot" dans un document
  n'apporte presque rien de plus que le 3e.
  b (~0.75) controle la normalisation par la longueur du document.
"""
from __future__ import annotations
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """
    TODO (J11) — tokeniseur simple mais adapte FR + AR.
    Points d'attention :
      - l'arabe ne se decoupe pas comme le francais (pas de majuscules,
        articles colles : "الشركة" = "ال" + "شركة")
      - normaliser les chiffres et les references ("art.145", "Art 145")
      - minuscules + suppression de la ponctuation pour le francais
    """
    raise NotImplementedError("Arc 4")


class BM25:
    """TODO (J11) — implementation depuis la formule. `rank_bm25` est
    installe UNIQUEMENT pour verifier que vos scores correspondent."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        raise NotImplementedError("Arc 4")

    def search(self, query: str, k: int = 10):
        raise NotImplementedError("Arc 4")
