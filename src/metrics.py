"""
ARC 6 (partie 1) — Les metriques de retrieval.

C'EST LE FICHIER LE PLUS IMPORTANT DU PROJET.
Sans lui, tous les choix des Arcs 1 a 5 sont des opinions.
Avec lui, ce sont des decisions justifiees par des chiffres.

Toutes ces metriques doivent etre ecrites DEPUIS LA FORMULE.
Le jour ou un recruteur demande "c'est quoi le MRR ?", vous ne voulez pas
repondre "c'est une fonction de la librairie".
"""
from __future__ import annotations
import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    TODO (J8) — "Parmi les documents vraiment pertinents, quelle proportion
    ai-je retrouvee dans mon top-k ?"

        recall@k = |pertinents ∩ top_k| / |pertinents|

    C'est la metrique reine du retrieval : si le bon passage n'est pas
    remonte, aucun LLM au monde ne pourra repondre correctement.
    """
    raise NotImplementedError("Arc 6")


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """TODO (J8) — "Parmi ce que j'ai remonte, quelle proportion est pertinente ?" """
    raise NotImplementedError("Arc 6")


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    TODO (J8) — Mean Reciprocal Rank : 1 / (rang du premier resultat pertinent).

    Le bon resultat en 1re position -> 1.0 ; en 5e position -> 0.2.
    Le MRR punit le fait d'avoir raison "trop tard" dans la liste.
    Complementaire du recall : on peut avoir un bon recall et un mauvais MRR.
    """
    raise NotImplementedError("Arc 6")


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """
    TODO (J8) — NDCG : la seule metrique qui gere la pertinence GRADUEE
    (tres pertinent / moyennement / pas du tout), avec un amortissement
    logarithmique selon la position.

        DCG@k  = somme  rel_i / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k     (IDCG = DCG du classement ideal)
    """
    raise NotImplementedError("Arc 6")


def evaluate_retrieval(run: dict[str, list[str]], golden: dict[str, set[str]], k_values=(1, 3, 5, 10)) -> dict:
    """
    TODO (J8) — agrege les metriques sur tout le golden dataset.
    run    : {query_id: [chunk_ids classes]}
    golden : {query_id: {chunk_ids pertinents}}
    retour : {"recall@5": 0.78, "mrr": 0.64, ...}
    """
    raise NotImplementedError("Arc 6")
