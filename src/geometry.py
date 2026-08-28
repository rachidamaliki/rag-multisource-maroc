"""
ARC 0 — La geometrie des embeddings.

C'est le seul fichier ou TOUT doit etre ecrit a la main, en numpy.
Interdiction d'importer sklearn ici (sauf dans les tests, pour verifier).

Concept a acquerir :
  Un embedding est un point dans un espace a N dimensions. "Similaire"
  = "pointe dans la meme direction". La similarite cosinus mesure
  l'angle, pas la distance : deux textes de longueurs tres differentes
  mais de meme sens pointent dans la meme direction, alors que leur
  distance euclidienne serait grande. C'est pour ca que le cosinus
  gagne sur du texte.
"""
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    TODO (J1) — A implementer a la main.

    a : (d,) ou (n, d)
    b : (d,) ou (m, d)
    retourne : scalaire, (n,) ou (n, m)

    Rappel de la formule : cos(a, b) = (a . b) / (||a|| * ||b||)

    Pieges :
      - normaliser AVANT le produit scalaire evite de tout recalculer
      - division par zero si un vecteur est nul -> ajouter un epsilon
      - utilisez le broadcasting numpy, PAS de boucle for
    """
    raise NotImplementedError("Arc 0 — a vous de jouer")


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TODO (J1) — pour comparer avec le cosinus et comprendre la difference."""
    raise NotImplementedError("Arc 0")


def normalize(x: np.ndarray) -> np.ndarray:
    """TODO (J1) — normalisation L2. Une fois normalise, cosinus == produit scalaire."""
    raise NotImplementedError("Arc 0")


def brute_force_knn(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int = 5):
    """
    TODO (J2) — BOSS FIGHT Arc 0.

    Recherche exhaustive des k plus proches voisins, sans aucune librairie.
    Retourne (indices, scores) tries par score decroissant.

    C'est votre VERITE TERRAIN. En Arc 3, l'index HNSW sera compare a ce
    resultat : la difference, c'est le "recall loss" de la recherche
    approximative. Sans cette baseline, vous ne pouvez rien prouver.

    Indice : np.argpartition est plus rapide que np.argsort pour un top-k.
    """
    raise NotImplementedError("Arc 0 — boss fight")
