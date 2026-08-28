"""
ARC 3 — Le stockage vectoriel et la recherche approximative (ANN).

Concept a acquerir :
  Chercher exhaustivement dans 10 000 vecteurs est instantane. Dans
  10 millions, c'est impossible en temps reel. HNSW construit un graphe
  en couches : on "atterrit" grossierement dans la bonne region, puis on
  affine. On echange de la PRECISION contre de la VITESSE.

  Le parametre `ef_search` controle cet echange. La valeur par defaut
  n'est presque jamais la bonne pour vos donnees. Le boss fight de l'arc
  consiste a tracer la courbe et a trouver VOTRE point d'equilibre.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    rank: int


class BruteForceStore:
    """Verite terrain. Reutilise brute_force_knn de l'Arc 0.
    C'est la reference contre laquelle HNSW sera juge."""

    def __init__(self, vectors: np.ndarray, chunk_ids: list[str]):
        self.vectors = vectors
        self.chunk_ids = chunk_ids

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[SearchResult]:
        """TODO (J9) — appeler src.geometry.brute_force_knn et emballer le resultat."""
        raise NotImplementedError("Arc 3")


class FaissHNSWStore:
    """Index HNSW local via FAISS. Gratuit, aucun serveur a installer."""

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200):
        self.dim, self.m, self.ef_construction = dim, m, ef_construction
        self.index = None
        self.chunk_ids: list[str] = []

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        """
        TODO (J9) — construire l'index.
        Indice : faiss.IndexHNSWFlat(dim, m) ; index.hnsw.efConstruction = ...
        Les vecteurs doivent etre normalises L2 pour que le produit
        scalaire equivaille au cosinus (voir Arc 0).
        """
        raise NotImplementedError("Arc 3")

    def search(self, query_vec: np.ndarray, k: int = 10, ef_search: int = 64) -> list[SearchResult]:
        """TODO (J9) — ne pas oublier de regler index.hnsw.efSearch AVANT la requete."""
        raise NotImplementedError("Arc 3")


class PgVectorStore:
    """
    Variante Postgres + pgvector (optionnelle mais recommandee).
    Pourquoi : en entreprise, on stocke rarement les vecteurs a part.
    Les avoir dans la meme base que les metadonnees permet de filtrer
    (par date, par source, par type de document) DANS la requete SQL.

    Lancer la base (une seule commande, gratuite) :
      docker run -d --name pgvector -p 5432:5432 \
        -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=ragpass -e POSTGRES_DB=ragzero \
        pgvector/pgvector:pg16
    """

    def __init__(self, dsn: str, dim: int):
        self.dsn, self.dim = dsn, dim

    def setup(self) -> None:
        """TODO (J9) — CREATE EXTENSION vector; CREATE TABLE chunks(...); CREATE INDEX ... USING hnsw."""
        raise NotImplementedError("Arc 3")

    def insert(self, chunks, vectors: np.ndarray) -> None:
        raise NotImplementedError("Arc 3")

    def search(self, query_vec: np.ndarray, k: int = 10, filters: dict | None = None):
        """TODO (J9) — checklist 4 : mesurer le surcout du filtrage par metadonnees."""
        raise NotImplementedError("Arc 3")


def recall_loss_curve(brute: BruteForceStore, ann: FaissHNSWStore,
                      queries: np.ndarray, ef_values: list[int], k: int = 10):
    """
    TODO (J10) — BOSS FIGHT Arc 3 : "Exact vs Approximate Showdown".

    Pour chaque valeur d'ef_search :
      - lancer les memes requetes sur l'index exact et sur HNSW
      - calculer le recall@k (proportion des vrais top-k retrouves)
      - mesurer la latence mediane (p50) et p95

    Puis tracer recall@10 en fonction de la latence. Le "coude" de la
    courbe est votre reglage optimal.

    >>> Ce graphique seul est un artefact de portfolio legitime. <<<
    Sauvegardez-le dans reports/figures/arc3_recall_vs_latency.png
    """
    raise NotImplementedError("Arc 3 — boss fight")
