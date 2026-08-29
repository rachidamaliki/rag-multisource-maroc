"""
Le stockage vectoriel : recherche exacte vs approximative.

LE PROBLEME
-----------
Chercher le plus proche voisin parmi 3 000 vecteurs : instantane, on compare
tout. Parmi 10 millions : impossible en temps reel. Il faut renoncer a
l'exhaustivite.

HNSW (Hierarchical Navigable Small World) construit un graphe en couches.
La couche du haut contient peu de points relies par de longues aretes : on
s'y deplace vite et grossierement. Les couches basses sont denses : on y
affine. On "atterrit" dans la bonne region, puis on precise — au lieu de
tout examiner.

CE QU'ON ECHANGE
----------------
De la PRECISION contre de la VITESSE. L'index approximatif peut manquer de
vrais voisins. Combien ? Personne ne peut le dire a priori : cela depend de
vos donnees et de vos parametres. Il faut le MESURER — c'est le role de
`recall_loss_curve()` plus bas, et l'une des figures du projet.

    ef_search : nombre de candidats explores pendant la recherche.
                Plus haut = plus precis et plus lent.
                La valeur par defaut n'est presque jamais la bonne.
    m         : nombre de connexions par noeud dans le graphe.
    ef_construction : effort mis a la construction de l'index.

LE PIEGE PRINCIPAL
------------------
Traiter l'index comme une boite noire qu'on ne regle jamais. Un `ef_search`
trop bas detruit le recall SILENCIEUSEMENT : aucune erreur, juste de moins
bons resultats. C'est invisible sans baseline exacte a laquelle se comparer.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import brute_force_knn


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    rank: int


class BruteForceStore:
    """Recherche exhaustive : la VERITE TERRAIN.

    Trop lente pour la production, indispensable pour l'evaluation : c'est la
    reference contre laquelle on mesure ce que l'index approximatif fait perdre.
    Sans elle, aucune affirmation sur la qualite de l'index n'est demontrable.
    """

    def __init__(self, vectors: np.ndarray, chunk_ids: list[str]):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunk_ids = list(chunk_ids)

    def search(self, query_vec: np.ndarray, k: int = 10, **_) -> list[SearchResult]:
        idx, scores = brute_force_knn(query_vec, self.vectors, k=k)
        return [SearchResult(self.chunk_ids[i], float(s), r)
                for r, (i, s) in enumerate(zip(np.atleast_1d(idx), np.atleast_1d(scores)))]

    def __len__(self):
        return len(self.chunk_ids)


class FaissHNSWStore:
    """Index HNSW local. Gratuit, aucun serveur.

    Note technique importante : on indexe des vecteurs NORMALISES avec un
    index de produit scalaire (INNER_PRODUCT). Sur des vecteurs de norme 1,
    produit scalaire == similarite cosinus. C'est la relation demontree au J1
    (dist^2 = 2(1-cos)) : normaliser permet d'utiliser l'operation la plus
    rapide sans changer le classement.
    """

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200):
        import faiss
        self.dim, self.m = dim, m
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.chunk_ids: list[str] = []

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        # securite : si les vecteurs ne sont pas normalises, le produit
        # scalaire ne correspond plus au cosinus et le classement change.
        normes = np.linalg.norm(v, axis=1)
        if not np.allclose(normes, 1.0, atol=1e-3):
            v = v / np.maximum(normes[:, None], 1e-12)
        self.index.add(v)
        self.chunk_ids = list(chunk_ids)

    def search(self, query_vec: np.ndarray, k: int = 10, ef_search: int = 64,
               allowed_ids: set[str] | None = None) -> list[SearchResult]:
        """ef_search est regle AVANT chaque requete : c'est le curseur
        precision/latence, et c'est lui qu'on fait varier dans la courbe.

        allowed_ids : filtrage par metadonnees (source, langue...). On elargit
        la recherche puis on filtre — c'est la limite de FAISS, qui ne sait pas
        filtrer nativement. pgvector le ferait dans la clause SQL.
        """
        self.index.hnsw.efSearch = ef_search
        q = np.atleast_2d(np.asarray(query_vec, dtype=np.float32))
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

        # Sur-recuperation quand on filtre : sinon le top-k peut etre
        # entierement compose de resultats exclus, et il ne reste rien.
        k_eff = k if allowed_ids is None else min(k * 10, len(self.chunk_ids))
        scores, idx = self.index.search(q, k_eff)

        out, rang = [], 0
        for i, s in zip(idx[0], scores[0]):
            if i < 0:
                continue
            cid = self.chunk_ids[i]
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            out.append(SearchResult(cid, float(s), rang))
            rang += 1
            if len(out) >= k:
                break
        return out

    def save(self, path: Path) -> None:
        import faiss
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "hnsw.faiss"))
        (path / "ids.pkl").write_bytes(pickle.dumps(self.chunk_ids))

    @classmethod
    def load(cls, path: Path) -> "FaissHNSWStore":
        import faiss
        idx = faiss.read_index(str(path / "hnsw.faiss"))
        store = cls.__new__(cls)
        store.index = idx
        store.dim = idx.d
        store.m = 0
        store.chunk_ids = pickle.loads((path / "ids.pkl").read_bytes())
        return store

    def __len__(self):
        return len(self.chunk_ids)


def recall_loss_curve(brute: BruteForceStore, ann: FaissHNSWStore,
                      queries: np.ndarray, ef_values=(16, 32, 64, 128, 256),
                      k: int = 10) -> list[dict]:
    """Combien l'index approximatif fait-il perdre, et pour quel gain de vitesse ?

    Pour chaque ef_search :
      - memes requetes sur l'index exact et sur HNSW
      - recall@k = proportion des VRAIS top-k effectivement retrouves
      - latence mediane (p50) et 95e centile (p95)

    Le p95 compte autant que le p50 : une requete sur vingt est bien plus
    lente que la mediane, et c'est celle-la que l'utilisateur remarque.

    Le "coude" de la courbe recall/latence est le reglage a retenir. Il depend
    de VOS donnees : c'est pourquoi la valeur par defaut est rarement la bonne.
    """
    verite = []
    for q in queries:
        t = time.perf_counter()
        verite.append(({r.chunk_id for r in brute.search(q, k=k)}, (time.perf_counter() - t) * 1000))

    exact_ms = float(np.median([t for _, t in verite]))
    lignes = []
    for ef in ef_values:
        recalls, temps = [], []
        for q, (vrais, _) in zip(queries, verite):
            t = time.perf_counter()
            trouves = {r.chunk_id for r in ann.search(q, k=k, ef_search=ef)}
            temps.append((time.perf_counter() - t) * 1000)
            recalls.append(len(vrais & trouves) / max(len(vrais), 1))
        lignes.append({
            "ef_search": ef,
            f"recall@{k}": float(np.mean(recalls)),
            "latence_p50_ms": float(np.percentile(temps, 50)),
            "latence_p95_ms": float(np.percentile(temps, 95)),
            "acceleration_vs_exact": exact_ms / max(float(np.median(temps)), 1e-6),
        })
    return lignes
