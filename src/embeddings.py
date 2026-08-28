"""
ARC 2 — La couche d'embedding.

Concept a acquerir :
  Un embedding coute du temps et parfois de l'argent. Re-embedder deux
  fois le meme chunk est la faute n.1 des projets RAG amateurs. Cette
  couche rend tous les arcs suivants rapides : sans cache, chaque
  experience d'Arc 6 relance des heures de calcul.

Piege classique (documente dans le roadmap) :
  La plupart des modeles attendent un PREFIXE different pour les requetes
  ("query: ...") et pour les documents ("passage: ..."). Les melanger fait
  chuter le recall silencieusement — aucune erreur, juste de mauvais
  resultats. Voir EMBEDDING_MODELS dans config.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import CACHE_DIR, EMBEDDING_MODELS, DEFAULT_EMBEDDING


class EmbeddingCache:
    """Cache disque adresse par le hash du contenu.

    Implemente pour vous : c'est de la plomberie, pas de l'apprentissage.
    Cle = sha256(nom_du_modele + prefixe + texte) -> jamais de collision
    entre deux modeles.
    """

    def __init__(self, model_key: str):
        self.dir = CACHE_DIR / model_key
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "index.json"
        self._index: dict[str, int] = (
            json.loads(self._index_path.read_text()) if self._index_path.exists() else {}
        )
        self._vecs_path = self.dir / "vectors.npy"
        self._vecs: np.ndarray | None = (
            np.load(self._vecs_path) if self._vecs_path.exists() else None
        )

    @staticmethod
    def key(text: str, prefix: str = "") -> str:
        return hashlib.sha256((prefix + text).encode("utf-8")).hexdigest()

    def get(self, text: str, prefix: str = "") -> np.ndarray | None:
        k = self.key(text, prefix)
        if k in self._index and self._vecs is not None:
            return self._vecs[self._index[k]]
        return None

    def put_many(self, texts: list[str], vecs: np.ndarray, prefix: str = "") -> None:
        start = 0 if self._vecs is None else self._vecs.shape[0]
        self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])
        for i, t in enumerate(texts):
            self._index[self.key(t, prefix)] = start + i

    def flush(self) -> None:
        if self._vecs is not None:
            np.save(self._vecs_path, self._vecs)
        self._index_path.write_text(json.dumps(self._index))

    def __len__(self) -> int:
        return len(self._index)


class Embedder:
    """Interface unique pour plusieurs modeles (Arc 2, checklist 1).

    Le but : pouvoir ecrire `Embedder("e5-base")` puis `Embedder("bge-m3")`
    et relancer tout le benchmark sans changer une ligne ailleurs.
    """

    def __init__(self, model_key: str = DEFAULT_EMBEDDING, use_cache: bool = True):
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Modele inconnu : {model_key}. Choix : {list(EMBEDDING_MODELS)}")
        self.key = model_key
        self.cfg = EMBEDDING_MODELS[model_key]
        self.cache = EmbeddingCache(model_key) if use_cache else None
        self._model = None  # chargement paresseux : ne charge le modele que si besoin

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg["name"])
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._embed(texts, self.cfg["doc_prefix"], batch_size)

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._embed(texts, self.cfg["query_prefix"], batch_size)

    def _embed(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        missing_idx, missing_txt = [], []

        if self.cache:
            for i, t in enumerate(texts):
                hit = self.cache.get(t, prefix)
                if hit is not None:
                    out[i] = hit
                else:
                    missing_idx.append(i)
                    missing_txt.append(t)
        else:
            missing_idx, missing_txt = list(range(len(texts))), list(texts)

        if missing_txt:
            model = self._load()
            vecs = model.encode(
                [prefix + t for t in missing_txt],
                batch_size=batch_size,
                show_progress_bar=len(missing_txt) > 100,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            if self.cache:
                self.cache.put_many(missing_txt, vecs, prefix)
                self.cache.flush()
            for j, i in enumerate(missing_idx):
                out[i] = vecs[j]

        return np.vstack(out)


def benchmark_models(texts: list[str], queries: list[str], model_keys: list[str]) -> "pd.DataFrame":
    """
    TODO (J6) — Arc 2, checklist 4.

    Comparer les modeles sur : qualite de retrieval, latence, taille, et
    ROBUSTESSE MULTILINGUE (le boss fight de l'arc).

    Boss fight "Multilingual Gauntlet", adapte a votre profil :
      Prenez 20 concepts de votre corpus, ecrits en francais ET en arabe.
      Interrogez en francais, cherchez a retrouver le passage arabe
      correspondant (et inversement). Quel modele tient reellement ?
      C'est un resultat que TRES peu de gens ont mesure, et il est
      directement valorisable sur le marche marocain.

    Retourne un DataFrame : une ligne par modele, prêt pour reports/.
    """
    raise NotImplementedError("Arc 2 — boss fight")
