"""
ARC 5 — Le reranking par cross-encoder.

Concept a acquerir (question d'entretien quasi garantie) :

  BI-ENCODER (votre modele d'embedding) :
    encode la question et le document SEPAREMENT, puis compare les vecteurs.
    -> les documents sont encodes UNE FOIS a l'avance : cout O(1) par requete.
    -> mais le modele n'a jamais "vu" la question et le document ensemble.

  CROSS-ENCODER (le reranker) :
    lit la question ET le document EN MEME TEMPS, et sort un score de
    pertinence. Beaucoup plus precis.
    -> mais il faut un passage de modele PAR PAIRE : cout O(n) par requete.
    -> impossible sur 100 000 documents, parfait sur 30.

  D'ou l'architecture en deux etages : le bi-encodeur ratisse large et vite
  (top-30), le cross-encodeur affine avec precision (top-5). On ne peut pas
  utiliser le cross-encodeur partout, et c'est toute la reponse a la question
  "pourquoi ne pas juste utiliser le meilleur modele ?".
"""
from __future__ import annotations
from src.config import RERANKER_MODEL


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int = 5):
        """
        TODO (J13).
        candidates : [(chunk_id, texte), ...] issus de la recherche hybride
        retourne   : les top_k reordonnes, avec leur score de pertinence

        Piege documente : ne JAMAIS reranker tout le corpus. Passer un
        cross-encodeur sur 500 candidats annule tout le benefice d'avoir
        une recherche rapide.
        """
        raise NotImplementedError("Arc 5")


def precision_latency_curve(pipeline, golden, candidate_pool_sizes=(10, 30, 100)):
    """
    TODO (J14) — BOSS FIGHT Arc 5.

    Tracer precision@5 en fonction de la latence bout-en-bout, en faisant
    varier le nombre de candidats reranked.
    Puis ECRIRE UN PARAGRAPHE justifiant votre choix comme si vous le
    defendiez devant un chef de produit : "je prends top-30 parce que le
    passage a top-100 coute +180 ms pour +2 % de precision — pas rentable
    pour un assistant interactif."

    Cette capacite a arbitrer avec des chiffres est precisement ce qui
    separe un profil junior d'un profil qu'on embauche.

    Sauvegarder : reports/figures/arc5_precision_vs_latency.png
    """
    raise NotImplementedError("Arc 5 — boss fight")
