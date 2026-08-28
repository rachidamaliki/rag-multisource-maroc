"""
J10 / J14 / J19 — Generer les 3 figures de portfolio.

  1. arc3_recall_vs_latency.png     recall@10 vs latence (ef_search variable)
  2. arc4_hybrid_comparison.png     vectoriel / BM25 / hybride, par type de requete
  3. arc5_precision_vs_latency.png  precision@5 vs latence (taille du pool reranke)

Conseils de lisibilite (ces images seront vues par des recruteurs) :
  - titre explicite, axes nommes AVEC les unites
  - annoter le point choisi ("configuration retenue")
  - une seule idee par figure
"""
import matplotlib.pyplot as plt
from src.config import FIGURES


def figure_recall_vs_latency(results):
    """TODO (J10)"""
    raise NotImplementedError("Arc 3")


def figure_hybrid_comparison(results):
    """TODO (J12)"""
    raise NotImplementedError("Arc 4")


def figure_precision_vs_latency(results):
    """TODO (J14)"""
    raise NotImplementedError("Arc 5")


if __name__ == "__main__":
    print(f"Figures sauvegardees dans : {FIGURES}")
