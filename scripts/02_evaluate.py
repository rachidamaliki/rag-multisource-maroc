"""
J8 puis J19 — Le harnais d'evaluation. Le coeur du projet.

Usage :
  python scripts/02_evaluate.py --config all --output reports/master_table.csv

C'est ce script qui produit LE TABLEAU MAITRE : une ligne par combinaison
(chunking x modele x retrieval x reranking), colonnes = recall@5, MRR,
NDCG@10, faithfulness, latence p50/p95, cout.

>>> Cet unique tableau vaut plus, en entretien, que n'importe quel
    tutoriel de framework termine. <<<
"""
import argparse
import itertools

CHUNKERS = ["fixed", "sentence", "semantic", "structural"]
RETRIEVALS = ["vector", "bm25", "hybrid"]
RERANKING = [False, True]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all")
    ap.add_argument("--output", default="reports/master_table.csv")
    args = ap.parse_args()

    # TODO (J8, minimal) : evaluer UNE config sur le golden dataset
    # TODO (J19, complet) : boucler sur toutes les combinaisons
    for chunker, retrieval, rerank in itertools.product(CHUNKERS, RETRIEVALS, RERANKING):
        if retrieval == "bm25" and rerank:
            pass  # combinaison valable, a inclure aussi
        # -> charger l'index, lancer les requetes du golden, calculer les metriques
    raise NotImplementedError("Arc 6")


if __name__ == "__main__":
    main()
