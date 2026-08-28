"""
J9 — Construire un index complet pour une configuration donnee.

Usage :
  python scripts/01_build_index.py --chunker structural --model bge-m3

Pourquoi un script parametre plutot qu'un notebook ?
Parce qu'en Arc 6 vous devrez reconstruire l'index pour CHAQUE combinaison
du tableau maitre. Un notebook rend ca ingerable ; un script en ligne de
commande le rend reproductible et scriptable dans une boucle.
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="structural", choices=["fixed", "sentence", "semantic", "structural"])
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--store", default="faiss", choices=["faiss", "pgvector", "brute"])
    args = ap.parse_args()

    # TODO (J9) :
    #  1. charger le corpus depuis data/corpus/*.json
    #  2. decouper avec CHUNKERS[args.chunker]
    #  3. embedder (le cache evite de tout recalculer a chaque essai)
    #  4. construire l'index + l'index BM25
    #  5. sauvegarder dans data/index/{chunker}_{model}/
    raise NotImplementedError("Arc 3")


if __name__ == "__main__":
    main()
