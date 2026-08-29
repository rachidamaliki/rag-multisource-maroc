"""
Demonstration : ou la recherche vectorielle echoue, et pourquoi BM25 existe.

    python scripts/demo_vectoriel_vs_bm25.py

Protocole : on prend N numeros d'articles REELLEMENT presents dans le corpus,
on interroge « article N », et on verifie si l'article demande figure dans le
top-k. La verite terrain est gratuite ici — elle vient de `unit_ref`, la
reference attachee a chaque chunk au moment du decoupage structurel.

C'est le cas ideal pour commencer une evaluation : une question dont on connait
la reponse sans avoir a annoter quoi que ce soit.
"""
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INDEX_DIR
from src.embeddings import Embedder
from src.vectorstore import FaissHNSWStore

IDX = INDEX_DIR / "structural_bge-m3"


def charger():
    chunks = {json.loads(l)["id"]: json.loads(l)
              for l in (IDX / "chunks.jsonl").open(encoding="utf-8")}
    return chunks, FaissHNSWStore.load(IDX), pickle.loads((IDX / "bm25.pkl").read_bytes())


def main(n_requetes: int = 20):
    chunks, store, bm25 = charger()
    emb = Embedder("bge-m3")

    nums = sorted({int(m.group(1)) for c in chunks.values()
                   if c["lang"] == "fr" and (m := re.match(r"article (\d+)$", c["unit_ref"] or ""))})
    echantillon = nums[::max(1, len(nums) // n_requetes)][:n_requetes]

    def trouve(res, cible):
        return any((chunks[r.chunk_id]["unit_ref"] or "") == f"article {cible}" for r in res)

    res = []
    for n in echantillon:
        q = f"article {n}"
        qv = emb.embed_queries([q])[0]
        res.append({
            "num": n,
            "vect_top1": trouve(store.search(qv, k=1, ef_search=128), n),
            "vect_top5": trouve(store.search(qv, k=5, ef_search=128), n),
            "bm25_top1": trouve(bm25.search(q, k=1), n),
            "bm25_top5": trouve(bm25.search(q, k=5), n),
        })

    def pct(cle):
        return 100 * sum(r[cle] for r in res) / len(res)

    print(f"Echantillon : {len(res)} requetes « article N »\n")
    print(f"{'methode':<22} {'top-1':>8} {'top-5':>8}")
    print("-" * 40)
    print(f"{'vectoriel (bge-m3)':<22} {pct('vect_top1'):>7.0f}% {pct('vect_top5'):>7.0f}%")
    print(f"{'BM25':<22} {pct('bm25_top1'):>7.0f}% {pct('bm25_top5'):>7.0f}%")
    return res


if __name__ == "__main__":
    main()
