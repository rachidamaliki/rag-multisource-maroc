"""
Evaluation de la RECHERCHE sur le golden dataset.

    python scripts/05_evaluate_golden.py

Compare plusieurs configurations sur le MEME examen, ventile par type de
question, et ecrit reports/golden_retrieval.csv.

Les questions `unanswerable` ne sont pas notees ici : il n'existe pas de
« bon passage » a retrouver. Elles serviront a mesurer le refus, au moment
de la generation.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GOLDEN_DIR, INDEX_DIR, REPORTS
from src.embeddings import Embedder
from src.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.metrics import evaluate_retrieval
from src.sources import balanced_merge
from src.vectorstore import FaissHNSWStore

IDX = INDEX_DIR / "structural_bge-m3"
K = (1, 3, 5, 10)
VIVIER = 20


def charger():
    chunks = {json.loads(l)["id"]: json.loads(l) for l in (IDX / "chunks.jsonl").open(encoding="utf-8")}
    return chunks, FaissHNSWStore.load(IDX), pickle.loads((IDX / "bm25.pkl").read_bytes())


def quota_langue(liste, chunks, top_n=10, min_par_langue=2):
    groupes = {}
    for r in liste:
        groupes.setdefault(chunks[r.chunk_id]["lang"], []).append(r)
    return balanced_merge(groupes, total_k=top_n, min_per_group=min_par_langue)


def main():
    chunks, store, bm25 = charger()
    emb = Embedder("bge-m3")
    golden = [json.loads(l) for l in (GOLDEN_DIR / "golden.jsonl").open(encoding="utf-8")]
    evaluables = [g for g in golden if g["answerable"]]

    qvs = emb.embed_queries([g["question"] for g in evaluables])

    configs = ["vectoriel", "bm25", "rrf", "ponderee", "ponderee+quota"]
    runs = {c: {} for c in configs}
    for g, qv in zip(evaluables, qvs):
        lv = store.search(qv, k=VIVIER, ef_search=128)
        lb = bm25.search(g["question"], k=VIVIER)
        pond = weighted_score_fusion([lv, lb], poids=[0.5, 1.0], top_n=VIVIER)
        runs["vectoriel"][g["qid"]] = [r.chunk_id for r in lv[:10]]
        runs["bm25"][g["qid"]] = [r.chunk_id for r in lb[:10]]
        runs["rrf"][g["qid"]] = [r.chunk_id for r in reciprocal_rank_fusion([lv, lb], top_n=10)]
        runs["ponderee"][g["qid"]] = [r.chunk_id for r in pond[:10]]
        runs["ponderee+quota"][g["qid"]] = [r.chunk_id for r in quota_langue(pond, chunks)]

    types = ["TOUS"] + sorted({g["type"] for g in evaluables})
    lignes = []
    for t in types:
        sel = [g for g in evaluables if t == "TOUS" or g["type"] == t]
        gold = {g["qid"]: set(g["relevant_chunk_ids"]) for g in sel}
        for c in configs:
            m = evaluate_retrieval({q: runs[c][q] for q in gold}, gold, k_values=K)
            lignes.append({"type": t, "configuration": c, **m})

    REPORTS.mkdir(exist_ok=True)
    cles = ["type", "configuration", "n_requetes", "pertinents_par_requete",
            "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "recall@10", "diagnostic"]
    with (REPORTS / "golden_retrieval.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cles, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)

    for t in types:
        n = next(l["n_requetes"] for l in lignes if l["type"] == t)
        print(f"\n=== {t} ({n} questions) ===")
        print(f"{'configuration':<16} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>7} {'MRR':>6}   diagnostic")
        for l in (l for l in lignes if l["type"] == t):
            print(f"{l['configuration']:<16} {l['hit@1']:>6.2f} {l['hit@3']:>6.2f} {l['hit@5']:>6.2f} "
                  f"{l['hit@10']:>7.2f} {l['mrr']:>6.2f}   {l['diagnostic']}")

    # pour inspection manuelle : ce que la meilleure config rate
    (REPORTS / "golden_runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
