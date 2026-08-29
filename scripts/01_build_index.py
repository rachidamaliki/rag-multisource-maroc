"""
Construction d'un index complet pour UNE configuration.

    python scripts/01_build_index.py --chunker structural --model bge-m3

Pourquoi un script parametre plutot qu'un notebook : au moment de l'evaluation,
il faudra reconstruire l'index pour CHAQUE combinaison du tableau comparatif.
Un notebook rend ca ingerable ; une commande se met dans une boucle.

Produit dans data/index/<chunker>_<model>/ :
    chunks.jsonl   les chunks avec leurs metadonnees
    vectors.npy    les embeddings (normalises)
    hnsw.faiss     l'index approximatif
    bm25.pkl       l'index lexical
    meta.json      la configuration et les statistiques
"""
import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import CHUNKERS, chunk_stats
from src.config import CORPUS_DIR, INDEX_DIR, EMBEDDING_MODELS, HNSW
from src.embeddings import Embedder
from src.lexical import BM25, tokenize
from src.vectorstore import FaissHNSWStore


def charger_corpus(sources: list[str] | None = None) -> list[dict]:
    docs = []
    for f in sorted(CORPUS_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if sources and d["source_id"] not in sources:
            continue
        docs.append(d)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="structural", choices=list(CHUNKERS))
    ap.add_argument("--model", default="bge-m3", choices=list(EMBEDDING_MODELS))
    ap.add_argument("--source", action="append", help="limiter a certaines sources")
    ap.add_argument("--lang", help="limiter a une langue (fr / ar)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = charger_corpus(args.source)
    if not docs:
        print("Aucun document. Lancer d'abord scripts/00_ingest.py")
        return
    print(f"corpus : {len(docs)} document(s), {sum(d['n_pages'] for d in docs)} pages")

    # --- 1. decoupage ---
    fn = CHUNKERS[args.chunker]
    embedder = Embedder(args.model)
    chunks = []
    for d in docs:
        if args.chunker == "semantic":
            chunks += fn(d, embed_fn=embedder.embed_documents)
        else:
            chunks += fn(d)
    if args.lang:
        chunks = [c for c in chunks if c.lang == args.lang]
    st = chunk_stats(chunks)
    print(f"decoupage '{args.chunker}' : {st['n_chunks']} chunks, "
          f"{st['tokens_moy']:.0f} tokens moy., {st['avec_reference']} avec reference")

    # --- 2. embeddings (le cache evite de tout recalculer a chaque essai) ---
    t = time.perf_counter()
    vecs = embedder.embed_documents([c.text_norm for c in chunks])
    print(f"embeddings : {vecs.shape} en {time.perf_counter()-t:.1f}s")

    # --- 3. index vectoriel ---
    store = FaissHNSWStore(dim=vecs.shape[1], m=HNSW["m"], ef_construction=HNSW["ef_construction"])
    store.build(vecs, [c.id for c in chunks])

    # --- 4. index lexical ---
    # BM25 travaille sur text_norm, comme les embeddings : meme normalisation
    # pour les deux voies, sinon elles ne parlent pas de la meme chose.
    toks = [tokenize(c.text_norm) for c in chunks]
    bm25 = BM25(toks, chunk_ids=[c.id for c in chunks])
    print(f"BM25 : {len(bm25.inverse)} termes distincts, {bm25.avgdl:.0f} tokens/chunk en moyenne")

    # --- 5. sauvegarde ---
    nom = f"{args.chunker}_{args.model}" + (f"_{args.lang}" if args.lang else "")
    out = INDEX_DIR / nom
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    np.save(out / "vectors.npy", vecs)
    store.save(out)
    (out / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (out / "meta.json").write_text(json.dumps({
        "chunker": args.chunker, "model": args.model, "lang": args.lang,
        "sources": sorted({c.source_id for c in chunks}),
        "n_docs": len(docs), "dim": int(vecs.shape[1]),
        "hnsw": HNSW, "chunk_stats": st,
        "n_termes_bm25": len(bm25.inverse),
        "duree_totale_s": round(time.perf_counter() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nindex ecrit dans {out}  ({time.perf_counter()-t0:.1f}s au total)")


if __name__ == "__main__":
    main()
