"""
Le pipeline complet — assemble tous les arcs.

    question
      -> embed (Arc 2)
      -> recherche vectorielle (Arc 3) + BM25 (Arc 4)
      -> fusion RRF (Arc 4)
      -> reranking cross-encoder (Arc 5)
      -> generation citee (Arc 7)
      -> verification des citations (Arc 7)

Ce fichier est volontairement une COQUILLE : chaque etage appelle un
module que vous aurez ecrit. C'est ici qu'on voit que RAG = assemblage
de briques comprises, pas une boite noire.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    citations_valid: bool = True
    refused: bool = False
    latency_ms: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, embedder, vector_store, bm25, reranker, llm,
                 top_k_retrieve: int = 30, top_k_rerank: int = 5,
                 use_hybrid: bool = True, use_rerank: bool = True):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.llm = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank

    def retrieve(self, question: str) -> list[dict]:
        """
        TODO (J12 puis J13) — assembler vectoriel + BM25 + RRF.

        Les drapeaux use_hybrid / use_rerank existent pour une raison
        precise : en Arc 6 vous devrez comparer "vectoriel seul",
        "BM25 seul", "hybride", "hybride + rerank". Sans ces interrupteurs,
        vous ne pourrez pas produire le tableau maitre.
        """
        raise NotImplementedError("Arc 4/5")

    def answer(self, question: str) -> RAGResponse:
        """TODO (J17) — retrieve -> build_context -> LLM -> verify_citations.
        Mesurer la latence de CHAQUE etage separement : c'est ce qui vous
        permettra de dire ou passe le temps (Arc 8)."""
        raise NotImplementedError("Arc 7")
