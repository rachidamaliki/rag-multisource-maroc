"""
ARC 8 — L'API de production.

Concept a acquerir :
  Un notebook qui marche une fois n'est pas un systeme. Ce qui distingue
  un prototype d'un service : le cache, le suivi du cout, le monitoring,
  et surtout la DEGRADATION GRACIEUSE.

Piege documente : que se passe-t-il si le vector store ne repond plus,
ou si l'API du reranker tombe ? Un systeme de production repond quand
meme (en mode degrade, en le signalant), il ne renvoie pas une erreur 500.
"""
from __future__ import annotations
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG From Zero", version="0.1.0")

_pipeline = None          # initialise au demarrage
_response_cache: dict = {}
_query_log: list[dict] = []


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_rerank: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    refused: bool
    citations_valid: bool
    latency_ms: dict
    cached: bool = False


@app.on_event("startup")
def startup():
    """TODO (J20) — charger l'index, les modeles et construire le pipeline UNE fois.
    Charger un modele a chaque requete est l'erreur classique."""
    global _pipeline
    _pipeline = None


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    TODO (J20) — l'endpoint complet :
      embed -> recherche hybride -> rerank -> generer -> citer

    TODO (J21) — y ajouter :
      - cache de reponses (cle = hash de la question normalisee)
      - log du cout et de la latence par etage dans _query_log
      - try/except avec repli : si le reranker echoue, renvoyer quand meme
        les resultats hybrides en indiquant "mode degrade"
    """
    raise NotImplementedError("Arc 8")


@app.get("/stats")
def stats():
    """TODO (J21) — nombre de requetes, latence p50/p95, taux de cache,
    cout cumule en tokens. C'est votre monitoring minimal."""
    raise NotImplementedError("Arc 8")


@app.get("/drift")
def drift():
    """
    TODO (J21) — moniteur de derive.
    Comparer la distribution des questions reelles a celle du golden
    dataset (Arc 6). Si les utilisateurs posent des questions tres
    differentes de celles sur lesquelles vous avez optimise, vos
    metriques ne veulent plus rien dire.
    Version simple et suffisante : similarite cosinus moyenne entre les
    embeddings des requetes recentes et ceux du golden set.
    """
    raise NotImplementedError("Arc 8")
