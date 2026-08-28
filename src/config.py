"""
Configuration centrale du projet.

Pourquoi un fichier de config ?
Parce qu'en Arc 6 vous allez comparer des dizaines de combinaisons
(chunking x modele x retrieval x reranking). Si les parametres sont
eparpilles dans le code, vos resultats ne seront pas reproductibles
— et un benchmark non reproductible ne vaut rien.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
CACHE_DIR = DATA / "cache"
GOLDEN_DIR = DATA / "golden"
INDEX_DIR = DATA / "index"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (CORPUS_DIR, CACHE_DIR, GOLDEN_DIR, INDEX_DIR, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- Generation ---
    llm_provider: str = "groq"           # groq | gemini | ollama
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    google_api_key: str = ""

    # --- Postgres / pgvector (Arc 3) ---
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "ragzero"
    pg_user: str = "rag"
    pg_password: str = "ragpass"

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password}"
        )


settings = Settings()

# ---------------------------------------------------------------
# Modeles d'embedding testes en Arc 2.
# bge-m3 est le choix par defaut : multilingue (FR + AR), gratuit,
# et il tient la route sur un corpus juridique/comptable marocain.
# ---------------------------------------------------------------
EMBEDDING_MODELS = {
    "bge-m3":       {"name": "BAAI/bge-m3",                          "dim": 1024, "query_prefix": "",        "doc_prefix": ""},
    "e5-base":      {"name": "intfloat/multilingual-e5-base",        "dim": 768,  "query_prefix": "query: ",  "doc_prefix": "passage: "},
    "paraphrase":   {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dim": 384, "query_prefix": "", "doc_prefix": ""},
}
DEFAULT_EMBEDDING = "bge-m3"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ===============================================================
# SOURCES — le corpus est MULTI-SOURCE (decision structurante)
# ===============================================================
# Pourquoi c'est le coeur du projet :
#   Un RAG mono-source est un exercice. Des qu'il y a plusieurs sources,
#   trois vrais problemes d'ingenierie apparaissent :
#     1. FILTRAGE  — "reponds uniquement d'apres le Code du travail"
#        Il faut filtrer DANS la requete vectorielle, pas apres, sinon
#        on ampute le top-k. (Arc 3, checklist 4)
#     2. AUTORITE  — deux sources se contredisent. Laquelle fait foi ?
#        La loi prime sur la doctrine, le recent prime sur l'ancien.
#     3. FORMAT    — PDF structure, tableaux, HTML date. Le chunker
#        structurel (Arc 1) devient un vrai defi.
#
# `authority` : 3 = texte de loi / norme  |  2 = doctrine administrative
#               1 = document explicatif, guide, note interne
# En cas de conflit entre passages, l'autorite la plus elevee gagne,
# puis la date la plus recente. Cette regle est explicite et mesurable
# — c'est ce qui la rend defendable en entretien.

SOURCES = {
    "code_travail": {
        "label": "Code du travail marocain",
        "authority": 3,
        "langs": ["fr", "ar"],
        "format": "pdf",
        "unit": "article",       # unite atomique -> ne JAMAIS couper
        "priority": 1,           # ingere au J3
    },
    "cgnc": {
        "label": "Code General de Normalisation Comptable",
        "authority": 3,
        "langs": ["fr"],
        "format": "pdf",
        "unit": "rubrique",
        "priority": 2,           # ajoute au J9
    },
    "dgi_circulaires": {
        "label": "Circulaires et notes DGI",
        "authority": 2,
        "langs": ["fr", "ar"],
        "format": "pdf+html",
        "unit": "section",
        "priority": 2,           # ajoute au J9
    },
}

# Regle de resolution des conflits (Arc 7).
# A implementer et surtout A MESURER : combien de fois le systeme
# choisit-il la bonne source quand deux passages se contredisent ?
CONFLICT_RULE = ("authority", "date")   # ordre de priorite decroissant

# --- Parametres de decoupage (Arc 1) ---
CHUNK_CONFIGS = {
    "fixed":     {"max_tokens": 400, "overlap": 60},
    "sentence":  {"max_tokens": 400, "overlap": 1},     # overlap = nb de phrases
    "semantic":  {"max_tokens": 600, "similarity_drop": 0.25},
    "structural": {"max_tokens": 800, "overlap": 0},
}

# --- Parametres HNSW (Arc 3) ---
HNSW = {"m": 16, "ef_construction": 200, "ef_search": 64}

# --- Pipeline (Arc 4/5) ---
TOP_K_RETRIEVE = 30      # candidats sortis de la recherche hybride
TOP_K_PER_SOURCE = 10    # multi-source : garantir une representation minimale
                         # de chaque source avant fusion, sinon la source la
                         # plus volumineuse ecrase toutes les autres
TOP_K_RERANK = 5         # ce qu'on envoie finalement au LLM
RRF_K = 60               # constante de la Reciprocal Rank Fusion
