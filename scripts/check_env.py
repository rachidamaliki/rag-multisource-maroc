"""
Verification de l'environnement — a lancer apres l'installation.

    python scripts/check_env.py

Verifie les imports lourds, la config, la cle LLM et l'arborescence.
Ne telecharge aucun modele.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, KO = "[OK]  ", "[KO]  "
problems = []


def check(label, fn):
    try:
        detail = fn()
        print(f"{OK}{label}{f' — {detail}' if detail else ''}")
    except Exception as e:
        print(f"{KO}{label} — {type(e).__name__}: {e}")
        problems.append(label)


print("=== Environnement ===")
check("python 3.12", lambda: sys.version.split()[0])
check("numpy", lambda: __import__("numpy").__version__)
check("torch", lambda: __import__("torch").__version__)
check("faiss", lambda: __import__("faiss").__version__ if hasattr(__import__("faiss"), "__version__") else "installe")
check("sentence-transformers", lambda: __import__("sentence_transformers").__version__)
check("tiktoken", lambda: str(len(__import__("tiktoken").get_encoding("cl100k_base").encode("test"))) + " tokens sur 'test'")
check("pymupdf", lambda: __import__("fitz").__doc__.strip().splitlines()[0] if __import__("fitz").__doc__ else "installe")
check("fastapi + streamlit", lambda: __import__("fastapi").__version__)

print("\n=== Projet ===")
check("src.config", lambda: f"{len(__import__('src.config', fromlist=['SOURCES']).SOURCES)} sources declarees")


def check_env_file():
    from src.config import settings, ROOT
    if not (ROOT / ".env").exists():
        raise FileNotFoundError(".env absent — faire : copy .env.example .env")
    key = settings.groq_api_key or settings.google_api_key
    if not key:
        raise ValueError("aucune cle LLM renseignee dans .env")
    return f"{settings.llm_provider} configure"


check(".env + cle LLM", check_env_file)


def check_dirs():
    from src.config import CORPUS_DIR, CACHE_DIR, GOLDEN_DIR, DATA
    raw = DATA / "raw"
    n = sum(1 for d in raw.iterdir() if d.is_dir()) if raw.exists() else 0
    return f"{n} dossiers sources dans data/raw/"


check("arborescence data/", check_dirs)


def check_corpus():
    from src.config import CORPUS_DIR, DATA
    pdfs = list((DATA / "raw").rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("aucun PDF dans data/raw/ — deposer le Code du travail")
    return f"{len(pdfs)} document(s) source(s)"


check("corpus present", check_corpus)

print()
if problems:
    print(f"{len(problems)} point(s) a regler : " + ", ".join(problems))
    print("(normal avant le J3 : la cle LLM et le corpus)")
else:
    print("Tout est pret. Vous pouvez demarrer le J1.")
