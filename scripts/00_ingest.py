"""
J3 (puis J9) — Ingestion MULTI-SOURCE : PDF / HTML -> texte propre + metadonnees.

STRATEGIE EN DEUX TEMPS (ne pas s'en ecarter) :
  J3 : UNE seule source, la plus propre (code_travail). Le pipeline doit
       tourner de bout en bout avant d'ajouter quoi que ce soit.
  J9 : ajouter cgnc et dgi_circulaires, une fois que tout fonctionne.

Pourquoi : l'ingestion est la journee la plus risquee du planning. Traiter
3 formats differents le meme jour est la meilleure facon de perdre 3 jours
et de faire echouer le mois. Un pipeline qui marche sur une source se
generalise en quelques heures ; un pipeline qui ne marche sur aucune ne se
generalise jamais.

Organisation attendue des fichiers bruts :

    data/raw/
      code_travail/     *.pdf          <- J3
      cgnc/             *.pdf          <- J9
      dgi_circulaires/  *.pdf, *.html  <- J9

Le NOM DU DOSSIER porte l'identifiant de source. C'est volontaire : la
metadonnee est ainsi impossible a oublier.

Usage :
    python scripts/00_ingest.py                     # tout ce qui est present
    python scripts/00_ingest.py --source code_travail
    python scripts/00_ingest.py --priority 1        # J3 : sources prioritaires seules
"""
import argparse
import json
import re
from pathlib import Path

import fitz  # pymupdf

from src.config import CORPUS_DIR, DATA, SOURCES

RAW_DIR = DATA / "raw"
RAW_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# Extraction par format
# ---------------------------------------------------------------

def pdf_to_pages(path: Path) -> list[dict]:
    doc = fitz.open(path)
    return [{"page": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]


def html_to_pages(path: Path) -> list[dict]:
    """
    TODO (J9) — pour les circulaires DGI publiees en HTML.
    BeautifulSoup, en retirant nav / footer / scripts. Conserver les
    <table> : les tableaux de taux sont exactement l'information utile.
    """
    raise NotImplementedError("multi-source J9")


EXTRACTORS = {".pdf": pdf_to_pages, ".html": html_to_pages, ".htm": html_to_pages}


# ---------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------

def clean(text: str) -> str:
    """
    TODO (J3) — a adapter a CHAQUE source ; elles ne sont jamais sales
    de la meme facon.
      - supprimer en-tetes / pieds de page / numeros de page repetes
      - recoller les mots coupes par un tiret en fin de ligne
      - normaliser les espaces et les retours a la ligne
      - arabe : normaliser alef (ا/أ/إ), ya (ي/ى), retirer les diacritiques
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_lang(text: str) -> str:
    """TODO (J3) — heuristique suffisante : proportion de caracteres
    dans la plage Unicode arabe (\\u0600-\\u06FF). Pas besoin d'une
    librairie de detection de langue pour deux langues aussi distinctes."""
    raise NotImplementedError("multi-source")


def extract_unit_refs(text: str, unit: str) -> list[dict]:
    """
    TODO (J5) — reperer les unites atomiques de la source.

      code_travail     -> "Article 145", "المادة 145"
      cgnc             -> "Rubrique 3.2.1", numeros de comptes
      dgi_circulaires  -> "Circulaire n° 721/2025", sections datees

    C'est ce reperage qui alimente le chunker structurel de l'Arc 1.
    Une source dont on connait l'unite atomique se decoupe parfaitement ;
    une source qu'on decoupe a l'aveugle produit un RAG mediocre.
    """
    raise NotImplementedError("multi-source")


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------

def ingest_source(source_id: str) -> int:
    cfg = SOURCES[source_id]
    src_dir = RAW_DIR / source_id
    if not src_dir.exists():
        print(f"  (ignoree : {src_dir} absent)")
        return 0

    files = [f for f in src_dir.iterdir() if f.suffix.lower() in EXTRACTORS]
    n = 0
    for f in files:
        pages = EXTRACTORS[f.suffix.lower()](f)
        cleaned = [{"page": p["page"], "text": clean(p["text"])} for p in pages]
        out = {
            "doc_id": f"{source_id}::{f.stem}",
            "source_id": source_id,
            "source_label": cfg["label"],
            "authority": cfg["authority"],     # sert a arbitrer les conflits (Arc 7)
            "unit": cfg["unit"],               # sert au chunker structurel (Arc 1)
            "file": f.name,
            "pages": cleaned,
        }
        dest = CORPUS_DIR / f"{source_id}__{f.stem}.json"
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  OK  {f.name} -> {len(pages)} pages")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), help="n'ingerer qu'une source")
    ap.add_argument("--priority", type=int, help="1 = sources du J3, 2 = ajouts du J9")
    args = ap.parse_args()

    targets = [args.source] if args.source else [
        sid for sid, c in SOURCES.items()
        if args.priority is None or c["priority"] == args.priority
    ]

    total = 0
    for sid in targets:
        print(f"[{sid}] {SOURCES[sid]['label']}")
        total += ingest_source(sid)

    if total == 0:
        print(f"\nAucun document trouve. Deposez vos fichiers dans :")
        for sid in targets:
            print(f"  {RAW_DIR / sid}/")
    else:
        print(f"\n{total} document(s) ingere(s) -> {CORPUS_DIR}")


if __name__ == "__main__":
    main()
