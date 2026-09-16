"""
Ingestion MULTI-SOURCE : PDF / HTML -> texte propre + metadonnees.

Sortie : un JSON par document dans data/corpus/, avec pour chaque page
  - `text`      : texte nettoye, destine a l'AFFICHAGE et aux citations
  - `text_norm` : texte normalise, destine a la RECHERCHE (embeddings + BM25)

Cette separation est structurante : un texte de loi cite dans une reponse doit
etre exact au caractere pres, alors que la recherche a besoin d'une forme
canonique (voir src/arabic.py).

Usage :
    python scripts/00_ingest.py                  # tout ce qui est present
    python scripts/00_ingest.py --priority 1     # sources prioritaires
    python scripts/00_ingest.py --source code_travail
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arabic import arabic_ratio, normalize_for_search
from src.config import CORPUS_DIR, DATA, SOURCES

RAW_DIR = DATA / "raw"
RAW_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# Extraction par format
# ---------------------------------------------------------------

def pdf_to_pages(path: Path) -> list[dict]:
    doc = pymupdf.open(path)
    return [{"page": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]


def html_to_pages(path: Path) -> list[dict]:
    """Circulaires DGI publiees en HTML.

    On retire nav/footer/script mais on CONSERVE les <table> : dans une
    circulaire fiscale, le tableau de taux est precisement l'information utile.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    # Les tableaux sont aplatis en lignes " | " pour rester lisibles apres chunking
    for table in soup.find_all("table"):
        lignes = [" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"]))
                  for tr in table.find_all("tr")]
        table.replace_with("\n".join(lignes))
    return [{"page": 1, "text": soup.get_text("\n")}]


EXTRACTORS = {".pdf": pdf_to_pages, ".html": html_to_pages, ".htm": html_to_pages}


# ---------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------

# En-tetes / pieds de page repetes sur chaque page : ils polluent le texte et,
# pire, ils creent de faux voisins semantiques (toutes les pages se ressemblent).
BRUIT = [
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.M),            # numero de page isole
    re.compile(r"^\s*BULLETIN OFFICIEL.*$", re.M | re.I),
    re.compile(r"^\s*Bulletin Officiel n[°o]\s*\d+.*$", re.M | re.I),
    re.compile(r"^\s*CODE DU TRAVAIL\s*$", re.M | re.I),
]

# Mot coupe par un tiret en fin de ligne : "travail-\nleur" -> "travailleur"
COUPURE = re.compile(r"(\w)-\s*\n\s*(\w)")


def clean(text: str) -> str:
    """Nettoyage destine a l'affichage. Ne modifie AUCUNE lettre."""
    for motif in BRUIT:
        text = motif.sub("", text)
    text = COUPURE.sub(r"\1\2", text)
    text = text.replace(" ", " ").replace(" ", " ")   # espaces insecables
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_lang(text: str) -> str:
    """fr / ar selon la proportion de lettres arabes. Deux langues aussi
    distinctes ne justifient pas une librairie de detection."""
    return "ar" if arabic_ratio(text) > 0.4 else "fr"


def detect_source(path: Path) -> str:
    """L'identifiant de source vient du NOM DU DOSSIER parent.
    Choix volontaire : la metadonnee devient impossible a oublier."""
    for parent in path.parents:
        if parent.name in SOURCES:
            return parent.name
    raise ValueError(f"source inconnue pour {path} — le dossier doit s'appeler comme une cle de SOURCES")


# ---------------------------------------------------------------
# Reperage des unites atomiques (sert au chunker structurel)
# ---------------------------------------------------------------

UNIT_PATTERNS = {
    # FR : "Article 145" / "Article premier"
    # ------------------------------------------------------------------
    # CORRECTIF (golden dataset v1). L'ancien motif etait insensible a la
    # casse et non ancre : « l'article 43 ci-dessus » etait pris pour le
    # DEBUT de l'article 43. Le chunker creait alors des chunks-debris qui
    # volaient les premieres places (28 % des chunks FR etiquetes article).
    #
    # Nouveau motif : « Article » avec majuscule, en DEBUT DE LIGNE, et non
    # suivi d'une formule de renvoi.
    # Mesure sur les deux editions francaises :
    #     ancien  : 844 reperages, 595 numeros, 249 doublons
    #     nouveau : 589 reperages, 589 numeros,   0 doublon
    # Le Code du travail compte exactement 589 articles.
    # ------------------------------------------------------------------
    "article_fr": re.compile(
        r"^[ \t]*Article[ \t]+(\d+|premier)\b"
        r"(?![ \t]*(ci-|ci ?dessus|ci ?dessous|du |de la |de l'|des ))", re.M),
    # AR : "المادة 145" et la variante ou le numero precede le mot (ordre RTL
    #      inverse a l'extraction — diagnostic fait avant l'ingestion)
    # Le motif tolere les deux ta marbuta (ة / ه), les chiffres latins ET
    # arabo-indiens, et les deux ordres : il s'applique au texte BRUT, pas
    # au texte normalise (les offsets doivent pointer dans `text`).
    #
    # Cote arabe, meme ancrage en debut de ligne. Compromis ASSUME :
    #     version ministere : 451 numeros / 142 doublons -> 396 / 5
    #     version justice   : 118 numeros /  35 doublons ->  27 / 1
    # On perd de la couverture (surtout sur la version justice, dont
    # l'extraction RTL est tres degradee) pour gagner en precision. Raison :
    # un FAUX en-tete cree un debris qui vole la 1re place ; un en-tete MANQUE
    # fait seulement retomber la page sur un decoupage par phrases.
    # Variante testee et rejetee : exclusion par la preposition precedente
    # (« في المادة 43 ») -> 412 numeros mais 26 doublons.
    "article_ar": re.compile(
        r"^[ \t]*(?:الماد[ةه][ \t]*[:\-]?[ \t]*([\d٠-٩]+)|([\d٠-٩]+)[ \t]*الماد[ةه])"
        r"(?![ \t]*(أعلاه|أدناه|من الظهير|من القانون|من المرسوم|من هذا|من هذه))", re.M),
    "rubrique": re.compile(r"\bRubrique\s+([\d.]+)", re.I),
    "circulaire": re.compile(r"\bCirculaire\s+n[°o]?\s*([\d/\-]+)", re.I),
}


def extract_unit_refs(text: str, unit: str, lang: str) -> list[dict]:
    """Positions des unites atomiques dans le texte (offset + reference).

    C'est ce reperage qui permet au chunker structurel de ne JAMAIS couper
    un article en deux. Sans lui, l'article 145 se retrouve eclate sur deux
    chunks et devient irrecuperable.
    """
    cles = {
        "article": ["article_ar"] if lang == "ar" else ["article_fr"],
        "rubrique": ["rubrique"],
        "section": ["circulaire", "article_fr"],
    }.get(unit, ["article_fr"])

    refs = []
    for cle in cles:
        for m in UNIT_PATTERNS[cle].finditer(text):
            num = next((g for g in m.groups() if g), None)
            if num:
                refs.append({"ref": f"{unit} {num}", "num": str(num), "start": m.start()})
    return sorted(refs, key=lambda r: r["start"])


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------

def ingest_file(path: Path, source_id: str) -> dict:
    cfg = SOURCES[source_id]
    pages_brutes = EXTRACTORS[path.suffix.lower()](path)

    pages, refs_total = [], 0
    for p in pages_brutes:
        txt = clean(p["text"])
        if not txt:
            continue
        # La table des matieres cite chaque article avec des points de suite.
        # Indexee, elle repond a TOUTES les questions « que dit l'article N ? »
        # sans jamais contenir la reponse. Observe dans le golden dataset :
        # elle occupait la 1re place BM25 sur « Contenu de l'article 40 ».
        if "TABLE DES MATI" in txt.upper() or len(re.findall(r"\.{5,}", txt)) >= 5:
            continue
        lang = detect_lang(txt)
        refs = extract_unit_refs(txt, cfg["unit"], lang)
        refs_total += len(refs)
        pages.append({
            "page": p["page"],
            "lang": lang,
            "text": txt,                              # affichage / citation
            "text_norm": normalize_for_search(txt),   # recherche
            "unit_refs": refs,
        })

    langs = {p["lang"] for p in pages}
    return {
        "doc_id": f"{source_id}::{path.stem}",
        "source_id": source_id,
        "source_label": cfg["label"],
        "authority": cfg["authority"],
        "unit": cfg["unit"],
        "file": path.name,
        "langs": sorted(langs),
        "n_pages": len(pages),
        "n_unit_refs": refs_total,
        "pages": pages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES))
    ap.add_argument("--priority", type=int, help="1 = sources prioritaires, 2 = ajouts")
    args = ap.parse_args()

    targets = [args.source] if args.source else [
        sid for sid, c in SOURCES.items()
        if args.priority is None or c["priority"] == args.priority
    ]

    total = 0
    for sid in targets:
        src_dir = RAW_DIR / sid
        print(f"[{sid}] {SOURCES[sid]['label']}")
        if not src_dir.exists():
            print(f"  (dossier absent : {src_dir})")
            continue
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() not in EXTRACTORS:
                continue
            doc = ingest_file(f, sid)
            dest = CORPUS_DIR / f"{sid}__{f.stem}.json"
            dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  OK  {f.name:<40} {doc['n_pages']:>4} pages | "
                  f"langues {','.join(doc['langs'])} | {doc['n_unit_refs']:>4} references d'unite")
            total += 1

    print(f"\n{total} document(s) -> {CORPUS_DIR}" if total else "\nAucun document trouve.")


if __name__ == "__main__":
    main()
