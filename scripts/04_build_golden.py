"""
Construction et AUDIT du golden dataset.

    python scripts/04_build_golden.py

Entree  : data/golden/golden_source.jsonl  (ecrit a la main)
Sortie  : data/golden/golden.jsonl          (enrichi des chunks pertinents)
          data/golden/audit.md              (rapport lisible)

POURQUOI UN AUDITEUR
--------------------
Un golden dataset est l'examen corrige du systeme. Si le corrige est faux,
toutes les notes sont fausses — et personne ne s'en apercoit.

Personne ici n'est expert du droit du travail marocain. On remplace donc
l'autorite de l'expert par une PREUVE VERIFIABLE : chaque reponse porte un
extrait mot pour mot de l'article, et ce script verifie qu'il y figure.

Quatre controles, dont trois BLOQUANTS :

  1. [bloquant] la preuve existe mot pour mot dans l'article cite
  2. [bloquant] l'article cite a au moins un vrai chunk dans l'index
  3. [bloquant] les sujets des questions "sans reponse" sont reellement
                absents du corpus
  4. [alerte]   une question "semantic" ne doit pas recopier le vocabulaire
                de l'article, sinon elle teste la recherche par mots-cles
                au lieu de la comprehension, et gonfle les scores

DEFINITION DES CHUNKS PERTINENTS
--------------------------------
Mesure faite avant d'ecrire ce script : 28 % des chunks francais etiquetes
"article N" sont en realite des RENVOIS depuis un autre article
("... l'article 43 ci-dessus ..."). Les compter comme bonnes reponses
recompenserait un systeme qui renvoie un debris. Seuls comptent les chunks
qui COMMENCENT par le vrai en-tete de l'article, dans toutes les langues.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arabic import normalize_for_search
from src.config import GOLDEN_DIR, INDEX_DIR

IDX = INDEX_DIR / "structural_bge-m3"
SOURCE = GOLDEN_DIR / "golden_source.jsonl"

STOP = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "a", "au", "aux",
        "en", "dans", "par", "pour", "sur", "que", "qui", "quel", "quelle", "quels",
        "est", "sont", "il", "elle", "mon", "ma", "mes", "son", "sa", "ses", "ce",
        "cette", "peut", "doit", "combien", "comment", "avec", "sans", "plus", "pas",
        "faire", "fait", "quelqu", "chaque", "tous", "tout", "après", "avant", "entre"}

SEUIL_RECOPIE = 0.5


def norm(t: str) -> str:
    """Normalisation de COMPARAISON : apostrophes typographiques, espaces,
    casse. N'altere pas les mots, contrairement a normalize_for_search."""
    t = t.replace("’", "'").replace("‘", "'").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def est_entete(c: dict) -> bool:
    """Le chunk commence-t-il par le VRAI en-tete de son article ?"""
    t = c["text"].lstrip()
    n = c["unit_ref"].split()[-1]
    if c["lang"] == "fr":
        return bool(re.match(rf"Article\s+{n}\b(?!\s*(ci-|du |de la |de l'|des ))", t))
    return bool(re.match(rf"(الماد[ةه]\s*[:\-]?\s*{n}\b|{n}\s*الماد[ةه])", t))


def mots_contenu(t: str) -> set[str]:
    return {m for m in re.findall(r"[a-zà-ÿ]+", norm(t)) if len(m) > 3 and m not in STOP}


def main():
    chunks = [json.loads(l) for l in (IDX / "chunks.jsonl").open(encoding="utf-8")]
    vrais = [c for c in chunks
             if re.match(r"article \d+$", c["unit_ref"] or "") and est_entete(c)]
    par_article: dict[int, list[dict]] = {}
    for c in vrais:
        par_article.setdefault(int(c["unit_ref"].split()[-1]), []).append(c)

    corpus_fr = norm(" ".join(c["text"] for c in chunks))
    corpus_ar = normalize_for_search(" ".join(c["text"] for c in chunks if c["lang"] == "ar"))

    source = [json.loads(l) for l in SOURCE.open(encoding="utf-8") if l.strip()]
    sortie, bloquants, alertes, lignes_audit = [], [], [], []

    for q in source:
        qid, typ = q["qid"], q["type"]
        entree = {k: v for k, v in q.items() if k not in ("article", "evidence", "absent_terms")}

        if typ == "unanswerable":
            presents = []
            for terme in q["absent_terms"]:
                if re.search(r"[؀-ۿ]", terme):
                    trouve = normalize_for_search(terme) in corpus_ar
                else:
                    trouve = re.search(rf"(?<![a-zà-ÿ]){re.escape(norm(terme))}(?![a-zà-ÿ])", corpus_fr)
                if trouve:
                    presents.append(terme)
            if presents:
                bloquants.append(f"{qid} : sujet PRESENT dans le corpus : {presents}")
            entree.update(answerable=False, relevant_chunk_ids=[], source_expected=[])
            lignes_audit.append((qid, typ, "-", "absent" if not presents else "PRESENT", "-", "-"))
            sortie.append(entree)
            continue

        n = q["article"]
        cibles = par_article.get(n, [])
        if not cibles:
            bloquants.append(f"{qid} : aucun vrai chunk pour l'article {n}")

        # 1. la preuve figure-t-elle mot pour mot dans l'article ?
        preuve = norm(q["evidence"])
        textes_fr = [norm(c["text"]) for c in cibles if c["lang"] == "fr"]
        preuve_ok = any(preuve in t for t in textes_fr)
        if not preuve_ok:
            bloquants.append(f"{qid} : preuve introuvable dans l'article {n} : « {q['evidence']} »")

        # 4. la question recopie-t-elle l'article ?
        recopie = 0.0
        if q["lang"] == "fr":
            mq = mots_contenu(q["question"])
            ma = mots_contenu(" ".join(c["text"] for c in cibles if c["lang"] == "fr"))
            recopie = len(mq & ma) / len(mq) if mq else 0.0
            if typ == "semantic" and recopie > SEUIL_RECOPIE:
                alertes.append(f"{qid} : {recopie:.0%} des mots de la question viennent de l'article")

        langs = sorted({c["lang"] for c in cibles})
        entree.update(
            answerable=True,
            article_ref=f"article {n}",
            evidence=q["evidence"],
            evidence_verified=preuve_ok,
            source_expected=["code_travail"],
            relevant_chunk_ids=sorted(c["id"] for c in cibles),
            relevant_langs=langs,
            lexical_overlap=round(recopie, 2),
        )
        lignes_audit.append((qid, typ, f"art. {n}", "OK" if preuve_ok else "ECHEC",
                             f"{recopie:.0%}" if q["lang"] == "fr" else "n/a", "+".join(langs)))
        sortie.append(entree)

    # --- rapport ---
    types = {}
    for e in sortie:
        types[e["type"]] = types.get(e["type"], 0) + 1

    md = ["# Audit du golden dataset", "",
          f"{len(sortie)} questions — " + ", ".join(f"{v} `{k}`" for k, v in sorted(types.items())), "",
          f"- controles bloquants en echec : **{len(bloquants)}**",
          f"- alertes de recopie (seuil {SEUIL_RECOPIE:.0%}) : **{len(alertes)}**", "",
          "| qid | type | article | preuve | recopie | langues pertinentes |",
          "|---|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in lignes_audit]
    if bloquants:
        md += ["", "## Echecs bloquants", *[f"- {x}" for x in bloquants]]
    if alertes:
        md += ["", "## Alertes", *[f"- {x}" for x in alertes]]
    md += ["", "## Limites connues", "",
           "- Types `cross_source` et `conflict` absents : ils exigent le CGNC et les "
           "circulaires DGI, pas encore ingeres.",
           "- Les preuves sont verifiees sur la version francaise consolidee ; les chunks "
           "arabes du meme article sont comptes pertinents par correspondance de numero.",
           "- Validation par preuve textuelle, non par un juriste : la reponse est garantie "
           "presente dans le texte, pas son interpretation juridique."]
    (GOLDEN_DIR / "audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md[:6]))
    print(f"\n{'qid':<5} {'type':<13} {'article':<9} {'preuve':<7} {'recopie':>8}  langues")
    print("-" * 62)
    for a, b, c, d, e, f in lignes_audit:
        print(f"{a:<5} {b:<13} {c:<9} {d:<7} {e:>8}  {f}")
    for x in bloquants:
        print("BLOQUANT :", x)
    for x in alertes:
        print("ALERTE   :", x)

    if bloquants:
        print("\n=> dataset REFUSE : corriger golden_source.jsonl")
        sys.exit(1)

    with (GOLDEN_DIR / "golden.jsonl").open("w", encoding="utf-8") as f:
        for e in sortie:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n=> dataset VALIDE : {len(sortie)} questions ecrites dans data/golden/golden.jsonl")


if __name__ == "__main__":
    main()
