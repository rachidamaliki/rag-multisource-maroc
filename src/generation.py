"""
Generation ancree avec citations.

Un bon retrieval ne garantit pas une bonne reponse : le modele peut recevoir le
bon passage et l'ignorer, ou le melanger avec ses connaissances internes.
L'ancrage se construit explicitement, en trois mecanismes :

  1. un prompt qui NUMEROTE les passages et exige des citations [1], [2]
  2. un verificateur qui controle que chaque citation existe reellement
     (le modele ecrit [4] alors qu'on ne lui a donne que 3 passages)
  3. un chemin de REFUS : si les passages ne couvrent pas la question, la
     reponse correcte est « INFORMATION INSUFFISANTE », pas une invention
"""
from __future__ import annotations

import re

REFUS = "INFORMATION INSUFFISANTE"

RAG_PROMPT = """Tu es un assistant juridique. Tu reponds UNIQUEMENT a partir des passages numerotes ci-dessous.

REGLES ABSOLUES :
1. Chaque affirmation doit etre suivie de sa source entre crochets : [1], [2]...
2. N'utilise JAMAIS de connaissance exterieure aux passages, meme si tu la crois vraie.
3. Si les passages ne permettent pas de repondre, ecris exactement : {refus}
   puis une phrase indiquant ce qui manque. N'invente aucun chiffre.
4. Reponds dans la langue de la question, en 5 phrases au maximum.

PASSAGES :
{context}

QUESTION : {question}

REPONSE :"""


def build_context(chunks: list[dict]) -> str:
    """Passages numerotes, avec leur source et leur reference.

    Afficher la reference (« article 205 ») aide le modele a citer juste, et
    l'utilisateur a verifier. Le meilleur passage est place en premier : les
    LLM exploitent mieux le debut du contexte (effet « lost in the middle »).
    """
    blocs = []
    for i, c in enumerate(chunks, start=1):
        ref = c.get("unit_ref") or "sans reference"
        texte = " ".join(c["text"].split())
        blocs.append(f"[{i}] ({c.get('source_id', '?')} — {ref} — {c.get('lang', '?')})\n{texte}")
    return "\n\n".join(blocs)


def build_prompt(question: str, chunks: list[dict]) -> str:
    return RAG_PROMPT.format(refus=REFUS, context=build_context(chunks), question=question)


def verify_citations(answer: str, n_contexts: int) -> dict:
    """Controle les citations de la reponse.

    - hallucinees : numeros cites qui ne correspondent a aucun passage fourni
    - sans_citation : la reponse affirme quelque chose sans rien citer
    Un refus n'a pas besoin de citation.
    """
    cites = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    valides = sorted({n for n in cites if 1 <= n <= n_contexts})
    hallucinees = sorted({n for n in cites if not 1 <= n <= n_contexts})
    refus = has_refused(answer)
    return {
        "citations": sorted(set(cites)),
        "valides": valides,
        "hallucinees": hallucinees,
        "sans_citation": not cites and not refus,
        "conforme": not hallucinees and (bool(cites) or refus),
    }


def has_refused(answer: str) -> bool:
    return REFUS.lower() in (answer or "").lower()


def generate(question: str, chunks: list[dict], llm) -> dict:
    """Appelle le LLM et renvoie la reponse avec son controle de citations."""
    prompt = build_prompt(question, chunks)
    reponse = llm.complete(prompt, temperature=0.0, max_tokens=1024) or ""
    return {
        "answer": reponse.strip(),
        "refused": has_refused(reponse),
        "citations": verify_citations(reponse, len(chunks)),
        "prompt": prompt,
    }
