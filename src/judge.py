"""
ARC 6 (partie 2) — Les metriques de generation (LLM-as-judge).

Le retrieval se mesure avec des ensembles. La generation, non : il faut
juger si une reponse en langue naturelle est FIDELE aux sources.

Deux metriques a implementer :
  - FAITHFULNESS (fidelite) : chaque affirmation de la reponse est-elle
    reellement soutenue par les passages fournis ? C'est la mesure de
    l'hallucination. La plus importante en contexte juridique/comptable.
  - ANSWER RELEVANCE : la reponse repond-elle vraiment a la question ?

PIEGE MAJEUR (documente dans le roadmap) :
  Un juge LLM sans grille et sans calibrage est PIRE que pas de metrique
  du tout — il donne une fausse confiance. Obligation : noter 10 exemples
  a la main, puis verifier que le juge est d'accord avec vous. Si l'accord
  est faible, la grille est mauvaise, pas vos annotations.
"""
from __future__ import annotations

FAITHFULNESS_PROMPT = """Tu es un evaluateur rigoureux. On te donne une QUESTION,
des PASSAGES SOURCES, et une REPONSE generee.

Decoupe la REPONSE en affirmations elementaires. Pour chacune, dis si elle est
DIRECTEMENT soutenue par les PASSAGES (oui/non). Une affirmation vraie dans
l'absolu mais absente des passages compte comme NON soutenue.

Retourne uniquement du JSON :
{{"affirmations": [{{"texte": "...", "soutenue": true, "passage": 2}}],
  "score": <nb_soutenues / nb_total>}}

QUESTION: {question}
PASSAGES: {contexts}
REPONSE: {answer}
"""

ANSWER_RELEVANCE_PROMPT = """TODO (J16) — a ecrire vous-meme.
Objectif : noter de 0 a 1 si la reponse traite reellement la question posee.
Une reponse exacte mais hors-sujet doit obtenir un score bas.
"""


def judge_faithfulness(question: str, contexts: list[str], answer: str) -> dict:
    """TODO (J16) — appeler le LLM avec FAITHFULNESS_PROMPT, parser le JSON,
    gerer les sorties malformees (ca arrive : prevoir un retry + un fallback)."""
    raise NotImplementedError("Arc 6")


def calibrate_judge(hand_labeled: list[dict]) -> dict:
    """
    TODO (J16) — ETAPE OBLIGATOIRE.
    Vous notez 10 exemples a la main, le juge les note aussi, puis on
    mesure l'accord (correlation, ou % d'accord exact).
    Rapportez ce chiffre dans le README : c'est ce qui rend TOUTES vos
    autres metriques de generation credibles.
    """
    raise NotImplementedError("Arc 6")
