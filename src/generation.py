"""
ARC 7 — La generation ancree (grounded) avec citations.

Concept a acquerir (piege documente) :
  Un bon retrieval NE GARANTIT PAS une bonne reponse. Le modele peut
  recevoir le passage parfait et l'ignorer, ou melanger le passage avec
  ses connaissances internes. L'ancrage se construit explicitement,
  cote generation — ce n'est pas gratuit.

Trois mecanismes a implementer :
  1. Un prompt qui NUMEROTE les passages et exige des citations [1], [2]
  2. Un verificateur qui controle que chaque citation existe reellement
     (on attrape ainsi les citations hallucinees : le modele ecrit [4]
     alors qu'on ne lui a donne que 3 passages)
  3. Un chemin de REFUS : si les passages ne couvrent pas la question,
     la reponse correcte est "je ne sais pas", pas une invention.

En contexte juridique ou comptable, le point 3 n'est pas un detail de
confort : une reponse inventee sur un article de loi est un risque reel
pour le client. C'est un argument commercial autant que technique.
"""
from __future__ import annotations
import re

RAG_PROMPT = """Tu reponds UNIQUEMENT a partir des passages numerotes ci-dessous.

REGLES ABSOLUES :
1. Chaque affirmation doit etre suivie de sa source : [1], [2]...
2. N'utilise JAMAIS de connaissance exterieure aux passages.
3. Si les passages ne permettent pas de repondre, ecris exactement :
   "INFORMATION INSUFFISANTE" puis explique ce qui manque.
4. Reponds dans la langue de la question.

PASSAGES :
{context}

QUESTION : {question}

REPONSE :"""


def build_context(chunks: list[dict]) -> str:
    """
    TODO (J17) — formater les chunks en passages numerotes [1], [2]...

    Checklist 4 de l'arc : tester l'ORDRE des passages.
    Effet "lost in the middle" : les LLM traitent mieux ce qui est au debut
    et a la fin du contexte, et negligent le milieu. Mettre le meilleur
    passage en premier ou en dernier change mesurablement la qualite.
    A tester, pas a supposer.
    """
    raise NotImplementedError("Arc 7")


def verify_citations(answer: str, n_contexts: int) -> dict:
    """
    TODO (J17) — extraire tous les [n] de la reponse et verifier que
    1 <= n <= n_contexts.
    Retourne : {"valides": [...], "hallucinees": [...], "sans_citation": bool}

    Une phrase affirmative sans aucune citation est un signal d'alerte
    aussi fort qu'une citation inventee.
    """
    raise NotImplementedError("Arc 7")


def has_refused(answer: str) -> bool:
    """TODO (J18) — detecter le refus ("INFORMATION INSUFFISANTE")."""
    raise NotImplementedError("Arc 7")


def adversarial_refusal_test(pipeline, impossible_questions: list[str]) -> dict:
    """
    TODO (J18) — BOSS FIGHT Arc 7.

    Ecrire 15 questions auxquelles votre corpus ne peut PAS repondre
    (sujets voisins mais absents, questions hors-domaine, questions
    a premisse fausse : "quel est le taux prevu par l'article 999 ?").

    Critere de reussite : au plus 1 ou 2 hallucinations sur 15.
    Si vous echouez, l'arc n'est pas fini — renforcez la logique de refus.

    C'est un test que presque personne ne fait, et c'est le premier que
    posera un client serieux.
    """
    raise NotImplementedError("Arc 7 — boss fight")
