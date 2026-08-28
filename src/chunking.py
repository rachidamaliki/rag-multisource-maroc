"""
ARC 1 — Les strategies de decoupage.

Concept a acquerir :
  Le decoupage est la variable la plus sous-estimee de la qualite d'un RAG.
  Un chunk trop grand noie l'information pertinente dans du bruit ; trop
  petit, il perd le contexte necessaire pour etre compris seul.
  Le pire cas : la reponse est coupee en deux entre deux chunks
  ("fragmentation du contexte") — aucun des deux ne suffit a repondre.
"""
from dataclasses import dataclass, field
from typing import Any
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Compter en TOKENS, pas en caracteres. L'arabe consomme beaucoup plus
    de tokens par mot que le francais — le verifier vous-meme est instructif."""
    return len(_enc.encode(text))


@dataclass
class Chunk:
    """Un chunk porte toujours ses metadonnees : sans elles, pas de citation
    possible en Arc 7, et pas de filtrage en Arc 3."""
    id: str
    text: str
    doc_id: str
    position: int
    n_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.n_tokens:
            self.n_tokens = count_tokens(self.text)


# ---------------------------------------------------------------
# Les 4 strategies. Meme signature pour toutes -> comparables en Arc 6.
# ---------------------------------------------------------------

def chunk_fixed(text: str, doc_id: str, max_tokens: int = 400, overlap: int = 60) -> list[Chunk]:
    """
    TODO (J4) — Strategie 1 : fenetres de tokens fixes avec recouvrement.
    C'est la BASELINE. Simple, rapide, et elle coupe au milieu des phrases.
    Toutes les autres strategies devront prouver qu'elles font mieux qu'elle.
    """
    raise NotImplementedError("Arc 1")


def chunk_by_sentence(text: str, doc_id: str, max_tokens: int = 400, overlap: int = 1) -> list[Chunk]:
    """
    TODO (J4) — Strategie 2 : ne jamais couper au milieu d'une phrase.
    Attention : le decoupage de phrases en arabe ne suit pas les memes
    regles qu'en francais (ponctuation '؟' '،' , absence de majuscules).
    Ecrivez votre propre regex, c'est le but de l'arc.
    """
    raise NotImplementedError("Arc 1")


def chunk_semantic(text: str, doc_id: str, embed_fn, max_tokens: int = 600,
                   similarity_drop: float = 0.25) -> list[Chunk]:
    """
    TODO (J5) — Strategie 3 : couper la ou le SENS change.
    Methode : embedder chaque phrase, calculer la similarite entre phrases
    consecutives, couper quand elle chute de plus de `similarity_drop`.
    C'est couteux (un embedding par phrase) — le mesurer fait partie du travail.
    """
    raise NotImplementedError("Arc 1")


def chunk_structural(text: str, doc_id: str, max_tokens: int = 800) -> list[Chunk]:
    """
    TODO (J5) — Strategie 4 : respecter la structure du document.
    BOSS FIGHT : un tableau ou un article de loi ne doit JAMAIS etre coupe.
    Sur un corpus juridique (CGNC, Code du travail), l'unite atomique est
    l'article. Un chunker qui coupe l'article 145 en deux rend le systeme
    inutilisable — et c'est exactement ce que fait le chunking naif.
    """
    raise NotImplementedError("Arc 1 — boss fight")


def context_fragmentation_rate(chunks: list[Chunk], golden_passages: list[str]) -> float:
    """
    TODO (J5) — Metrique maison : quelle proportion des reponses attendues
    se retrouve eclatee sur 2 chunks ou plus ? C'est LE chiffre qui justifie
    votre choix de strategie, bien plus qu'une intuition.
    """
    raise NotImplementedError("Arc 1")


CHUNKERS = {
    "fixed": chunk_fixed,
    "sentence": chunk_by_sentence,
    "semantic": chunk_semantic,
    "structural": chunk_structural,
}
