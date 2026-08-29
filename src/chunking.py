"""
Les 4 strategies de decoupage.

POURQUOI LE DECOUPAGE EST LA VARIABLE LA PLUS SOUS-ESTIMEE
----------------------------------------------------------
On ne cherche jamais dans un document entier : on cherche dans des morceaux.
La taille et les frontieres de ces morceaux decident de tout.

  - chunk trop grand  -> l'information pertinente est noyee dans du bruit,
                         et l'embedding "moyenne" plusieurs sujets
  - chunk trop petit  -> le morceau perd le contexte necessaire pour etre
                         compris seul ("il" renvoie a quoi ?)
  - frontiere mal placee -> LE pire cas : la reponse est coupee en deux.
                         Aucun des deux morceaux ne repond, et le systeme
                         echoue sans qu'aucune erreur ne s'affiche.

Ce dernier cas s'appelle la FRAGMENTATION DU CONTEXTE. Sur un corpus
juridique, il est fatal : un article coupe en deux devient inutilisable.

Les 4 strategies ont la meme signature -> elles sont comparables dans le
tableau d'evaluation final. C'est tout l'interet : on ne choisit pas la
"meilleure" par intuition, on la mesure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import tiktoken

from .arabic import normalize_for_search

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Compter en TOKENS, pas en caracteres.

    Important : l'arabe consomme environ 3x plus de tokens par mot que le
    francais avec ce tokeniseur (concu pour l'anglais). Un chunk de 400 tokens
    contient donc beaucoup moins de texte arabe que de texte francais — c'est
    un biais reel a garder en tete quand on compare les langues.
    """
    return len(_enc.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    """Un chunk porte toujours ses metadonnees.

    Sans elles : pas de citation tracable (Arc 7), pas de filtrage par source
    (Arc 3), pas de resolution de conflit. Les metadonnees ne sont pas un
    supplement, elles font partie de l'unite de recherche.
    """
    id: str
    text: str                    # exact, pour l'affichage et la citation
    text_norm: str = ""          # canonique, pour l'embedding et BM25
    doc_id: str = ""
    source_id: str = ""
    authority: int = 0
    lang: str = "fr"
    page: int = 0
    position: int = 0
    unit_ref: str = ""           # "article 145" quand on a pu l'identifier
    n_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text_norm:
            self.text_norm = normalize_for_search(self.text)
        if not self.n_tokens:
            self.n_tokens = count_tokens(self.text)


# ---------------------------------------------------------------
# Decoupage en phrases (FR + AR)
# ---------------------------------------------------------------

# Le francais s'appuie sur . ! ? suivis d'une majuscule.
# L'arabe n'a PAS de majuscules et utilise une ponctuation propre :
#   ؟ (point d'interrogation), ؛ (point-virgule), ، (virgule)
# Un decoupeur concu pour l'anglais coupe donc mal l'arabe : c'est une des
# raisons pour lesquelles les pipelines RAG anglophones echouent sur l'arabe.
FIN_PHRASE = re.compile(r"(?<=[.!?؟؛])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    phrases = [p.strip() for p in FIN_PHRASE.split(text) if p and p.strip()]
    return phrases or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------
# Fabrique commune
# ---------------------------------------------------------------

def _make_chunk(texte: str, page: dict, doc: dict, pos: int, unit_ref: str = "") -> Chunk:
    return Chunk(
        id=f"{doc['doc_id']}::p{page['page']}::c{pos}",
        text=texte,
        doc_id=doc["doc_id"],
        source_id=doc["source_id"],
        authority=doc["authority"],
        lang=page["lang"],
        page=page["page"],
        position=pos,
        unit_ref=unit_ref,
    )


# ---------------------------------------------------------------
# Strategie 1 : fenetres de tokens fixes  [BASELINE]
# ---------------------------------------------------------------

def chunk_fixed(doc: dict, max_tokens: int = 400, overlap: int = 60) -> list[Chunk]:
    """Decoupe tous les `max_tokens` tokens, avec `overlap` tokens de recouvrement.

    C'est LA BASELINE. Elle est simple, rapide, et elle coupe au milieu des
    phrases sans etat d'ame. Son role n'est pas d'etre bonne : c'est d'etre le
    point de comparaison que les autres devront battre AVEC DES CHIFFRES.

    Le recouvrement est une rustine : en faisant se chevaucher les chunks, on
    espere qu'une information coupee se retrouvera entiere dans l'un des deux.
    Ca marche a moitie, et ca coute des tokens dupliques.
    """
    chunks, pos = [], 0
    for page in doc["pages"]:
        ids = _enc.encode(page["text"], disallowed_special=())
        pas = max(1, max_tokens - overlap)
        for start in range(0, len(ids), pas):
            fenetre = ids[start:start + max_tokens]
            if not fenetre:
                continue
            texte = _enc.decode(fenetre).strip()
            if len(texte) < 20:
                continue
            chunks.append(_make_chunk(texte, page, doc, pos))
            pos += 1
            if start + max_tokens >= len(ids):
                break
    return chunks


# ---------------------------------------------------------------
# Strategie 2 : frontieres de phrases
# ---------------------------------------------------------------

def chunk_by_sentence(doc: dict, max_tokens: int = 400, overlap_sentences: int = 1) -> list[Chunk]:
    """Accumule des phrases entieres jusqu'a `max_tokens`. Ne coupe jamais
    au milieu d'une phrase.

    Gain attendu : chaque chunk est grammaticalement complet, donc plus
    facile a embedder correctement et lisible tel quel dans une citation.
    """
    chunks, pos = [], 0
    for page in doc["pages"]:
        phrases = split_sentences(page["text"])
        courant, n_tok = [], 0
        for ph in phrases:
            t = count_tokens(ph)
            if courant and n_tok + t > max_tokens:
                chunks.append(_make_chunk(" ".join(courant), page, doc, pos))
                pos += 1
                courant = courant[-overlap_sentences:] if overlap_sentences else []
                n_tok = sum(count_tokens(p) for p in courant)
            courant.append(ph)
            n_tok += t
        if courant:
            chunks.append(_make_chunk(" ".join(courant), page, doc, pos))
            pos += 1
    return chunks


# ---------------------------------------------------------------
# Strategie 3 : rupture semantique
# ---------------------------------------------------------------

def chunk_semantic(doc: dict, embed_fn: Callable, max_tokens: int = 600,
                   similarity_drop: float = 0.25) -> list[Chunk]:
    """Coupe la ou le SUJET change.

    Methode : embedder chaque phrase, calculer la similarite entre phrases
    consecutives, couper quand elle chute brutalement. L'idee est qu'une
    rupture de similarite signale un changement de theme.

    COUT : un embedding PAR PHRASE. Sur 500 pages, c'est des dizaines de
    milliers d'appels. Cette strategie doit donc prouver qu'elle vaut son
    prix — c'est exactement ce que le tableau comparatif mesurera
    (colonne "cout" a cote de la colonne "recall").
    """
    chunks, pos = [], 0
    for page in doc["pages"]:
        phrases = split_sentences(page["text"])
        if len(phrases) < 2:
            if phrases:
                chunks.append(_make_chunk(phrases[0], page, doc, pos)); pos += 1
            continue

        vecs = embed_fn(phrases)                       # (n, d), deja normalises
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)    # similarite i / i+1

        # Seuil relatif : on coupe quand la similarite descend nettement
        # sous la moyenne locale. Un seuil absolu ne se transpose pas d'un
        # corpus a l'autre — le niveau general de similarite varie beaucoup.
        seuil = sims.mean() - similarity_drop * sims.std()

        courant, n_tok = [phrases[0]], count_tokens(phrases[0])
        for i, ph in enumerate(phrases[1:]):
            t = count_tokens(ph)
            rupture = sims[i] < seuil
            if courant and (rupture or n_tok + t > max_tokens):
                chunks.append(_make_chunk(" ".join(courant), page, doc, pos))
                pos += 1
                courant, n_tok = [], 0
            courant.append(ph)
            n_tok += t
        if courant:
            chunks.append(_make_chunk(" ".join(courant), page, doc, pos)); pos += 1
    return chunks


# ---------------------------------------------------------------
# Strategie 4 : structure du document  [celle qui devrait gagner ici]
# ---------------------------------------------------------------

def chunk_structural(doc: dict, max_tokens: int = 800) -> list[Chunk]:
    """Decoupe sur les UNITES ATOMIQUES du document : un chunk = un article.

    Les positions des articles ont ete reperees a l'ingestion
    (`unit_refs` de chaque page). On coupe exactement dessus.

    Sur un corpus juridique, c'est structurellement la bonne reponse :
    l'article EST l'unite de sens et de citation. Un utilisateur demande
    "que dit l'article 145 ?" — pas "que disent les 400 tokens autour de
    l'article 145 ?".

    Bonus decisif : chaque chunk porte sa reference (`unit_ref`), donc la
    citation devient exacte et verifiable au lieu d'un vague "page 42".
    """
    chunks, pos = [], 0
    for page in doc["pages"]:
        refs = page.get("unit_refs", [])
        texte = page["text"]

        if not refs:      # page sans article identifie (sommaire, annexe...)
            for c in chunk_by_sentence({**doc, "pages": [page]}, max_tokens=max_tokens):
                c.position = pos; pos += 1
                chunks.append(c)
            continue

        # frontieres = debut de chaque article, + fin de page
        bornes = [r["start"] for r in refs] + [len(texte)]
        # texte avant le premier article (intitule de chapitre, etc.)
        if bornes[0] > 50:
            chunks.append(_make_chunk(texte[:bornes[0]].strip(), page, doc, pos)); pos += 1

        for i, r in enumerate(refs):
            bloc = texte[bornes[i]:bornes[i + 1]].strip()
            if len(bloc) < 20:
                continue
            # Un article trop long est redecoupe en phrases, mais chaque
            # morceau CONSERVE la reference de l'article : la tracabilite
            # survit au decoupage.
            if count_tokens(bloc) > max_tokens:
                sous = split_sentences(bloc)
                courant, n_tok = [], 0
                for ph in sous:
                    t = count_tokens(ph)
                    if courant and n_tok + t > max_tokens:
                        chunks.append(_make_chunk(" ".join(courant), page, doc, pos, r["ref"]))
                        pos += 1; courant, n_tok = [], 0
                    courant.append(ph); n_tok += t
                if courant:
                    chunks.append(_make_chunk(" ".join(courant), page, doc, pos, r["ref"])); pos += 1
            else:
                chunks.append(_make_chunk(bloc, page, doc, pos, r["ref"])); pos += 1
    return chunks


# ---------------------------------------------------------------
# Mesure : la fragmentation du contexte
# ---------------------------------------------------------------

def context_fragmentation_rate(chunks: list[Chunk], passages: list[str]) -> dict:
    """Quelle proportion des passages de reference se retrouve ECLATEE sur
    plusieurs chunks ?

    C'est LA metrique qui justifie le choix d'une strategie de decoupage,
    bien avant toute intuition. Elle se calcule sans modele et sans LLM :
    on verifie simplement si chaque passage attendu tient dans un seul chunk.

    Retourne le taux de fragmentation ET le detail, pour pouvoir inspecter
    les cas problematiques a la main.
    """
    from .arabic import normalize_for_search as N

    textes = [N(c.text) for c in chunks]
    entiers, fragmentes = 0, []
    for p in passages:
        pn = N(p)
        if any(pn in t for t in textes):
            entiers += 1
        else:
            fragmentes.append(p[:80])
    total = max(len(passages), 1)
    return {
        "taux_fragmentation": 1 - entiers / total,
        "entiers": entiers,
        "total": total,
        "exemples_fragmentes": fragmentes[:5],
    }


def chunk_stats(chunks: list[Chunk]) -> dict:
    toks = [c.n_tokens for c in chunks] or [0]
    return {
        "n_chunks": len(chunks),
        "tokens_moy": float(np.mean(toks)),
        "tokens_med": float(np.median(toks)),
        "tokens_ecart_type": float(np.std(toks)),
        "tokens_min": int(np.min(toks)),
        "tokens_max": int(np.max(toks)),
        "avec_reference": sum(1 for c in chunks if c.unit_ref),
    }


CHUNKERS: dict[str, Callable] = {
    "fixed": chunk_fixed,
    "sentence": chunk_by_sentence,
    "semantic": chunk_semantic,     # exige embed_fn
    "structural": chunk_structural,
}
