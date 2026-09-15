"""
MULTI-SOURCE — routage, filtrage et resolution de conflits.

Ce module n'existe pas dans le roadmap original. Il est ajoute parce que
le corpus est multi-source, et c'est precisement ce qui distingue ce
projet d'un exercice : trois problemes reels apparaissent des qu'il y a
plus d'un document de reference.

--------------------------------------------------------------------
PROBLEME 1 — LE FILTRAGE DOIT SE FAIRE PENDANT LA RECHERCHE
--------------------------------------------------------------------
Erreur classique : recuperer le top-30 global puis jeter ce qui n'est
pas de la bonne source. Si les 30 meilleurs viennent tous du CGNC et
que l'utilisateur voulait le Code du travail, il reste ZERO resultat.
Le filtre doit etre applique DANS la requete (clause SQL avec pgvector,
ou masque d'ids avec FAISS).

--------------------------------------------------------------------
PROBLEME 2 — LA SOURCE LA PLUS VOLUMINEUSE ECRASE LES AUTRES
--------------------------------------------------------------------
Si le Code du travail fait 3 000 chunks et les circulaires 200, le
top-30 sera presque toujours 100 % Code du travail — non parce qu'il
est plus pertinent, mais parce qu'il est plus gros. C'est un biais de
volume, pas de qualite. La parade : garantir un quota minimal par
source avant la fusion.

--------------------------------------------------------------------
PROBLEME 3 — LES SOURCES SE CONTREDISENT
--------------------------------------------------------------------
Une circulaire DGI de 2025 peut contredire un guide de 2019. Le systeme
doit trancher selon une regle EXPLICITE : autorite d'abord (loi >
doctrine > guide), puis date. Et surtout, il doit le SIGNALER a
l'utilisateur plutot que de choisir en silence.

C'est la question d'architecture que pose un recruteur senior. Avoir une
reponse mesuree vaut plus que dix fonctionnalites supplementaires.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date

from .config import SOURCES, TOP_K_PER_SOURCE


@dataclass
class SourceMeta:
    """Metadonnees portees par CHAQUE chunk. Sans elles, ni filtrage,
    ni citation tracable, ni resolution de conflit."""
    source_id: str          # cle dans SOURCES
    authority: int          # 3 = loi/norme, 2 = doctrine, 1 = guide
    lang: str               # "fr" | "ar"
    doc_id: str
    unit_ref: str = ""      # "article 145", "rubrique 3.2.1", "circulaire 721/25"
    published: date | None = None
    url: str = ""


def detect_source(path_or_name: str) -> str:
    """
    TODO (J3) — deduire l'identifiant de source depuis le nom de fichier
    ou le dossier. Simple, mais a faire proprement : une source mal
    etiquetee au J3 pollue tout le reste du projet.
    """
    raise NotImplementedError("multi-source")


def build_filter(sources: list[str] | None = None,
                 langs: list[str] | None = None,
                 min_authority: int | None = None,
                 after: date | None = None) -> dict:
    """
    TODO (J9) — construire un filtre de metadonnees applicable pendant
    la recherche.

    Avec pgvector : une clause WHERE dans la meme requete SQL.
    Avec FAISS : un IDSelector, ou un pre-filtrage des ids autorises.

    Checklist Arc 3 : MESURER le surcout du filtrage. Filtrer sur une
    petite source peut paradoxalement RALENTIR la recherche HNSW, parce
    que le graphe doit explorer beaucoup plus loin pour trouver assez de
    candidats valides. C'est un resultat contre-intuitif et tres bien vu
    en entretien.
    """
    raise NotImplementedError("multi-source")


def balanced_merge(results_by_group: dict[str, list],
                   total_k: int = 30,
                   min_per_group: int = 2) -> list:
    """Fusion avec QUOTA : garantit un minimum de resultats par groupe.

    Le groupe peut etre une source (code_travail / cgnc / dgi) ou une LANGUE
    (fr / ar) : le mecanisme est le meme, seule la cle change.

    ---------------------------------------------------------------
    POURQUOI — mesure sur le corpus, requete francaise, index mixte
    ---------------------------------------------------------------
    Composition reelle du top-5 sans quota (40 requetes) :

        chunks FR d'AUTRES articles   73 %   <-- du bruit
        chunks FR du bon article      19 %
        chunks AR du bon article       7 %   <-- la cible
        chunks AR d'autres articles    1 %

    La requete etant en francais, les chunks francais gagnent par proximite
    de LANGUE, pas par pertinence. Le bon passage arabe existe et serait
    trouve — il est simplement evince.

    Effet du quota, meme mesure :

        politique                 cible AR    bon article (toute langue)
        aucune (top-5 brut)          45 %          92 %
        quota 3 FR + 2 AR            90 %         100 %
        filtre langue = AR           95 %          95 %

    Le quota double le recall cross-lingue SANS rien perdre : le bon article
    passe meme de 92 % a 100 %, parce qu'on cesse de gaspiller cinq places
    sur une seule langue.

    Le filtre strict fait un point de mieux sur la cible arabe, mais il exige
    de connaitre la langue voulue a l'avance. Le quota, lui, ne suppose rien —
    c'est le bon defaut quand on ne sait pas ce que l'utilisateur veut.

    ATTENTION : ce raisonnement vaut identiquement pour le biais de VOLUME
    entre sources. Une source de 3 000 chunks evince une source de 200 pour
    la meme raison — elle occupe simplement plus de place dans l'espace.
    """
    if not results_by_group:
        return []

    groupes = {g: list(res) for g, res in results_by_group.items() if res}
    if not groupes:
        return []

    retenus: list = []
    vus: set[str] = set()

    # 1. le quota : les `min_per_group` meilleurs de chaque groupe
    for g, res in groupes.items():
        for r in res[:min_per_group]:
            if r.chunk_id not in vus:
                vus.add(r.chunk_id)
                retenus.append(r)

    # 2. on complete au merite, tous groupes confondus, en respectant
    #    l'ordre d'origine de chaque liste (le rang, pas le score brut :
    #    les scores ne sont pas comparables entre groupes)
    restants = [r for res in groupes.values() for r in res[min_per_group:]]
    restants.sort(key=lambda r: r.rank)
    for r in restants:
        if len(retenus) >= total_k:
            break
        if r.chunk_id not in vus:
            vus.add(r.chunk_id)
            retenus.append(r)

    retenus = retenus[:total_k]
    # rangs reattribues : une fusion de fusions serait faussee sinon
    for nouveau_rang, r in enumerate(retenus):
        r.rank = nouveau_rang
    return retenus


def resolve_conflict(passages: list[dict]) -> dict:
    """
    TODO (J17) — trancher entre passages contradictoires.

    Regle : autorite decroissante, puis date la plus recente.
    Retourne : {"retenu": passage, "ecartes": [...], "conflit": bool}

    IMPORTANT : quand un conflit est detecte, la reponse generee doit le
    MENTIONNER ("selon la circulaire de 2025 [2], qui actualise le guide
    de 2019 [5]"). Un systeme qui tranche en silence est plus dangereux
    qu'un systeme qui hesite a voix haute.
    """
    raise NotImplementedError("multi-source")


def per_source_report(run: dict, golden: dict) -> "pd.DataFrame":
    """
    TODO (J19) — ventiler TOUTES les metriques par source.

    C'est une colonne supplementaire du tableau maitre, et souvent la
    plus revelatrice : un modele d'embedding peut exceller sur le
    juridique et s'effondrer sur les tableaux comptables. Sans cette
    ventilation, la moyenne globale cache le probleme.
    """
    raise NotImplementedError("multi-source")
