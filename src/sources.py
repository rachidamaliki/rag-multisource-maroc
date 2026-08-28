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


def balanced_merge(results_by_source: dict[str, list],
                   total_k: int = 30,
                   min_per_source: int = 3) -> list:
    """
    TODO (J12) — fusion equilibree entre sources.

    Garantit `min_per_source` resultats par source avant de completer
    avec les meilleurs scores globaux. Corrige le biais de volume
    decrit plus haut.

    A COMPARER avec la RRF simple sur votre golden dataset : sur les
    questions qui exigent de croiser deux sources (type `multi_hop`),
    la fusion equilibree devrait gagner nettement. C'est une ligne de
    plus dans le tableau maitre, et une que personne d'autre n'a.
    """
    raise NotImplementedError("multi-source")


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
