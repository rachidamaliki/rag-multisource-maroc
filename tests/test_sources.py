"""
Tests du quota multi-groupe (langue ou source).

    pytest tests/test_sources.py -q
"""
import pytest

from src.sources import balanced_merge
from src.vectorstore import SearchResult


def liste(prefixe, n, depart=0):
    """n resultats classes, rangs a partir de `depart`."""
    return [SearchResult(f"{prefixe}{i}", 1.0 - 0.01 * (depart + i), depart + i)
            for i in range(n)]


def test_quota_garantit_une_place_au_petit_groupe():
    """Le cas reel : une langue (ou source) volumineuse en evince une petite.
    Sans quota, les 5 places iraient toutes au groupe le plus gros."""
    groupes = {"fr": liste("FR", 20), "ar": liste("AR", 20)}
    res = balanced_merge(groupes, total_k=5, min_per_group=2)
    ids = [r.chunk_id for r in res]
    assert sum(1 for i in ids if i.startswith("AR")) >= 2
    assert sum(1 for i in ids if i.startswith("FR")) >= 2
    assert len(res) == 5


def test_quota_respecte_total_k():
    groupes = {"a": liste("A", 10), "b": liste("B", 10), "c": liste("C", 10)}
    assert len(balanced_merge(groupes, total_k=4, min_per_group=2)) == 4


def test_quota_peut_depasser_si_min_par_groupe_trop_grand():
    """Garde-fou : 3 groupes x 3 minimum = 9 > total_k. La troncature a
    total_k doit primer, sinon le top-k promis a l'appelant serait faux."""
    groupes = {"a": liste("A", 5), "b": liste("B", 5), "c": liste("C", 5)}
    res = balanced_merge(groupes, total_k=5, min_per_group=3)
    assert len(res) == 5


def test_groupe_plus_petit_que_le_quota():
    """Une source qui n'a qu'un seul resultat ne doit pas casser la fusion
    ni faire perdre des places aux autres."""
    groupes = {"gros": liste("G", 20), "petit": liste("P", 1)}
    res = balanced_merge(groupes, total_k=5, min_per_group=2)
    ids = [r.chunk_id for r in res]
    assert "P0" in ids
    assert len(res) == 5


def test_deduplication():
    """Un meme chunk peut figurer dans deux listes (ex. fusion de fusions).
    Il ne doit apparaitre qu'une fois."""
    commun = SearchResult("X", 0.9, 0)
    groupes = {"a": [commun] + liste("A", 5, depart=1),
               "b": [commun] + liste("B", 5, depart=1)}
    ids = [r.chunk_id for r in balanced_merge(groupes, total_k=10, min_per_group=2)]
    assert ids.count("X") == 1


def test_rangs_reattribues():
    """Les rangs de sortie doivent etre 0, 1, 2... : une fusion de fusions
    (quota par langue applique apres une fusion par source) serait faussee
    si les rangs d'origine etaient conserves."""
    groupes = {"fr": liste("FR", 10), "ar": liste("AR", 10)}
    res = balanced_merge(groupes, total_k=6, min_per_group=2)
    assert [r.rank for r in res] == list(range(6))


def test_completion_au_merite_apres_le_quota():
    """Apres le quota, on complete par RANG et non par score brut : les
    scores ne sont pas comparables entre groupes (cosinus vs BM25)."""
    groupes = {
        "a": [SearchResult("A0", 0.99, 0), SearchResult("A1", 0.98, 1), SearchResult("A2", 0.97, 2)],
        "b": [SearchResult("B0", 15.0, 0), SearchResult("B1", 14.0, 1), SearchResult("B2", 13.0, 2)],
    }
    res = balanced_merge(groupes, total_k=5, min_per_group=2)
    # les 4 premiers viennent du quota ; le 5e est un rang 2, pas le plus gros score
    assert res[4].chunk_id in {"A2", "B2"}


def test_entrees_vides():
    assert balanced_merge({}) == []
    assert balanced_merge({"a": [], "b": []}) == []


def test_un_seul_groupe_preserve_l_ordre():
    res = balanced_merge({"fr": liste("FR", 5)}, total_k=5, min_per_group=2)
    assert [r.chunk_id for r in res] == ["FR0", "FR1", "FR2", "FR3", "FR4"]
