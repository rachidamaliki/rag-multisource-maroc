"""
Tests de la fusion de resultats.

Le cas central est calcule A LA MAIN dans le test : c'est ce qui permet de
verifier l'implementation sans faire confiance a une librairie de reference
(il n'en existe pas pour RRF).

    pytest tests/test_fusion.py -q
"""
import pytest

from src.fusion import RRF_K, reciprocal_rank_fusion, weighted_score_fusion
from src.vectorstore import SearchResult


def liste(*paires):
    """Construit une liste de resultats classee, rangs a partir de 0."""
    return [SearchResult(cid, score, rang) for rang, (cid, score) in enumerate(paires)]


# Les deux listes de l'exemple de reference :
#   vectoriel : A (0,58)  B (0,54)  C (0,50)
#   BM25      : D (15,5)  A (15,3)  B (9,1)
VECT = liste(("A", 0.58), ("B", 0.54), ("C", 0.50))
BM25 = liste(("D", 15.5), ("A", 15.3), ("B", 9.1))


def test_rrf_scores_calcules_a_la_main():
    """RRF : somme de 1/(k + rang), rangs comptes a partir de 1."""
    attendu = {
        "A": 1 / 61 + 1 / 62,   # 1er en vectoriel, 2e en BM25
        "B": 1 / 62 + 1 / 63,   # 2e en vectoriel, 3e en BM25
        "D": 1 / 61,            # 1er en BM25, absent du vectoriel
        "C": 1 / 63,            # 3e en vectoriel, absent de BM25
    }
    obtenu = {r.chunk_id: r.score for r in reciprocal_rank_fusion([VECT, BM25])}
    assert obtenu == pytest.approx(attendu, abs=1e-12)


def test_rrf_favorise_le_consensus():
    """La propriete essentielle : un document trouve par les DEUX moteurs bat
    un document trouve par un seul, meme en premiere position.

    D est 1er chez BM25 mais absent du vectoriel : il doit passer derriere
    A et B, que les deux moteurs ont trouves."""
    ordre = [r.chunk_id for r in reciprocal_rank_fusion([VECT, BM25])]
    assert ordre == ["A", "B", "D", "C"]


def test_somme_naive_des_scores_est_biaisee():
    """Demonstration du probleme que RRF resout : en additionnant les scores
    bruts, l'echelle de BM25 (0 a 30) ecrase celle du cosinus (0 a 1)."""
    brut: dict[str, float] = {}
    for l in (VECT, BM25):
        for r in l:
            brut[r.chunk_id] = brut.get(r.chunk_id, 0.0) + r.score
    ordre_naif = sorted(brut, key=lambda c: -brut[c])
    # D, trouve par un seul moteur, devance B que les deux ont trouve
    assert ordre_naif.index("D") < ordre_naif.index("B")


def test_rrf_k_aplatit_les_ecarts():
    """Role de la constante k : sans elle, le 1er vaudrait 2x le 2e et la tete
    d'une seule liste dominerait la fusion."""
    sans_k = (1 / 1) / (1 / 2)                    # rang 1 vs rang 2
    avec_k = (1 / (RRF_K + 1)) / (1 / (RRF_K + 2))
    assert sans_k == pytest.approx(2.0)
    assert avec_k < 1.02


def test_poids_permet_de_neutraliser_une_liste():
    """Un poids nul doit rendre la liste correspondante sans effet sur l'ordre
    des documents qu'elle seule contient."""
    ordre = [r.chunk_id for r in reciprocal_rank_fusion([VECT, BM25], poids=[1.0, 0.0])]
    assert ordre[:3] == ["A", "B", "C"]           # l'ordre du vectoriel
    assert ordre[-1] == "D"                        # D ne remonte plus


def test_top_n_tronque():
    assert len(reciprocal_rank_fusion([VECT, BM25], top_n=2)) == 2


def test_listes_vides():
    assert reciprocal_rank_fusion([[], []]) == []
    assert weighted_score_fusion([[], []]) == []


def test_poids_de_taille_incorrecte_leve():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([VECT, BM25], poids=[1.0])


def test_ponderee_normalise_par_lot():
    """La fusion ponderee ramene chaque liste dans [0, 1] : le meilleur de
    chaque liste vaut 1, le pire vaut 0.

    C'est sa limite : le score depend du LOT. A change d'echelle uniquement
    parce que ses voisins ont change, alors que son score cosinus est le meme.
    """
    res = {r.chunk_id: r.score for r in weighted_score_fusion([VECT, BM25])}
    assert res["A"] == pytest.approx(1.0 + (15.3 - 9.1) / (15.5 - 9.1))
    assert res["C"] == pytest.approx(0.0)   # dernier du vectoriel, absent de BM25


def test_rangs_reattribues_en_sortie():
    """La liste fusionnee doit porter des rangs 0, 1, 2... coherents, sinon
    une fusion de fusions (utilisee en multi-source) serait faussee."""
    res = reciprocal_rank_fusion([VECT, BM25])
    assert [r.rank for r in res] == list(range(len(res)))
