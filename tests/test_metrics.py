"""
Tests de non-regression (Arc 6, checklist 5).

Objectif : un changement de configuration ne doit pas faire chuter vos
metriques sans que vous le sachiez. C'est exactement ce qui arrive dans
les projets RAG reels : quelqu'un change un chunk_size, le recall tombe
de 12 %, et personne ne s'en apercoit pendant deux mois.

Lancer :  pytest -q
"""
import numpy as np
import pytest

from src.metrics import (
    compare_runs,
    evaluate_retrieval,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.geometry import cosine_similarity


def test_cosine_matches_sklearn():
    """Verifie VOTRE implementation contre la reference."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    a = np.random.rand(5, 32).astype("float32")
    b = np.random.rand(3, 32).astype("float32")
    np.testing.assert_allclose(cosine_similarity(a, b), sk_cos(a, b), rtol=1e-5)


def test_recall_at_k_basic():
    assert recall_at_k(["c1", "c2", "c3"], {"c2"}, k=3) == 1.0
    assert recall_at_k(["c1", "c2", "c3"], {"c9"}, k=3) == 0.0
    assert recall_at_k(["c1", "c2"], {"c1", "c9"}, k=2) == 0.5


def test_mrr_position_matters():
    assert mrr(["c1", "c2"], {"c1"}) == 1.0
    assert mrr(["c9", "c1"], {"c1"}) == 0.5


def test_ndcg_ideal_is_one():
    rel = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], rel, k=3) == pytest.approx(1.0)


def test_recall_at_1_est_plafonne_par_le_nombre_de_pertinents():
    """LE PIEGE decouvert en validant sur donnees reelles.

    Avec 4 passages pertinents, un systeme PARFAIT (les 4 bons en tete)
    obtient recall@1 = 0,25. Un recall@1 bas ne signifie donc pas que le
    systeme classe mal — il peut simplement y avoir beaucoup de pertinents.
    """
    pertinents = {"c1", "c2", "c3", "c4"}
    parfait = ["c1", "c2", "c3", "c4", "c9"]
    assert recall_at_k(parfait, pertinents, k=1) == pytest.approx(0.25)
    assert recall_at_k(parfait, pertinents, k=4) == pytest.approx(1.0)
    # hit_rate et MRR, eux, voient bien que le systeme est parfait
    assert hit_rate_at_k(parfait, pertinents, k=1) == 1.0
    assert mrr(parfait, pertinents) == 1.0


def test_hit_rate_binaire():
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert hit_rate_at_k(["a"], set(), k=1) == 1.0        # rien a manquer


def test_precision_denominateur_est_k():
    """Rendre 2 resultats dont 2 bons pour un top-5 demande n'est pas
    une precision de 100 % : il manque trois places."""
    assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(0.4)
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_diagnostic_representation():
    """hit@kmax bas -> elargir ne sert a rien (le cas du vectoriel sur
    les requetes « article N » : 10 % en top-1 comme en top-5)."""
    golden = {f"q{i}": {f"bon{i}"} for i in range(10)}
    run = {f"q{i}": [f"faux{j}" for j in range(10)] for i in range(10)}
    m = evaluate_retrieval(run, golden, k_values=(1, 5, 10))
    assert m["hit@10"] == 0.0
    assert m["diagnostic"].startswith("representation")


def test_diagnostic_classement():
    """Le bon passage est present mais mal classe -> un reranker aiderait."""
    golden = {f"q{i}": {f"bon{i}"} for i in range(10)}
    run = {f"q{i}": ["faux"] * 4 + [f"bon{i}"] for i in range(10)}
    m = evaluate_retrieval(run, golden, k_values=(1, 5, 10))
    assert m["hit@1"] == 0.0 and m["hit@5"] == 1.0
    assert m["diagnostic"].startswith("classement")


def test_diagnostic_sain():
    golden = {f"q{i}": {f"bon{i}"} for i in range(10)}
    run = {f"q{i}": [f"bon{i}"] + ["faux"] * 9 for i in range(10)}
    m = evaluate_retrieval(run, golden, k_values=(1, 5, 10))
    assert m["diagnostic"] == "sain"


def test_evaluate_expose_le_nombre_de_pertinents():
    """Sans ce chiffre, le recall n'est pas interpretable."""
    golden = {"q1": {"a", "b"}, "q2": {"c"}}
    run = {"q1": ["a"], "q2": ["c"]}
    m = evaluate_retrieval(run, golden, k_values=(1,))
    assert m["pertinents_par_requete"] == pytest.approx(1.5)
    assert m["n_requetes"] == 2


def test_evaluate_ignore_les_requetes_absentes_du_run():
    golden = {"q1": {"a"}, "q2": {"b"}}
    run = {"q1": ["a"]}
    assert evaluate_retrieval(run, golden, k_values=(1,))["n_requetes"] == 1
    assert evaluate_retrieval({}, golden)["n_requetes"] == 0


def test_ndcg_penalise_le_mauvais_ordre():
    rel = {"a": 3.0, "b": 2.0, "c": 1.0}
    parfait = ndcg_at_k(["a", "b", "c"], rel, k=3)
    inverse = ndcg_at_k(["c", "b", "a"], rel, k=3)
    assert parfait == pytest.approx(1.0)
    assert inverse < parfait


def test_compare_runs_trie_par_recall_max():
    golden = {f"q{i}": {f"bon{i}"} for i in range(5)}
    bon = {f"q{i}": [f"bon{i}"] for i in range(5)}
    mauvais = {f"q{i}": ["faux"] for i in range(5)}
    lignes = compare_runs({"mauvais": mauvais, "bon": bon}, golden, k_values=(1, 5))
    assert lignes[0]["configuration"] == "bon"


@pytest.mark.skip(reason="A activer apres le tableau comparatif : garde-fou de non-regression")
def test_no_regression_recall():
    """TODO — charger reports/master_table.csv et verifier que la config
    retenue ne descend pas sous le seuil que vous aurez fixe."""
    ...
