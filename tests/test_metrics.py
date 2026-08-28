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

from src.metrics import recall_at_k, mrr, ndcg_at_k
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


@pytest.mark.skip(reason="A activer apres le J19 : garde-fou de non-regression")
def test_no_regression_recall():
    """TODO — charger reports/master_table.csv et verifier que la config
    retenue ne descend pas sous le seuil que vous aurez fixe."""
    ...
