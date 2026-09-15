"""
LES METRIQUES DE RETRIEVAL — le fichier le plus important du projet.

Sans lui, tous les choix faits jusqu'ici (decoupage, normalisation, fusion,
quota) sont des OPINIONS. Avec lui, ce sont des decisions justifiees.

On l'a deja verifie plusieurs fois dans ce projet : l'intuition se trompe.
L'hybride RRF paraissait evidemment meilleur — il faisait chuter le top-1 de
95 % a 20 %. Le cross-lingue paraissait casse — c'etait la metrique qui l'etait.

--------------------------------------------------------------------
QUELLE METRIQUE POUR QUELLE QUESTION
--------------------------------------------------------------------
    recall@k     "le bon passage est-il quelque part dans le top-k ?"
                 LA metrique reine du RAG : si le passage n'est pas remonte,
                 aucun LLM au monde ne pourra repondre correctement.
                 Aveugle a la POSITION : 1er ou 10e, c'est pareil.

    precision@k  "quelle proportion de ce que j'ai remonte est utile ?"
                 Compte quand le contexte envoye au LLM est limite : chaque
                 place occupee par un mauvais passage est une place perdue.

    MRR          "a quelle position arrive le PREMIER bon resultat ?"
                 1/rang du premier succes. 1er -> 1,0 ; 5e -> 0,2.
                 Complementaire du recall : on peut avoir un bon recall et un
                 mauvais MRR (le bon passage est la, mais trop bas).

    NDCG@k       la seule qui gere la pertinence GRADUEE (tres pertinent /
                 moyennement / pas du tout) avec un amortissement selon la
                 position. La plus fine, la plus couteuse a annoter.

--------------------------------------------------------------------
COMMENT LIRE L'ECART ENTRE DEUX METRIQUES
--------------------------------------------------------------------
C'est la competence la plus utile de ce fichier. L'ecart DIT quel levier
actionner :

    recall@1 bas + recall@10 haut   -> probleme de CLASSEMENT
                                       -> un reranker reglera ca
    recall@1 = recall@10, les deux bas -> probleme de REPRESENTATION
                                       -> aucun reglage n'aidera, il faut
                                          un autre mecanisme de recherche

Le second cas est exactement ce qu'on a mesure sur « article N » : le
vectoriel plafonnait a 10 % en top-1 ET en top-5. Elargir ne servait a rien,
il fallait BM25. Sans cette lecture, on aurait passe des heures a regler
ef_search pour rien.
"""
from __future__ import annotations

import math

import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion des passages PERTINENTS retrouves dans le top-k.

        recall@k = |pertinents ∩ top_k| / |pertinents|

    Denominateur = nombre de pertinents, PAS k. C'est ce qui distingue le
    recall de la precision, et l'erreur la plus courante.

    Convention : pas de passage pertinent -> 1,0 (rien a manquer). Ce cas
    concerne les questions sans reponse, evaluees par le taux de refus et
    non par le recall.
    """
    if not relevant_ids:
        return 1.0
    trouves = set(retrieved_ids[:k]) & relevant_ids
    return len(trouves) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1,0 si AU MOINS un passage pertinent figure dans le top-k, sinon 0,0.

    Aussi appelee "success@k". Indispensable a cote du recall, pour une raison
    mecanique qu'on a decouverte en validant ce fichier sur donnees reelles :

        Quand une requete a 4 passages pertinents, recall@1 ne peut pas
        depasser 1/4 = 0,25. MEME AVEC UN SYSTEME PARFAIT.

    On avait donc mesure pour BM25 : recall@1 = 0,30 mais MRR = 0,95. Lu
    naivement, le recall@1 bas suggerait un probleme de classement — alors que
    le premier resultat etait le bon dans 95 % des cas. Le recall@1 ne mesurait
    pas la qualite du systeme, il mesurait le nombre de passages pertinents.

    Regle a retenir : recall@k n'est comparable entre requetes que si elles ont
    un nombre similaire de passages pertinents. hit_rate et MRR, eux, sont
    toujours comparables.

    Pour le RAG, hit_rate@5 est souvent la metrique la plus parlante : le LLM
    n'a besoin que d'UN bon passage pour repondre.
    """
    if not relevant_ids:
        return 1.0
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion du top-k qui est reellement pertinente.

        precision@k = |pertinents ∩ top_k| / k

    Denominateur = k, meme si moins de k resultats ont ete renvoyes : rendre
    3 resultats dont 3 bons pour un top-5 demande n'est pas une precision de
    100 %, il manque deux places.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / k


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal Rank : 1 / (rang du premier resultat pertinent).

        1er -> 1,00    2e -> 0,50    5e -> 0,20    absent -> 0,00

    Punit le fait d'avoir raison TROP TARD dans la liste. Utile quand
    l'utilisateur ne lit que le premier resultat — ou quand le budget de
    contexte du LLM ne permet d'en passer qu'un ou deux.
    """
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg(gains: list[float]) -> float:
    """Discounted Cumulative Gain : somme des gains amortis par la position.

        DCG = somme  gain_i / log2(i + 1)     (i a partir de 1)

    Le log2 encode l'idee qu'un gain en 2e position vaut ~63 % d'un gain en
    1re, un gain en 10e ~30 %. La decroissance est douce, pas brutale.
    """
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """NDCG : le DCG obtenu, divise par le DCG du classement IDEAL.

        NDCG@k = DCG@k / IDCG@k

    La normalisation rend la valeur comparable entre requetes : une requete
    ayant 10 passages pertinents et une n'en ayant qu'un obtiennent tous deux
    1,0 si leur classement est parfait. Sans elle, les requetes « riches »
    domineraient la moyenne.

    `relevance` accepte des gains gradues (3 = tres pertinent, 1 = marginal).
    Un passage absent du dictionnaire a un gain de 0.
    """
    if not relevance or k <= 0:
        return 0.0
    obtenu = dcg([relevance.get(cid, 0.0) for cid in retrieved_ids[:k]])
    idealise = dcg(sorted(relevance.values(), reverse=True)[:k])
    return obtenu / idealise if idealise > 0 else 0.0


def evaluate_retrieval(run: dict[str, list[str]],
                       golden: dict[str, set[str]],
                       k_values=(1, 3, 5, 10),
                       relevance: dict[str, dict[str, float]] | None = None) -> dict:
    """Agrege les metriques sur tout le jeu d'evaluation.

    run      : {query_id: [chunk_ids classes]}          ce que le systeme a rendu
    golden   : {query_id: {chunk_ids pertinents}}       ce qu'il fallait rendre
    relevance: {query_id: {chunk_id: gain}}             optionnel, pour le NDCG

    On calcule la MOYENNE PAR REQUETE (macro), pas globale : sinon une requete
    ayant beaucoup de passages pertinents peserait plus que les autres, et le
    score refleterait la composition du jeu de test plutot que la qualite du
    systeme.

    On renvoie aussi l'ecart-type et le nombre de requetes : une moyenne de
    0,78 sur 12 requetes avec un ecart-type de 0,4 ne dit pas la meme chose
    qu'une moyenne de 0,78 sur 200 requetes avec un ecart-type de 0,05.
    """
    qids = [q for q in golden if q in run]
    if not qids:
        return {"n_requetes": 0}

    out: dict[str, float] = {"n_requetes": len(qids)}

    # Nombre moyen de passages pertinents : il PLAFONNE le recall aux petits k
    # (recall@1 <= 1/|pertinents|). A afficher systematiquement a cote du
    # recall, sinon la lecture est faussee.
    out["pertinents_par_requete"] = float(np.mean([len(golden[q]) for q in qids]))

    for k in k_values:
        rec = [recall_at_k(run[q], golden[q], k) for q in qids]
        pre = [precision_at_k(run[q], golden[q], k) for q in qids]
        hit = [hit_rate_at_k(run[q], golden[q], k) for q in qids]
        out[f"recall@{k}"] = float(np.mean(rec))
        out[f"precision@{k}"] = float(np.mean(pre))
        out[f"hit@{k}"] = float(np.mean(hit))
        out[f"recall@{k}_ecart_type"] = float(np.std(rec))

    rr = [mrr(run[q], golden[q]) for q in qids]
    out["mrr"] = float(np.mean(rr))
    out["mrr_ecart_type"] = float(np.std(rr))

    if relevance:
        for k in k_values:
            nd = [ndcg_at_k(run[q], relevance.get(q, {}), k) for q in qids]
            out[f"ndcg@{k}"] = float(np.mean(nd))

    # -----------------------------------------------------------------
    # Diagnostic automatique — base sur hit@k, PAS sur recall@k
    # -----------------------------------------------------------------
    # Premiere version de ce diagnostic : elle comparait recall@1 a
    # recall@kmax. Elle se trompait systematiquement. Sur les requetes
    # « article N » (4,2 passages pertinents en moyenne), BM25 obtenait
    # recall@1 = 0,30 et recall@10 = 0,93 : un ecart de 0,63 qui declenchait
    # « probleme de classement, un reranker devrait aider ». Or le MRR etait
    # de 0,95 — le premier resultat etait le bon dans 95 % des cas. L'ecart
    # ne venait pas du systeme mais du PLAFOND MECANIQUE : avec 4 passages
    # pertinents, recall@1 ne peut pas depasser 0,25.
    #
    # hit@k n'a pas ce plafond : il vaut 1 des qu'un bon passage est present.
    # L'ecart hit@1 -> hit@kmax mesure donc vraiment un defaut de classement.
    kmax = max(k_values)
    h1, hmax = out.get("hit@1", 0.0), out.get(f"hit@{kmax}", 0.0)
    if hmax < 0.5:
        out["diagnostic"] = "representation — elargir n'aidera pas, changer de mecanisme"
    elif hmax - h1 > 0.15:
        out["diagnostic"] = "classement — un reranker devrait aider"
    else:
        out["diagnostic"] = "sain"
    return out


def compare_runs(runs: dict[str, dict[str, list[str]]],
                 golden: dict[str, set[str]],
                 k_values=(1, 5, 10)) -> "list[dict]":
    """Evalue plusieurs configurations et renvoie une ligne par configuration.

    C'est la brique du tableau comparatif final : une ligne par combinaison
    (decoupage x recherche x fusion x reranking), toutes evaluees sur le MEME
    jeu de test. C'est la comparabilite qui donne sa valeur au tableau, pas
    les valeurs absolues.
    """
    lignes = []
    for nom, run in runs.items():
        m = evaluate_retrieval(run, golden, k_values=k_values)
        lignes.append({"configuration": nom, **m})
    cle = f"recall@{max(k_values)}"
    return sorted(lignes, key=lambda l: -l.get(cle, 0))
