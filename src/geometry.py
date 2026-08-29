"""
ARC 0 — La geometrie des embeddings.

numpy uniquement. Aucun import de sklearn ici (sauf dans les tests, pour verifier).

Concept :
  Un embedding est un point dans un espace a N dimensions. "Similaire" =
  "pointe dans la meme direction". La similarite cosinus mesure l'angle,
  pas la distance : deux textes de longueurs tres differentes mais de meme
  sens pointent dans la meme direction, alors que leur distance euclidienne
  serait grande. C'est pour ca que le cosinus gagne sur du texte.

--------------------------------------------------------------------
CONTRAT DE FORME — respecte par TOUT le reste du projet
--------------------------------------------------------------------
    a (d,)   x b (d,)   -> scalaire
    a (n, d) x b (d,)   -> (n,)
    a (d,)   x b (m, d) -> (m,)
    a (n, d) x b (m, d) -> (n, m)

Ce contrat n'est pas cosmetique : au J9, vectorstore.py compare 1 requete
a 3 000 chunks, et metrics.py compare 60 requetes a tout le corpus d'un
coup. Sans le mode matriciel, il faudrait une boucle Python — environ 100x
plus lent, parce que numpy delegue le calcul a du code compile (BLAS) alors
qu'une boucle repasse par l'interpreteur a chaque iteration.
"""
import numpy as np

# Evite la division par zero sur un vecteur nul (rare mais possible sur un
# chunk vide apres nettoyage). float32 : la precision machine est ~1.2e-7,
# donc 1e-12 est negligeable devant toute norme reelle.
_EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalisation L2 : ramene chaque vecteur a une longueur de 1.

    Une fois normalise, cosinus == produit scalaire. C'est la raison d'etre
    de cette fonction : elle transforme une division en une multiplication,
    et c'est exactement ce que fait FAISS en interne (J9).

    x : (d,) ou (n, d)
    retourne : meme forme, chaque ligne de norme 1
    """
    x = np.asarray(x, dtype=np.float32)
    # keepdims=True conserve la forme (n, 1) au lieu de (n,), ce qui permet
    # la division ligne par ligne par broadcasting.
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, _EPS)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarite cosinus : cos(a, b) = (a . b) / (||a|| * ||b||).

    Implementee comme un produit scalaire APRES normalisation — mathematiquement
    identique, mais numeriquement plus stable et reutilisable tel quel par
    l'index vectoriel.

    Valeurs : 1 = meme direction, 0 = sans rapport, -1 = opposees.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    # On memorise si l'entree etait un vecteur seul, pour restituer la bonne
    # forme en sortie (voir le contrat en tete de fichier).
    a_1d, b_1d = a.ndim == 1, b.ndim == 1

    A = normalize(np.atleast_2d(a))   # (n, d)
    B = normalize(np.atleast_2d(b))   # (m, d)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimensions incompatibles : {A.shape[1]} vs {B.shape[1]}")

    sims = A @ B.T                    # (n, m) — une seule multiplication matricielle

    if a_1d and b_1d:
        return float(sims[0, 0])
    if a_1d:
        return sims[0]                # (m,)
    if b_1d:
        return sims[:, 0]             # (n,)
    return sims                       # (n, m)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance euclidienne, pour comparer avec le cosinus.

    A quoi elle sert ici : montrer qu'elle donne un mauvais resultat sur du
    texte de longueurs inegales. Un resume et sa version detaillee pointent
    dans la meme direction (cosinus eleve) mais leurs vecteurs ont des normes
    tres differentes, donc la distance euclidienne les separe a tort.

    Note : sur des vecteurs DEJA normalises, les deux mesures deviennent
    equivalentes — dist^2 = 2 * (1 - cos). C'est pourquoi un index FAISS en
    distance L2 sur vecteurs normalises donne le meme classement qu'en cosinus.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_1d, b_1d = a.ndim == 1, b.ndim == 1

    A = np.atleast_2d(a)
    B = np.atleast_2d(b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimensions incompatibles : {A.shape[1]} vs {B.shape[1]}")

    # A[:, None, :] a la forme (n, 1, d), B[None, :, :] la forme (1, m, d).
    # Le broadcasting produit (n, m, d) sans ecrire une seule boucle.
    diff = A[:, None, :] - B[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))   # (n, m)

    if a_1d and b_1d:
        return float(dists[0, 0])
    if a_1d:
        return dists[0]
    if b_1d:
        return dists[:, 0]
    return dists


def brute_force_knn(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int = 5):
    """Recherche exhaustive des k plus proches voisins. Aucune librairie d'index.

    >>> C'EST LA VERITE TERRAIN DU PROJET. <<<
    Au J10, l'index HNSW sera compare a CE resultat : l'ecart est le
    "recall loss" de la recherche approximative. Sans cette baseline, aucune
    affirmation sur la qualite de l'index n'est demontrable.

    query_vec   : (d,) ou (n, d)
    corpus_vecs : (N, d)
    retourne    : (indices, scores), tries par score decroissant
                  formes (k,) si une seule requete, (n, k) sinon

    Pourquoi argpartition et pas argsort :
      argsort trie les N elements -> O(N log N).
      argpartition place juste les k meilleurs devant, sans les trier
      entre eux -> O(N). On ne trie ensuite que ces k elements.
      Sur 100 000 chunks pour k=5, c'est un ordre de grandeur de difference.
    """
    sims = cosine_similarity(query_vec, corpus_vecs)   # (N,) ou (n, N)
    single = sims.ndim == 1
    S = np.atleast_2d(sims)                            # (n, N)

    k = min(k, S.shape[1])
    # -S : argpartition trie par ordre croissant, on veut les scores les plus
    # eleves -> on partitionne sur l'oppose.
    part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]        # k meilleurs, non tries
    part_scores = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-part_scores, axis=1)                    # tri des k seulement
    idx = np.take_along_axis(part, order, axis=1)
    scores = np.take_along_axis(part_scores, order, axis=1)

    return (idx[0], scores[0]) if single else (idx, scores)
