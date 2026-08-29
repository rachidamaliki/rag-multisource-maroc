"""
Normalisation de l'arabe pour la recherche.

POURQUOI CE FICHIER EXISTE
--------------------------
Diagnostic mesure sur les PDF du Code du travail (voir data/raw/README.md) :
le mot `المقاولات` (les entreprises) ressort de l'extraction PDF sous la forme
`المقاوالت`. Visuellement identique, textuellement DIFFERENT.

Consequence si on ne corrige pas :
  - BM25 (recherche par mots exacts) ne trouvera jamais ce mot
  - l'embedding sera calcule sur une chaine erronee
  - aucune erreur ne sera levee : le recall baisse en silence

C'est le type de defaut qui rend un RAG arabe mediocre sans qu'on
comprenne pourquoi. Il se corrige en amont, une fois pour toutes.

LES 5 TRANSFORMATIONS
---------------------
1. Ligature lam-alef : `لا` est encodee dans le PDF comme deux caracteres
   separes (ل + ا) mal ordonnes. On recompose.
2. Alef : ا / أ / إ / آ sont la meme lettre a l'ecrit courant. L'utilisateur
   tape `الاجير`, le texte contient `الأجير` -> aucune correspondance.
3. Ya final : ي / ى interchangeables selon les conventions d'edition.
4. Ta marbuta : ة / ه confondues en fin de mot.
5. Diacritiques (harakat) : ُ ً ِ ... presentes dans les textes officiels,
   absentes des requetes utilisateur.

ATTENTION — pourquoi on garde le texte original
-----------------------------------------------
La normalisation sert a la RECHERCHE, pas a l'affichage. On indexe la version
normalisee mais on affiche la version d'origine : un texte de loi cite dans
une reponse doit etre exact au caractere pres. D'ou deux champs distincts sur
chaque chunk : `text` (affichage) et `text_norm` (recherche).
"""
from __future__ import annotations

import re
import unicodedata

# --- Plages Unicode ---
ARABIC_RANGE = ("؀", "ۿ")

# Diacritiques (harakat) : fatha, damma, kasra, sukun, shadda, tanwin...
DIACRITICS = re.compile(r"[ً-ْٰـ]")  # inclut le tatweel ـ

# Ligature lam-alef : le PDF produit parfois la forme decomposee ou la
# ligature de presentation (FEFB-FEFC etc.). On ramene tout a ل + ا.
LAM_ALEF = {
    "ﻻ": "لا",  # lam-alef
    "ﻼ": "لا",
    "ﻵ": "لآ",  # lam-alef madda
    "ﻶ": "لآ",
    "ﻷ": "لأ",  # lam-alef hamza above
    "ﻸ": "لأ",
    "ﻹ": "لإ",  # lam-alef hamza below
    "ﻺ": "لإ",
}

# --------------------------------------------------------------------
# CONFLATION lam-alef : la solution au probleme central du corpus arabe
# --------------------------------------------------------------------
# Constat mesure sur le PDF du ministere :
#     sequence CORRECTE   `لا` en milieu de mot :     79 occurrences
#     sequence CASSEE     `ال` en milieu de mot :  2 831 occurrences
#
# On ne peut PAS simplement inverser `ال` -> `لا` en milieu de mot : la
# sequence alef+lam existe legitimement dans de nombreux mots arabes
# (عالم "monde", سالم "sain", طالب "etudiant" — le schema فاعل est tres
# productif). L'inversion aveugle detruirait ces mots.
#
# LA SOLUTION : ne pas chercher a corriger, mais a UNIFIER.
# On applique la meme regle au corpus ET a la requete : en milieu de mot,
# `لا` devient `ال`. Peu importe alors laquelle des deux formes est
# "correcte" — les deux convergent vers la meme chaine, donc elles se
# rencontrent a la recherche.
#
#     corpus  : المقاوالت  (casse)     -> المقاوالت
#     requete : المقاولات  (correct)   -> المقاوالت   => elles matchent
#
# Cout : quelques collisions rares (عالم et علام deviennent identiques).
# Benefice : tout le vocabulaire contenant lam-alef redevient trouvable.
# En normalisation pour la recherche, la COHERENCE prime sur l'exactitude.
# Regle retenue : REGROUPEMENT. Tout groupe contigu de alef et lam est
# reecrit sous forme canonique (tous les alef, puis tous les lam).
#     المقاولات (correct) -> المقاوالت
#     المقاوالت (casse)   -> المقاوالت   => identiques
# Proprietes : independante de l'ordre, idempotente, et l'article defini
# `ال` en debut de mot reste inchange (1 alef + 1 lam = deja canonique).
# Testee sur 7 paires corpus/requete : 7/7 convergent.
ALEF_LAM_GROUP = re.compile(r"[ال]{2,}")


def _canonical_alef_lam(m: re.Match) -> str:
    g = m.group(0)
    return "ا" * g.count("ا") + "ل" * g.count("ل")

EQUIVALENCES = {
    # alef sous toutes ses formes
    "أ": "ا",  # أ
    "إ": "ا",  # إ
    "آ": "ا",  # آ
    "ٱ": "ا",  # ٱ
    # ya
    "ى": "ي",  # ى -> ي
    # ta marbuta
    "ة": "ه",  # ة -> ه
    # hamza sur support
    "ؤ": "و",  # ؤ -> و
    "ئ": "ي",  # ئ -> ي
}

# Chiffres arabo-indiens -> chiffres latins (les references d'articles doivent
# etre comparables entre les versions FR et AR du meme texte)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩",
                              "0123456789")


def is_arabic_char(c: str) -> bool:
    return ARABIC_RANGE[0] <= c <= ARABIC_RANGE[1]


def arabic_ratio(text: str) -> float:
    """Proportion de caracteres arabes. Sert a detecter la langue (fr/ar)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_arabic_char(c)) / len(letters)


def fix_ligatures(text: str) -> str:
    """Recompose les ligatures lam-alef mal extraites du PDF."""
    for bad, good in LAM_ALEF.items():
        text = text.replace(bad, good)
    # Formes de presentation arabes (U+FB50-FEFF) -> forme canonique.
    # NFKC fait ce travail : il remplace chaque glyphe contextuel (initial,
    # median, final) par la lettre de base.
    text = unicodedata.normalize("NFKC", text)
    return text


def fix_broken_spacing(text: str) -> str:
    """Recolle les espaces parasites inseres a l'INTERIEUR d'un mot arabe.

    Le PDF place parfois un espace entre deux lettres du meme mot :
    `المق تضيات` au lieu de `المقتضيات`.

    HEURISTIQUE VOLONTAIREMENT CONSERVATRICE : on ne recolle que si le
    fragment de gauche fait UNE seule lettre. Une premiere version acceptait
    aussi les fragments de deux lettres ; mesuree sur le corpus, elle fusionnait
    des mots legitimes (`الصحفيون المهنيون` -> un seul mot), ce qui detruit les
    frontieres de mots dont BM25 a besoin.

    Une lettre arabe isolee entre deux espaces n'est presque jamais un mot :
    les mots arabes d'une lettre se limitent aux prefixes colles (و، ف، ب، ل، ك)
    qui, eux, ne devraient justement pas etre separes.

    ATTENTION : cette correction reste une HYPOTHESE. On ne sait pas encore si
    elle ameliore le recall. C'est exactement ce que le harnais d'evaluation
    devra trancher — d'ou le drapeau `fix_spacing` de normalize_for_search().
    """
    # fragment d'UNE lettre + espace + suite du mot (2 lettres ou plus)
    pattern = re.compile(r"(?<![؀-ۿ])([؀-ۿ]) ([؀-ۿ]{2,})")
    prev = None
    while prev != text:            # boucle : un mot peut etre coupe en 3
        prev = text
        text = pattern.sub(r"", text)
    return text


def normalize_arabic(text: str, aggressive: bool = True) -> str:
    """Normalisation complete pour l'INDEXATION (pas pour l'affichage).

    aggressive=True applique les equivalences de lettres (alef, ya, ta marbuta).
    Utiliser aggressive=False si l'on veut conserver l'orthographe exacte.
    """
    text = fix_ligatures(text)
    text = DIACRITICS.sub("", text)
    text = text.translate(ARABIC_DIGITS)
    if aggressive:
        for a, b in EQUIVALENCES.items():
            text = text.replace(a, b)
        # conflation lam-alef : corpus et requete convergent vers la meme forme
        text = ALEF_LAM_GROUP.sub(_canonical_alef_lam, text)
    return text


def normalize_for_search(text: str, fix_spacing: bool = True) -> str:
    """Pipeline complet applique au champ `text_norm` de chaque chunk.

    C'est CETTE version qui est embeddee et indexee par BM25.
    Le champ `text` d'origine reste intact pour l'affichage et les citations.

    fix_spacing : hypothese a valider par l'evaluation. Mettre a False pour
    mesurer le recall sans recollage et comparer — c'est une des lignes du
    tableau comparatif final.
    """
    if fix_spacing:
        text = fix_broken_spacing(text)
    text = normalize_arabic(text, aggressive=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
