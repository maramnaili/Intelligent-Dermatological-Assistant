"""
AI-S2-08 — Entity Extraction

Entities extracted:
    - product_names
    - ingredients
    - skin_types

Support:
    - French
    - English

Les noms de produits sont recherchés dans le catalogue fourni.
"""

import re

from src.recommendation.knowledge_graph import (
    ACTIVE_INGREDIENTS,
    INGREDIENT_ALIASES,
    normalize_ingredient,
)


# ============================================================
# SKIN TYPES
# ============================================================
#
# Connecteur tolérant entre "peau" et le type/adjectif, pour
# les phrases françaises naturelles où un verbe ou un adverbe
# s'intercale ("Ma peau EST TRÈS sèche", "peau qui SEMBLE
# grasse"...). Sans ça, seule la forme rigide "peau sèche" ou
# "peau très sèche" matche, et une phrase comme "Ma peau est
# très sèche" ne matche RIEN : "est" casse le \s+ contigu entre
# "peau" et "très"/"sèche", donc le skin_type n'est jamais
# extrait et le profil ne se met pas à jour.

_FR_CONNECTOR = (
    r"(?:\s+(?:est|semble|devient|para[iî]t|reste|a\s+l['’]air))?"
    r"(?:\s+(?:tr[eè]s|un\s+peu|plut[oô]t|assez|super|hyper))?"
)


SKIN_TYPE_PATTERNS = {

    "grasse": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+grasse\b",
        r"\boily\s+skin\b",
        r"\bskin\s+(?:is|looks|feels|seems)"
        r"(?:\s+(?:very|really|quite))?\s+oily\b",
        r"\boily\b",
    ],

    "sèche": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+s[eè]che\b",
        r"\bdry\s+skin\b",
        r"\bskin\s+(?:is|looks|feels|seems)"
        r"(?:\s+(?:very|really|quite))?\s+dry\b",
        r"\bdry\b",
    ],

    "mixte": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+mixte\b",
        r"\bcombination\s+skin\b",
        r"\bskin\s+(?:is|looks|feels|seems)"
        r"(?:\s+(?:very|really|quite))?\s+combination\b",
        r"\bcombination\b",
    ],

    "normale": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+normale\b",
        r"\bnormal\s+skin\b",
        r"\bskin\s+(?:is|looks|feels|seems)"
        r"(?:\s+(?:very|really|quite))?\s+normal\b",
        r"\bnormal\b",
    ],

    "sensible": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+sensible\b",
        r"\bsensitive\s+skin\b",
        r"\bskin\s+(?:is|looks|feels|seems)"
        r"(?:\s+(?:very|really|quite))?\s+sensitive\b",
        r"\bsensitive\b",
    ],

    "ultra-sensible": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+ultra[-\s]?sensible\b",
        r"\bultra[-\s]?sensitive\s+skin\b",
        r"\bultra[-\s]?sensitive\b",
    ],

    "acnéique": [
        r"\bpeau" + _FR_CONNECTOR + r"\s+acn[eé]ique\b",
        r"\bacne[-\s]?prone\s+skin\b",
        r"\bacne[-\s]?prone\b",
    ],
}


# ============================================================
# INGREDIENTS
# ============================================================

def _build_ingredient_patterns():
    """
    Construit le dictionnaire  motif recherché -> identifiant normalisé.

    INGREDIENT_ALIASES associe un ALIAS (clé) à un identifiant
    normalisé (valeur, une chaîne). Il faut donc parcourir les
    paires (alias, identifiant), et non traiter la valeur comme une
    liste d'alias : itérer sur une chaîne enregistrerait chacune de
    ses lettres comme un "ingrédient" (ex. "l'", "a", "c" déclenchant
    à tort hyaluronic_acid). C'est le bug corrigé ici.
    """

    patterns = {}

    # Identifiants officiels (avec et sans underscore)
    for ingredient in ACTIVE_INGREDIENTS:

        name = str(ingredient).lower()

        patterns[name] = ingredient
        patterns[name.replace("_", " ")] = ingredient

    # Alias (FR / EN, avec ou sans accents)
    for alias, canonical in INGREDIENT_ALIASES.items():

        patterns[str(alias).lower()] = canonical

    return patterns


INGREDIENT_PATTERNS = (
    _build_ingredient_patterns()
)


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_text(text):

    text = text.lower()

    text = (
        text
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )

    return text


# ============================================================
# SKIN TYPE EXTRACTION
# ============================================================

def _extract_skin_types(text):

    normalized = _normalize_text(
        text
    )

    found = []

    # Les expressions les plus longues
    # sont recherchées en premier.
    ordered_patterns = []

    for canonical, patterns in SKIN_TYPE_PATTERNS.items():

        for pattern in patterns:

            ordered_patterns.append(
                (
                    canonical,
                    pattern,
                    len(pattern),
                )
            )

    ordered_patterns.sort(
        key=lambda x: x[2],
        reverse=True
    )

    for canonical, pattern, _ in ordered_patterns:

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):

            if canonical not in found:
                found.append(
                    canonical
                )

    return found


# ============================================================
# INGREDIENT EXTRACTION
# ============================================================

def _extract_ingredients(text):

    normalized = _normalize_text(
        text
    )

    found = []

    for alias, canonical in sorted(
        INGREDIENT_PATTERNS.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        pattern = (
            r"(?<!\w)"
            + re.escape(alias)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):

            normalized_ingredient = (
                normalize_ingredient(
                    canonical
                )
            )

            if (
                normalized_ingredient
                and normalized_ingredient
                not in found
            ):
                found.append(
                    normalized_ingredient
                )

    return found


# ============================================================
# PRODUCT NAME EXTRACTION
# ============================================================

def _extract_product_names(
    text,
    known_products=None
):

    if not known_products:
        return []

    normalized_text = _normalize_text(
        text
    )

    found = []

    # Produits les plus longs en premier
    products = sorted(
        known_products,
        key=lambda x: len(str(x)),
        reverse=True
    )

    for product in products:

        if not product:
            continue

        product_normalized = (
            _normalize_text(
                str(product)
            )
        )

        if product_normalized in normalized_text:

            if product not in found:
                found.append(
                    product
                )

    return found


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def extract_entities(
    text,
    known_products=None
):

    if not isinstance(text, str):
        raise ValueError(
            "text doit être une chaîne."
        )

    return {

        "product_names":
            _extract_product_names(
                text,
                known_products
            ),

        "ingredients":
            _extract_ingredients(
                text
            ),

        "skin_types":
            _extract_skin_types(
                text
            ),
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    tests = [

        "J'ai la peau grasse.",

        "I have oily skin.",

        "I have dry and sensitive skin.",

        "J'ai la peau sèche et sensible.",

        "I need a product with niacinamide.",

        "Je cherche un produit avec de la niacinamide.",

        "I have acne-prone skin.",

        "J'ai la peau grasse avec de l'acné.",
    ]

    for text in tests:

        print()
        print(
            "TEXT:",
            text
        )

        print(
            extract_entities(text)
        )