"""
Nettoyage et enrichissement du dataset skincare.

Entrée :
    data/raw/products/parafendri_products_raw.csv

Sortie :
    data/processed/parafendri_products_clean.csv
"""

from pathlib import Path
import argparse
import re
import unicodedata

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    BASE_DIR
    / "data"
    / "raw"
    / "products"
    / "parafendri_products_raw.csv"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "parafendri_products_clean.csv"
)


# ============================================================
# NORMALISATION
# ============================================================

def normalize(text):
    """
    Normalise un texte :
    - minuscules
    - suppression des accents
    - espaces multiples
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"\s+", " ", text)

    return text


def clean_text(text):
    """
    Nettoyage général d'un texte.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def contains_keyword(text_norm, keyword):
    """
    Vérifie la présence d'un mot-clé en bloquant les collisions à
    l'intérieur d'un autre mot (ex: "uv" trouvé dans "jouvence", "pha"
    trouvé dans "pharmaceris" à cause de "PHA" l'acide), SANS bloquer les
    compositions courantes du catalogue où un chiffre suit directement le
    mot-clé (ex: "spf50", "uv30"). On interdit donc une lettre
    immédiatement avant/après le mot-clé, mais on autorise un chiffre.
    """
    pattern = r"(?<![a-z])" + re.escape(keyword) + r"(?![a-z])"
    return re.search(pattern, text_norm) is not None


def any_keyword(text_norm, keywords):
    return any(contains_keyword(text_norm, k) for k in keywords)


# ============================================================
# DESCRIPTION
# ============================================================

def clean_description(text):
    """
    Nettoie la description récupérée par le scraping.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Suppression de quelques éléments fréquents du scraping
    text = re.sub(
        r"(?i)ajouter au panier.*",
        "",
        text
    )

    text = re.sub(
        r"(?i)référence\s*:?.*",
        "",
        text
    )

    text = re.sub(
        r"(?i)disponibilité\s*:?.*",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# INGREDIENTS
# ============================================================

def extract_ingredients_from_text(text):
    """
    Extrait la partie ingrédients depuis une description.

    Le dataset contient plusieurs formats possibles :
        - Composition:
        - Composition (INCI):
        - Liste complète d'ingrédients:
        - Ingrédients:
        - Ingredients:
        - Ingrédients clés:

    On essaie en priorité de récupérer la liste INCI complète.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # 1. Liste complète d'ingrédients
    # --------------------------------------------------------

    patterns_complete = [
        r"(?is)liste\s+complete\s+d['’]ingredients?\s*:?\s*(.*)",
        r"(?is)liste\s+complete\s+des\s+ingredients?\s*:?\s*(.*)",
        r"(?is)full\s+ingredients?\s*:?\s*(.*)",
        r"(?is)full\s+ingredient\s+list\s*:?\s*(.*)",
    ]

    for pattern in patterns_complete:
        match = re.search(pattern, text)

        if match:
            result = match.group(1)

            # On coupe si une nouvelle section apparaît
            result = re.split(
                r"(?i)\b(?:conseils?|mode d'emploi|utilisation|"
                r"precautions?|avantages?|proprietes?|description)\b",
                result
            )[0]

            result = clean_text(result)

            if result:
                return result

    # --------------------------------------------------------
    # 2. Composition / INCI
    # --------------------------------------------------------

    patterns_composition = [
        r"(?is)composition\s*\(?\s*inci\s*\)?\s*:?\s*(.*)",
        r"(?is)composition\s*:?\s*(.*)",
        r"(?is)ingredients?\s*\(?\s*inci\s*\)?\s*:?\s*(.*)",
        r"(?is)ingredients?\s*:?\s*(.*)",
        r"(?is)ingr[eé]dients?\s*:?\s*(.*)",
    ]

    for pattern in patterns_composition:
        match = re.search(pattern, text)

        if match:
            result = match.group(1)

            result = re.split(
                r"(?i)\b(?:conseils?|mode d'emploi|utilisation|"
                r"precautions?|avantages?|proprietes?|description|"
                r"benefices?)\b",
                result
            )[0]

            result = clean_text(result)

            if result:
                return result

    return ""


def get_ingredients_clean(row):
    """
    Détermine la meilleure source disponible pour les ingrédients.

    Priorité :
        1. colonne ingredients
        2. detail_description
        3. description
    """

    # --------------------------------------------------------
    # 1. Colonne ingredients
    # --------------------------------------------------------

    ingredients = row.get("ingredients", "")

    if not pd.isna(ingredients):
        ingredients = str(ingredients).strip()

        if ingredients and ingredients.lower() not in {
            "nan",
            "none",
            "null"
        }:
            return clean_text(ingredients)

    # --------------------------------------------------------
    # 2. detail_description
    # --------------------------------------------------------

    detail_description = row.get("detail_description", "")

    extracted = extract_ingredients_from_text(
        detail_description
    )

    if extracted:
        return extracted

    # --------------------------------------------------------
    # 3. description
    # --------------------------------------------------------

    description = row.get("description", "")

    extracted = extract_ingredients_from_text(
        description
    )

    if extracted:
        return extracted

    return ""


# ============================================================
# CATEGORY
# ============================================================

def parse_category(category_url):
    """
    Transforme category_url en catégorie lisible.
    """

    if pd.isna(category_url):
        return ""

    url = str(category_url).strip()

    if not url:
        return ""

    slug = url.rstrip("/").split("/")[-1]

    slug = slug.lower()

    # Nettoyage des éventuels identifiants
    slug = re.sub(r"^\d+-", "", slug)

    slug = slug.replace("_", "-")

    return slug


# ============================================================
# SKIN TYPE
# ============================================================

CATEGORY_MAP = {
    "peaux-grasses": "grasse",
    "peaux-seches": "seche",
    "peaux-mixtes": "mixte",
    "peaux-sensibles": "sensible",
    "peaux-sensibles-": "sensible",
}


def derive_skin_type(category):
    """
    Détermine le type de peau uniquement lorsque
    la catégorie permet de le faire clairement.

    IMPORTANT :
    On laisse None lorsqu'il n'est pas possible
    de déterminer le type de peau.
    """

    if not category:
        return None

    category_norm = normalize(category)

    for key, value in CATEGORY_MAP.items():
        if normalize(key) == category_norm:
            return value

    return None


# ============================================================
# CONCERNS
# ============================================================

CONCERN_KEYWORDS = {
    "acne": [
        "acne",
        "acneique",
        "imperfections",
        "boutons",
        "points noirs",
        "comedons",
        "anti-imperfections",
    ],

    "taches": [
        "taches",
        "tache",
        "pigmentation",
        "depigmentant",
        "depigmentation",
        "eclaircissant",
        "eclat",
        "teint",
    ],

    "deshydratation": [
        "deshydratation",
        "deshydrate",
        "hydratation",
        "repulpant",
        "hydratant",
    ],

    "secheresse": [
        "peau seche",
        "peaux seches",
        "secheresse",
    ],

    "rougeurs": [
        "rougeurs",
        "rougeur",
        "rosacee",
        "couperose",
        "apaise",
        "apaisant",
    ],

    "rides": [
        "rides",
        "ridules",
        "anti-age",
        "antirides",
        "fermete",
    ],

    "pores": [
        "pores",
        "pores dilates",
    ],

    "sensibilite": [
        "peau sensible",
        "peaux sensibles",
        "sensible",
    ],
}


def derive_concerns(text):
    """
    Détecte les préoccupations à partir du texte produit.
    """

    text_norm = normalize(text)

    concerns = []

    for concern, keywords in CONCERN_KEYWORDS.items():

        for keyword in keywords:

            if normalize(keyword) in text_norm:
                concerns.append(concern)
                break

    return concerns


# ============================================================
# ROUTINE STEP — strictement les 7 étapes du cahier des charges (§10)
#
#   Matin : nettoyage, serum, hydratant, protection_solaire
#   Soir  : demaquillage, nettoyage, traitement, reparation
#
# IMPORTANT : le matching se fait sur product_name UNIQUEMENT, jamais sur
# category ni sur la description. La catégorie "apres-solaire" du site
# mélange en réalité des écrans solaires ET des soins réparateurs -> 
# l'utiliser comme texte de matching pollue le résultat quel que soit
# l'ordre des règles (vérifié sur les données réelles).
#
# peeling et masque sont repliés dans "traitement" (ce sont des soins
# traitants, pas des étapes AM/PM prévues séparément par le cahier des
# charges).
# apres_soleil est replié dans "reparation" (réparer la peau après
# exposition solaire correspond exactement au créneau "Réparation" du
# soir) et vérifié AVANT protection_solaire pour éviter toute collision.
# ============================================================

def derive_routine_step(product_name):
    """
    Détermine l'étape de routine à partir du NOM du produit uniquement.
    """

    text_norm = normalize(product_name)

    # 1. démaquillage
    if any_keyword(text_norm, ["demaquillant"]):
        return "demaquillage"

    # 2. après-soleil -> réparation (repli cahier des charges).
    # "apres-soleil" (tiret) et "apres soleil" (espace) apparaissent
    # tous les deux dans les noms produits du catalogue.
    if any_keyword(text_norm, ["apres soleil", "apres-soleil", "after sun"]):
        return "reparation"

    # 3. protection solaire (mots composés uniquement, jamais "solaire"
    # seul, pour ne pas capturer "apres-solaire" ni d'autres faux positifs)
    if any_keyword(text_norm, [
        "spf", "ecran solaire", "creme solaire", "protection solaire",
        "sun secure",
    ]) or contains_keyword(text_norm, "uv"):
        return "protection_solaire"

    # 4. nettoyage
    if any_keyword(text_norm, [
        "eau micellaire", "nettoyant", "gel nettoyant", "mousse nettoyante",
        "mousse lavante", "lotion nettoyante", "huile lavante", "savon",
        "gel lavant",
    ]):
        return "nettoyage"

    # 5. sérum
    if any_keyword(text_norm, ["serum", "ampoule", "concentre"]):
        return "serum"

    # 6. traitement (peeling + masque repliés ici)
    if any_keyword(text_norm, [
        "anti-acne", "anti-imperfections", "correcteur", "traitement",
        "gel correcteur", "peeling", "exfoliant", "gommage",
        "acide glycolique", "acide lactique", "acide mandelique",
        "aha", "bha", "pha", "masque",
    ]):
        return "traitement"

    # 7. hydratant
    if any_keyword(text_norm, [
        "creme hydratante", "creme visage", "gel creme", "soin hydratant",
        "lait hydratant", "hydratant",
    ]):
        return "hydratant"

    # 8. réparation
    if any_keyword(text_norm, [
        "reparateur", "reparation", "baume reparateur", "cica",
        "barriere cutanee",
    ]):
        return "reparation"

    return None


# ============================================================
# ACTIVE INGREDIENTS
# ============================================================

ACTIVE_INGREDIENTS = {

    "acide_hyaluronique": [
        "acide hyaluronique",
        "hyaluronic acid",
        "sodium hyaluronate",
    ],

    "glycerine": [
        "glycerin",
        "glycerine",
    ],

    "uree": [
        "urea",
        "uree",
    ],

    "beurre_de_karite": [
        "shea butter",
        "beurre de karite",
        "butyrospermum parkii",
    ],

    "ceramides": [
        "ceramide",
        "ceramides",
    ],

    "squalane": [
        "squalane",
        "squalene",
    ],

    "niacinamide": [
        "niacinamide",
        "nicotinamide",
    ],

    "panthenol": [
        "panthenol",
        "provitamin b5",
    ],

    "aloe_vera": [
        "aloe vera",
        "aloe barbadensis",
    ],

    "acide_salicylique": [
        "acide salicylique",
        "salicylic acid",
    ],

    "acide_lactique": [
        "acide lactique",
        "lactic acid",
    ],

    "acide_glycolique": [
        "acide glycolique",
        "glycolic acid",
    ],

    "acide_mandelique": [
        "acide mandelique",
        "mandelic acid",
    ],

    "acide_lactobionique": [
        "acide lactobionique",
        "lactobionic acid",
    ],

    "acide_azelaique": [
        "acide azelaique",
        "azelaic acid",
    ],

    "benzoyl_peroxide": [
        "benzoyl peroxide",
        "peroxyde de benzoyle",
    ],

    "zinc": [
        "zinc",
        "zinc pca",
        "zinc gluconate",
    ],

    "retinol": [
        "retinol",
        "retinal",
        "retinaldehyde",
    ],

    "soufre": [
        "sulfur",
        "sulphur",
        "soufre",
    ],

    "vitamine_c": [
        "vitamin c",
        "vitamine c",
        "ascorbic acid",
        "ascorbyl glucoside",
        "sodium ascorbyl phosphate",
        "magnesium ascorbyl phosphate",
    ],

    "vitamine_e": [
        "vitamin e",
        "tocopherol",
        "tocopheryl acetate",
    ],
}


def detect_active_ingredients(ingredients_text):
    """
    Détecte les actifs UNIQUEMENT dans la liste d'ingrédients.

    Retourne une vraie liste Python.
    """

    if not ingredients_text:
        return []

    text_norm = normalize(ingredients_text)

    detected = []

    for active, keywords in ACTIVE_INGREDIENTS.items():

        for keyword in keywords:

            keyword_norm = normalize(keyword)

            if keyword_norm in text_norm:
                detected.append(active)
                break

    return detected


# ============================================================
# NON-SKINCARE
# ============================================================

NON_SKINCARE_KEYWORDS = [
    "shampooing",
    "shampoing",
    "cheveux",
    "apres shampoing",
    "apres-shampoing",
    "gel douche",
    "corps",
    "deodorant",
    "parfum",
    "dentifrice",
    "brosse a dents",
    "hygiene bucco",
    "coffret",
    "pack",
    "duo",
    "trousse",

    # Maquillage
    "fond de teint",
    "foundation",
    "blush",
    "bronzer",
    "highlighter",
    "mascara",
    "eyeliner",
    "fard a paupieres",
    "rouge a levres",
    "gloss",
    "poudre maquillage",
]


def is_non_skincare(text):
    """
    Détecte les produits clairement hors skincare visage.
    """

    text_norm = normalize(text)

    for keyword in NON_SKINCARE_KEYWORDS:

        if normalize(keyword) in text_norm:
            return True

    return False


# ============================================================
# CLEAN DATASET
# ============================================================

def build_clean_dataset(df):
    """
    Construit le dataset nettoyé.
    """

    out = pd.DataFrame()

    # --------------------------------------------------------
    # Colonnes de base
    # --------------------------------------------------------

    out["product_name"] = (
        df["product_name"]
        .fillna("")
        .apply(clean_text)
    )

    out["brand"] = (
        df["brand_name"]
        .fillna("")
        .apply(clean_text)
    )

    out["price_tnd"] = df["price_tnd"]

    out["availability"] = (
        df["availability"]
        .fillna("")
        .apply(clean_text)
    )

    out["image_url"] = (
        df["image_url"]
        .fillna("")
        .apply(clean_text)
    )

    out["product_url"] = (
        df["product_url"]
        .fillna("")
        .apply(clean_text)
    )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description_clean = (
        df["short_description"]
        .fillna("")
        .apply(clean_description)
    )

    detail_description_clean = (
        df["detail_description"]
        .fillna("")
        .apply(clean_description)
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    out["category"] = (
        df["category_url"]
        .apply(parse_category)
    )

    # --------------------------------------------------------
    # Ingredients
    # --------------------------------------------------------

    out["ingredients_clean"] = df.apply(
        get_ingredients_clean,
        axis=1
    )

    out["has_ingredients_data"] = (
        out["ingredients_clean"]
        .fillna("")
        .str.strip()
        .ne("")
    )

    # --------------------------------------------------------
    # Skin type
    # --------------------------------------------------------

    # IMPORTANT :
    # On garde None lorsque le type de peau ne peut pas être déterminé.
    out["skin_type"] = (
        out["category"]
        .apply(derive_skin_type)
    )

    # --------------------------------------------------------
    # Concerns
    # --------------------------------------------------------
    # (le texte de matching des concerns reste large : nom + descriptions
    # + catégorie, car contrairement à routine_step, on n'a pas observé de
    # collision problématique ici)

    concern_text = (
        out["product_name"].fillna("")
        + " "
        + description_clean.fillna("")
        + " "
        + detail_description_clean.fillna("")
        + " "
        + out["category"].fillna("")
    )

    out["concerns"] = concern_text.apply(
        derive_concerns
    )

    # --------------------------------------------------------
    # Routine step
    # --------------------------------------------------------
    # UNIQUEMENT sur product_name (voir commentaire au-dessus de
    # derive_routine_step pour la justification).

    out["routine_step"] = out["product_name"].apply(
        derive_routine_step
    )

    # --------------------------------------------------------
    # Active ingredients
    # --------------------------------------------------------

    # IMPORTANT :
    # On cherche les actifs UNIQUEMENT dans
    # ingredients_clean.
    out["active_ingredients"] = (
        out["ingredients_clean"]
        .fillna("")
        .apply(normalize)
        .apply(detect_active_ingredients)
    )

    # --------------------------------------------------------
    # Non skincare
    # --------------------------------------------------------

    non_skincare_text = (
        out["product_name"].fillna("")
        + " "
        + description_clean.fillna("")
        + " "
        + out["category"].fillna("")
    )

    out["is_non_skincare"] = (
        non_skincare_text
        .apply(is_non_skincare)
    )

    return out


# ============================================================
# RAPPORT
# ============================================================

def print_report(df):
    """
    Affiche un rapport de qualité du dataset.
    """

    print()
    print("=" * 60)
    print("RAPPORT DE NETTOYAGE")
    print("=" * 60)

    print(
        f"Total produits            : {len(df)}"
    )

    print(
        "routine_step manquant     : "
        f"{df['routine_step'].isna().sum()}"
    )

    print(
        "skin_type manquant        : "
        f"{df['skin_type'].isna().sum()} "
        "(laissé volontairement vide si inconnu)"
    )

    print(
        "has_ingredients_data=False: "
        f"{(~df['has_ingredients_data']).sum()}"
    )

    print(
        "is_non_skincare=True      : "
        f"{df['is_non_skincare'].sum()}"
    )

    active_empty = (
        df["active_ingredients"]
        .apply(lambda x: len(x) == 0)
        .sum()
    )

    print(
        "active_ingredients vide   : "
        f"{active_empty}"
    )

    print()
    print("Répartition routine_step:")

    print(
        df["routine_step"]
        .value_counts(dropna=False)
    )

    print()
    print("Répartition skin_type:")

    print(
        df["skin_type"]
        .value_counts(dropna=False)
    )

    print()
    print("Top actifs détectés:")

    active_counts = {}

    for values in df["active_ingredients"]:

        for active in values:
            active_counts[active] = (
                active_counts.get(active, 0) + 1
            )

    if active_counts:

        active_series = pd.Series(
            active_counts
        ).sort_values(
            ascending=False
        )

        print(active_series)

    else:
        print("Aucun actif détecté.")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Vérification input
    # --------------------------------------------------------

    if not args.input.exists():

        raise FileNotFoundError(
            f"\nFichier d'entrée introuvable :\n"
            f"{args.input}\n"
        )

    # --------------------------------------------------------
    # Création dossier output
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(f"Chargement du dataset : {args.input}")

    # --------------------------------------------------------
    # Lecture CSV
    # --------------------------------------------------------

    df_raw = pd.read_csv(
        args.input
    )

    print(
        f"Produits chargés : {len(df_raw)}"
    )

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    df_clean = build_clean_dataset(
        df_raw
    )

    # --------------------------------------------------------
    # Rapport
    # --------------------------------------------------------

    print_report(
        df_clean
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    # Les listes Python sont converties en texte CSV.
  
    df_clean.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"✅ Dataset nettoyé exporté vers :\n"
        f"{args.output}"
    )


if __name__ == "__main__":
    main()