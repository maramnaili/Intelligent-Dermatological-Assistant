"""
Knowledge Graph Dermatologique

Base de connaissances dermatologiques utilisée par le système
de recommandation et d'explication.

Actifs :
    - Rétinol
    - Acide salicylique
    - Acide glycolique
    - Niacinamide
    - Acide azélaïque
    - Vitamine C
    - Céramides
    - Panthénol
    - Acide hyaluronique

Relations :
    - Compatibilités
    - Synergies
    - Contre-indications
    - Interactions

Ce fichier contient à la fois :
    1. les connaissances du graphe ;
    2. les fonctions permettant de les interroger.
"""


# ============================================================
# 1. NOMS NORMALISÉS DES ACTIFS
# ============================================================

RETINOL = "retinol"
SALICYLIC_ACID = "salicylic_acid"
GLYCOLIC_ACID = "glycolic_acid"
NIACINAMIDE = "niacinamide"
AZELAIC_ACID = "azelaic_acid"
VITAMIN_C = "vitamin_c"
CERAMIDES = "ceramides"
PANTHENOL = "panthenol"
HYALURONIC_ACID = "hyaluronic_acid"


ACTIVE_INGREDIENTS = {
    RETINOL,
    SALICYLIC_ACID,
    GLYCOLIC_ACID,
    NIACINAMIDE,
    AZELAIC_ACID,
    VITAMIN_C,
    CERAMIDES,
    PANTHENOL,
    HYALURONIC_ACID,
}


# ============================================================
# 2. KNOWLEDGE GRAPH
# ============================================================

KNOWLEDGE_GRAPH = {

    RETINOL: {
        "compatible_with": {
            NIACINAMIDE,
            CERAMIDES,
            PANTHENOL,
            HYALURONIC_ACID,
        },

        "synergies": {
            NIACINAMIDE,
            CERAMIDES,
            HYALURONIC_ACID,
        },

        "contraindications": {
            "pregnancy",
        },

        "interactions": {
            GLYCOLIC_ACID,
            SALICYLIC_ACID,
            VITAMIN_C,
        },
    },


    SALICYLIC_ACID: {
        "compatible_with": {
            NIACINAMIDE,
            HYALURONIC_ACID,
            CERAMIDES,
            PANTHENOL,
        },

        "synergies": {
            NIACINAMIDE,
        },

        "contraindications": {
            "very_sensitive_skin",
        },

        "interactions": {
            RETINOL,
            GLYCOLIC_ACID,
        },
    },


    GLYCOLIC_ACID: {
        "compatible_with": {
            NIACINAMIDE,
            HYALURONIC_ACID,
            CERAMIDES,
            PANTHENOL,
        },

        "synergies": {
            NIACINAMIDE,
            HYALURONIC_ACID,
        },

        "contraindications": {
            "very_sensitive_skin",
            "rosacea",
        },

        "interactions": {
            RETINOL,
            SALICYLIC_ACID,
        },
    },


    NIACINAMIDE: {
        "compatible_with": {
            RETINOL,
            SALICYLIC_ACID,
            GLYCOLIC_ACID,
            AZELAIC_ACID,
            VITAMIN_C,
            CERAMIDES,
            PANTHENOL,
            HYALURONIC_ACID,
        },

        "synergies": {
            RETINOL,
            SALICYLIC_ACID,
            AZELAIC_ACID,
            CERAMIDES,
        },

        "contraindications": set(),

        "interactions": set(),
    },


    AZELAIC_ACID: {
        "compatible_with": {
            NIACINAMIDE,
            HYALURONIC_ACID,
            CERAMIDES,
            PANTHENOL,
        },

        "synergies": {
            NIACINAMIDE,
        },

        "contraindications": set(),

        "interactions": {
            RETINOL,
        },
    },


    VITAMIN_C: {
        "compatible_with": {
            NIACINAMIDE,
            HYALURONIC_ACID,
            CERAMIDES,
            PANTHENOL,
        },

        "synergies": {
            HYALURONIC_ACID,
            VITAMIN_C,
        },

        "contraindications": {
            "very_sensitive_skin",
        },

        "interactions": {
            RETINOL,
        },
    },


    CERAMIDES: {
        "compatible_with": {
            RETINOL,
            SALICYLIC_ACID,
            GLYCOLIC_ACID,
            NIACINAMIDE,
            AZELAIC_ACID,
            VITAMIN_C,
            PANTHENOL,
            HYALURONIC_ACID,
        },

        "synergies": {
            RETINOL,
            NIACINAMIDE,
            PANTHENOL,
            HYALURONIC_ACID,
        },

        "contraindications": set(),

        "interactions": set(),
    },


    PANTHENOL: {
        "compatible_with": {
            RETINOL,
            SALICYLIC_ACID,
            GLYCOLIC_ACID,
            NIACINAMIDE,
            AZELAIC_ACID,
            VITAMIN_C,
            CERAMIDES,
            HYALURONIC_ACID,
        },

        "synergies": {
            RETINOL,
            CERAMIDES,
            HYALURONIC_ACID,
        },

        "contraindications": set(),

        "interactions": set(),
    },


    HYALURONIC_ACID: {
        "compatible_with": {
            RETINOL,
            SALICYLIC_ACID,
            GLYCOLIC_ACID,
            NIACINAMIDE,
            AZELAIC_ACID,
            VITAMIN_C,
            CERAMIDES,
            PANTHENOL,
        },

        "synergies": {
            RETINOL,
            VITAMIN_C,
            CERAMIDES,
            PANTHENOL,
        },

        "contraindications": set(),

        "interactions": set(),
    },
}


# ============================================================
# 3. NORMALISATION DES NOMS
# ============================================================

INGREDIENT_ALIASES = {

    "retinol": RETINOL,
    "rétinol": RETINOL,

    "salicylic acid": SALICYLIC_ACID,
    "acide salicylique": SALICYLIC_ACID,

    "glycolic acid": GLYCOLIC_ACID,
    "acide glycolique": GLYCOLIC_ACID,

    "niacinamide": NIACINAMIDE,

    "azelaic acid": AZELAIC_ACID,
    "acide azélaïque": AZELAIC_ACID,
    "acide azelaique": AZELAIC_ACID,

    "vitamin c": VITAMIN_C,
    "vitamine c": VITAMIN_C,

    "ceramides": CERAMIDES,
    "céramides": CERAMIDES,

    "panthenol": PANTHENOL,
    "panthénol": PANTHENOL,

    "hyaluronic acid": HYALURONIC_ACID,
    "acide hyaluronique": HYALURONIC_ACID,
}


def normalize_ingredient(ingredient):
    """
    Convertit le nom d'un actif vers son identifiant normalisé.

    Gère deux formats en entrée :
        - texte libre avec espaces   : "acide salicylique"
        - catalogue produit (underscores) : "acide_salicylique"

    Le catalogue RenYou Shop stocke les actifs composés avec des
    underscores (colonne active_ingredients de products_metadata),
    alors que INGREDIENT_ALIASES utilise des espaces. Sans ce
    fallback, la quasi-totalité des actifs composés (acide
    salicylique, acide hyaluronique, acide azélaïque, vitamine C...)
    ne matchaient jamais le Knowledge Graph.
    """

    if ingredient is None:
        return None

    value = str(ingredient).strip().lower()

    normalized = INGREDIENT_ALIASES.get(value)

    if normalized is not None:
        return normalized

    return INGREDIENT_ALIASES.get(value.replace("_", " "))


# ============================================================
# 4. VERIFICATION D'EXISTENCE
# ============================================================

def is_known_ingredient(ingredient):
    """
    Retourne True si l'actif appartient au Knowledge Graph.
    """

    normalized = normalize_ingredient(ingredient)

    return normalized in ACTIVE_INGREDIENTS


# ============================================================
# 5. RECUPERATION DES RELATIONS
# ============================================================

def get_compatible_ingredients(ingredient):
    """
    Retourne les actifs compatibles avec l'actif donné.
    """

    normalized = normalize_ingredient(ingredient)

    if normalized not in KNOWLEDGE_GRAPH:
        return set()

    return KNOWLEDGE_GRAPH[
        normalized
    ]["compatible_with"]


def get_synergies(ingredient):
    """
    Retourne les actifs présentant une synergie
    avec l'actif donné.
    """

    normalized = normalize_ingredient(ingredient)

    if normalized not in KNOWLEDGE_GRAPH:
        return set()

    return KNOWLEDGE_GRAPH[
        normalized
    ]["synergies"]


def get_contraindications(ingredient):
    """
    Retourne les situations dans lesquelles
    l'actif est contre-indiqué selon le graphe.
    """

    normalized = normalize_ingredient(ingredient)

    if normalized not in KNOWLEDGE_GRAPH:
        return set()

    return KNOWLEDGE_GRAPH[
        normalized
    ]["contraindications"]


def get_interactions(ingredient):
    """
    Retourne les actifs présentant une interaction
    avec l'actif donné.
    """

    normalized = normalize_ingredient(ingredient)

    if normalized not in KNOWLEDGE_GRAPH:
        return set()

    return KNOWLEDGE_GRAPH[
        normalized
    ]["interactions"]


# ============================================================
# 6. VERIFICATION D'UNE RELATION ENTRE DEUX ACTIFS
# ============================================================

def are_compatible(ingredient_a, ingredient_b):
    """
    Vérifie si deux actifs sont compatibles.
    """

    a = normalize_ingredient(ingredient_a)
    b = normalize_ingredient(ingredient_b)

    if a not in ACTIVE_INGREDIENTS:
        return False

    if b not in ACTIVE_INGREDIENTS:
        return False

    return b in KNOWLEDGE_GRAPH[a]["compatible_with"]


def have_synergy(ingredient_a, ingredient_b):
    """
    Vérifie si deux actifs présentent une synergie.
    """

    a = normalize_ingredient(ingredient_a)
    b = normalize_ingredient(ingredient_b)

    if a not in ACTIVE_INGREDIENTS:
        return False

    if b not in ACTIVE_INGREDIENTS:
        return False

    return b in KNOWLEDGE_GRAPH[a]["synergies"]


def have_interaction(ingredient_a, ingredient_b):
    """
    Vérifie si deux actifs présentent une interaction.
    """

    a = normalize_ingredient(ingredient_a)
    b = normalize_ingredient(ingredient_b)

    if a not in ACTIVE_INGREDIENTS:
        return False

    if b not in ACTIVE_INGREDIENTS:
        return False

    return (
        b in KNOWLEDGE_GRAPH[a]["interactions"]
        or
        a in KNOWLEDGE_GRAPH[b]["interactions"]
    )


# ============================================================
# 7. VERIFICATION DES CONTRE-INDICATIONS
# ============================================================

def is_contraindicated(ingredient, condition):
    """
    Vérifie si un actif est contre-indiqué
    dans une situation donnée.

    Exemples :
        pregnancy
        very_sensitive_skin
        rosacea
    """

    normalized = normalize_ingredient(ingredient)

    if normalized not in KNOWLEDGE_GRAPH:
        return False

    condition = str(condition).strip().lower()

    return condition in KNOWLEDGE_GRAPH[
        normalized
    ]["contraindications"]


# ============================================================
# 8. ANALYSE D'UN ENSEMBLE D'ACTIFS
# ============================================================

def analyze_ingredients(ingredients):
    """
    Analyse une liste d'actifs et retourne les relations
    trouvées entre eux.

    Retourne :
        {
            "compatible": [...],
            "synergies": [...],
            "interactions": [...]
        }
    """

    normalized_ingredients = []

    for ingredient in ingredients:

        normalized = normalize_ingredient(
            ingredient
        )

        if normalized and normalized not in normalized_ingredients:
            normalized_ingredients.append(
                normalized
            )

    compatible = []
    synergies = []
    interactions = []

    for i in range(
        len(normalized_ingredients)
    ):

        for j in range(
            i + 1,
            len(normalized_ingredients)
        ):

            ingredient_a = normalized_ingredients[i]
            ingredient_b = normalized_ingredients[j]

            if are_compatible(
                ingredient_a,
                ingredient_b
            ):
                compatible.append(
                    (
                        ingredient_a,
                        ingredient_b
                    )
                )

            if have_synergy(
                ingredient_a,
                ingredient_b
            ):
                synergies.append(
                    (
                        ingredient_a,
                        ingredient_b
                    )
                )

            if have_interaction(
                ingredient_a,
                ingredient_b
            ):
                interactions.append(
                    (
                        ingredient_a,
                        ingredient_b
                    )
                )

    return {
        "compatible": compatible,
        "synergies": synergies,
        "interactions": interactions,
    }


# ============================================================
# 9. TEST RAPIDE DU MODULE
# ============================================================

if __name__ == "__main__":

    print("Knowledge Graph Dermatologique")
    print("=" * 40)

    print(
        f"Nombre d'actifs : "
        f"{len(ACTIVE_INGREDIENTS)}"
    )

    print("\nActifs :")

    for ingredient in sorted(
        ACTIVE_INGREDIENTS
    ):
        print(f" - {ingredient}")

    print("\nSynergies du rétinol :")

    print(
        get_synergies("rétinol")
    )

    print("\nContre-indications du rétinol :")

    print(
        get_contraindications("retinol")
    )

    print("\nInteraction rétinol / acide glycolique :")

    print(
        have_interaction(
            "retinol",
            "acide glycolique"
        )
    )

    print("\nAnalyse niacinamide + acide azélaïque :")

    print(
        analyze_ingredients(
            [
                "niacinamide",
                "acide azélaïque",
            ]
        )
    )