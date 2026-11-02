"""
AI-S2-02 — Assemble Routine Output

Routine Matin :
    Nettoyage
    Sérum
    Hydratant
    Protection solaire

Routine Soir :
    Démaquillage
    Nettoyage
    Traitement
    Réparation
"""

from .recommender import recommend_products


MORNING_STEPS = [
    "nettoyage",
    "serum",
    "hydratant",
    "protection_solaire",
]

EVENING_STEPS = [
    "demaquillage",
    "nettoyage",
    "traitement",
    "reparation",
]


def normalize_step(value):

    if value is None:
        return ""

    return str(value).strip().lower()


def select_top_product(
    products,
    step
):
    """
    Sélectionne le produit ayant le meilleur score
    pour la catégorie demandée.
    """

    candidates = products[
        products[
            "routine_step"
        ]
        .apply(normalize_step)
        == step
    ]

    if candidates.empty:
        return None

    best = candidates.iloc[0]

    return best.to_dict()


def build_routine(
    profile,
    candidates=100
):
    """
    AI-S2-02

    Génère les routines AM et PM
    à partir des recommandations.
    """

    result = recommend_products(
        profile,
        top_n=candidates
    )

    products = result[
        "recommendations"
    ]

    safety = result[
        "safety"
    ]

    # --------------------------------------------------------
    # Routine matin
    # --------------------------------------------------------

    morning = {}

    for step in MORNING_STEPS:

        morning[step] = select_top_product(
            products,
            step
        )

    # --------------------------------------------------------
    # Routine soir
    # --------------------------------------------------------

    evening = {}

    for step in EVENING_STEPS:

        evening[step] = select_top_product(
            products,
            step
        )

    return {
        "morning": morning,
        "evening": evening,
        "safety": safety,
    }