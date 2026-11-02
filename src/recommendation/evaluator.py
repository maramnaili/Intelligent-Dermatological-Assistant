"""
AI-S2-03 — Evaluate Recommendation Model

Metrics:
    - Precision@3
    - Precision@5

Target:
    Precision@3 >= 0.70
"""

import ast
import pandas as pd

from .recommender import recommend_products


# ============================================================
# 20 profils de test
# ============================================================

TEST_PROFILES = [
    {"id": "P01", "skin_type": "grasse", "concerns": ["acne"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
    {"id": "P02", "skin_type": "grasse", "concerns": ["pores"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
    {"id": "P03", "skin_type": "seche", "concerns": ["deshydratation"], "preferred_ingredients": ["acide_hyaluronique"], "avoid_ingredients": []},
    {"id": "P04", "skin_type": "seche", "concerns": ["secheresse"], "preferred_ingredients": ["ceramides"], "avoid_ingredients": []},
    {"id": "P05", "skin_type": "sensible", "concerns": ["rougeurs"], "preferred_ingredients": ["panthenol"], "avoid_ingredients": []},
    {"id": "P06", "skin_type": "mixte", "concerns": ["acne"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
    {"id": "P07", "skin_type": "grasse", "concerns": ["taches"], "preferred_ingredients": ["vitamine_c"], "avoid_ingredients": []},
    {"id": "P08", "skin_type": "seche", "concerns": ["taches"], "preferred_ingredients": ["vitamine_c"], "avoid_ingredients": []},
    {"id": "P09", "skin_type": "sensible", "concerns": ["sensibilite"], "preferred_ingredients": ["panthenol"], "avoid_ingredients": []},
    {"id": "P10", "skin_type": "grasse", "concerns": ["rides"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
    {"id": "P11", "skin_type": "seche", "concerns": ["rides"], "preferred_ingredients": ["acide_hyaluronique"], "avoid_ingredients": []},
    {"id": "P12", "skin_type": "mixte", "concerns": ["pores"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
    {"id": "P13", "skin_type": "grasse", "concerns": ["deshydratation"], "preferred_ingredients": ["acide_hyaluronique"], "avoid_ingredients": []},
    {"id": "P14", "skin_type": "sensible", "concerns": ["rougeurs"], "preferred_ingredients": ["ceramides"], "avoid_ingredients": []},
    {"id": "P15", "skin_type": "seche", "concerns": ["sensibilite"], "preferred_ingredients": ["panthenol"], "avoid_ingredients": []},
    {"id": "P16", "skin_type": "grasse", "concerns": ["acne"], "preferred_ingredients": ["acide_salicylique"], "avoid_ingredients": []},
    {"id": "P17", "skin_type": "mixte", "concerns": ["taches"], "preferred_ingredients": ["vitamine_c"], "avoid_ingredients": []},
    {"id": "P18", "skin_type": "sensible", "concerns": ["rougeurs"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
    {"id": "P19", "skin_type": "seche", "concerns": ["deshydratation"], "preferred_ingredients": ["acide_hyaluronique"], "avoid_ingredients": []},
    {"id": "P20", "skin_type": "grasse", "concerns": ["pores"], "preferred_ingredients": ["niacinamide"], "avoid_ingredients": []},
]


# ============================================================
# UTILITAIRE
# ============================================================

def parse_list(value):
    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def normalize(value):
    if value is None:
        return ""
    return str(value).strip().lower()


# ============================================================
# PERTINENCE
# ============================================================

def is_relevant(product, profile):
    """
    Un produit est considéré pertinent s'il correspond
    à au moins 2 critères du profil.

    Critères :
        - type de peau
        - préoccupation
        - actif préféré

    Cette règle rend l'évaluation plus discriminante
    que la simple condition OR.
    """

    score = 0

    # Type de peau
    skin_type = normalize(profile.get("skin_type"))
    product_skin = normalize(product.get("skin_type"))

    if skin_type and product_skin == skin_type:
        score += 1

    # Préoccupation
    product_text = " ".join([
        normalize(product.get("product_name")),
        normalize(product.get("description")),
        normalize(product.get("category")),
    ])

    for concern in profile.get("concerns", []):
        if normalize(concern) in product_text:
            score += 1
            break

    # Actif préféré
    ingredients = {
        normalize(x)
        for x in parse_list(
            product.get("active_ingredients")
        )
    }

    preferred = {
        normalize(x)
        for x in profile.get(
            "preferred_ingredients", []
        )
    }

    if ingredients.intersection(preferred):
        score += 1

    return score >= 2


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(recommendations, profile, k):
    """
    Precision@K =
    nombre de produits pertinents dans Top-K / K
    """

    if recommendations.empty:
        return 0.0

    top_k = recommendations.head(k)

    relevant = sum(
        is_relevant(product, profile)
        for _, product in top_k.iterrows()
    )

    return relevant / k


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(profiles=None):

    if profiles is None:
        profiles = TEST_PROFILES

    results = []

    for profile in profiles:

        output = recommend_products(
            profile,
            top_n=5
        )

        recommendations = output[
            "recommendations"
        ]

        p3 = precision_at_k(
            recommendations,
            profile,
            3
        )

        p5 = precision_at_k(
            recommendations,
            profile,
            5
        )

        results.append({
            "profile_id": profile["id"],
            "precision_at_3": p3,
            "precision_at_5": p5,
        })

    df = pd.DataFrame(results)

    return {
        "results": df,
        "mean_precision_at_3":
            df["precision_at_3"].mean(),
        "mean_precision_at_5":
            df["precision_at_5"].mean(),
    }


# ============================================================
# RAPPORT
# ============================================================

def print_report(evaluation):

    df = evaluation["results"]

    print("\n" + "=" * 65)
    print("AI-S2-03 — EVALUATION")
    print("=" * 65)

    print(f"\nProfils testés : {len(df)}")

    print("\nRésultats :")
    print("-" * 65)

    for _, row in df.iterrows():
        print(
            f"{row['profile_id']} | "
            f"P@3 = {row['precision_at_3']:.3f} | "
            f"P@5 = {row['precision_at_5']:.3f}"
        )

    print("-" * 65)

    p3 = evaluation["mean_precision_at_3"]
    p5 = evaluation["mean_precision_at_5"]

    print(f"Precision@3 moyenne : {p3:.3f}")
    print(f"Precision@5 moyenne : {p5:.3f}")

    print(
        f"\nObjectif P@3 >= 0.70 : "
        f"{'✅ ATTEINT' if p3 >= 0.70 else '❌ NON ATTEINT'}"
    )

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluation = evaluate_model()

    print_report(evaluation)