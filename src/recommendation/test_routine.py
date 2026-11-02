from src.recommendation.routine_builder import build_routine


def print_product(
    step,
    product
):

    print(
        f"\n{step.upper()}"
    )

    if product is None:

        print(
            "   ❌ Aucun produit disponible"
        )

        return

    print(
        f"   Produit : "
        f"{product.get('product_name', '')}"
    )

    print(
        f"   Marque  : "
        f"{product.get('brand', '')}"
    )

    print(
        f"   Prix    : "
        f"{product.get('price_tnd', '')} TND"
    )

    print(
        f"   Score   : "
        f"{product.get('similarity_score', 0):.3f}"
    )

    print(
        f"   Actifs  : "
        f"{product.get('active_ingredients', '')}"
    )


def main():

    profile = {

        "skin_type": "grasse",

        "concerns": [
            "acne"
        ],

        "preferred_ingredients": [
            "niacinamide"
        ],

        "avoid_ingredients": [
            "retinol"
        ]
    }

    routine = build_routine(
        profile
    )

    # ========================================================
    # MORNING
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "🌞 ROUTINE DU MATIN"
    )

    print(
        "=" * 80
    )

    for step, product in routine[
        "morning"
    ].items():

        print_product(
            step,
            product
        )

    # ========================================================
    # EVENING
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "🌙 ROUTINE DU SOIR"
    )

    print(
        "=" * 80
    )

    for step, product in routine[
        "evening"
    ].items():

        print_product(
            step,
            product
        )


if __name__ == "__main__":
    main()