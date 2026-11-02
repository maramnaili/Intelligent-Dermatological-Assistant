from src.recommendation.recommender import recommend_products


def main():

    profile = {
        "skin_type": "grasse",
        "concerns": ["acne"],
        "preferred_ingredients": ["niacinamide"],
        "avoid_ingredients": ["retinol"]
    }

    results = recommend_products(
        profile,
        top_n=5
    )

    print("\n" + "=" * 80)
    print("RECOMMANDATIONS")
    print("=" * 80)

    if results.empty:
        print("Aucune recommandation.")
        return

    for i, product in results.iterrows():

        print(
            f"\n{i + 1}. {product['product_name']}"
        )

        print(
            f"   Score      : "
            f"{product['similarity_score']:.3f}"
        )

        print(
            f"   Marque     : "
            f"{product.get('brand', '')}"
        )

        print(
            f"   Prix       : "
            f"{product.get('price_tnd', '')} TND"
        )

        print(
            f"   Routine    : "
            f"{product.get('routine_step', '')}"
        )

        print(
            f"   Actifs     : "
            f"{product.get('active_ingredients', '')}"
        )


if __name__ == "__main__":
    main()