"""
Recommendation Model

Cosine similarity between:
    user profile vector
    product matrix

Integration:
    - safety_rules.py
    - knowledge_graph.py

Deliverable:
    recommend_products(profile)
"""

from pathlib import Path
import ast

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .safety_rules import (
    is_product_safe,
    evaluate_profile_safety,
)

from .knowledge_graph import (
    normalize_ingredient,
    have_synergy,
    have_interaction,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "recommendation"
)

VECTORIZER_PATH = (
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

PRODUCT_MATRIX_PATH = (
    MODEL_DIR / "product_matrix.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "products_metadata.joblib"
)


# ============================================================
# UTILITAIRES
# ============================================================

def parse_list(value):
    """
    Convertit une valeur pouvant contenir une liste
    sous forme Python/string en véritable liste.

    Exemple :
        "['niacinamide', 'retinol']"
        ->
        ['niacinamide', 'retinol']
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:

        result = ast.literal_eval(text)

        if isinstance(result, list):
            return result

    except (
        ValueError,
        SyntaxError
    ):
        pass

    return []


def build_profile_text(profile):
    """
    Construit le texte du profil à vectoriser.

    Les informations utilisées correspondent
    aux dimensions du profil utilisées par le moteur.
    """

    parts = []

    skin_type = profile.get(
        "skin_type"
    )

    if skin_type:

        parts.append(
            f"skin_type_{skin_type}"
        )

    for concern in profile.get(
        "concerns",
        []
    ):

        parts.append(
            f"concern_{concern}"
        )

    for ingredient in profile.get(
        "preferred_ingredients",
        []
    ):

        parts.append(
            f"active_{ingredient}"
        )

    return " ".join(parts)


def get_product_ingredients(product):
    """
    Récupère les actifs présents dans le produit.

    Les données proviennent de la colonne
    'active_ingredients' déjà préparée lors
    du nettoyage des données.
    """

    values = parse_list(
        product.get(
            "active_ingredients",
            []
        )
    )

    normalized = set()

    for value in values:

        ingredient = normalize_ingredient(
            value
        )

        if ingredient:
            normalized.add(
                ingredient
            )

    return normalized


def get_profile_ingredients(profile):
    """
    Récupère les actifs préférés de l'utilisateur
    et les normalise selon le Knowledge Graph.
    """

    values = profile.get(
        "preferred_ingredients",
        []
    )

    normalized = set()

    for value in values:

        ingredient = normalize_ingredient(
            value
        )

        if ingredient:
            normalized.add(
                ingredient
            )

    return normalized


def contains_avoided_ingredient(
    product,
    avoid_ingredients
):
    """
    Vérifie les ingrédients explicitement évités
    par l'utilisateur.
    """

    product_ingredients = {
        str(value).strip().lower()
        for value in parse_list(
            product.get(
                "active_ingredients"
            )
        )
    }

    avoided = {
        str(value).strip().lower()
        for value in avoid_ingredients
    }

    return bool(
        product_ingredients.intersection(
            avoided
        )
    )


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

def analyze_product_knowledge(
    product,
    profile
):
    """
    Analyse les relations du Knowledge Graph entre
    les actifs du produit et les actifs préférés
    du profil utilisateur.

    Cette fonction ne bloque pas le produit.

    Elle fournit uniquement des informations
    supplémentaires pour le ranking et
    l'explicabilité.

    Retourne :

        {
            "synergies": [...],
            "interactions": [...]
        }
    """

    product_ingredients = get_product_ingredients(
        product
    )

    profile_ingredients = get_profile_ingredients(
        profile
    )

    synergies = []
    interactions = []

    for product_ingredient in product_ingredients:

        for profile_ingredient in profile_ingredients:

            if have_synergy(
                product_ingredient,
                profile_ingredient
            ):

                synergies.append({
                    "product_ingredient":
                        product_ingredient,
                    "profile_ingredient":
                        profile_ingredient,
                })

            if have_interaction(
                product_ingredient,
                profile_ingredient
            ):

                interactions.append({
                    "product_ingredient":
                        product_ingredient,
                    "profile_ingredient":
                        profile_ingredient,
                })

    return {
        "synergies": synergies,
        "interactions": interactions,
    }


# ============================================================
# KNOWLEDGE GRAPH SCORE
# ============================================================

def calculate_knowledge_score(
    product,
    profile
):
    """
    Calcule un petit score complémentaire basé
    sur les relations du Knowledge Graph.

    Le score principal reste la cosine similarity.

    Le Knowledge Graph ne remplace donc pas le modèle
    de recommandation : il apporte une information
    supplémentaire au ranking.

    Règles :

        + bonus pour une synergie
        - pénalité pour une interaction

    Les valeurs restent faibles afin que le score
    TF-IDF reste le facteur principal.
    """

    analysis = analyze_product_knowledge(
        product,
        profile
    )

    synergy_count = len(
        analysis["synergies"]
    )

    interaction_count = len(
        analysis["interactions"]
    )

    synergy_bonus = (
        0.05 * synergy_count
    )

    interaction_penalty = (
        0.05 * interaction_count
    )

    return (
        synergy_bonus
        - interaction_penalty
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_products(
    profile,
    top_n=5
):
    """
    Retourne les produits les plus proches
    du profil utilisateur.

    Pipeline :

        1. Chargement du modèle
        2. Evaluation de la sécurité
        3. Construction du profil
        4. TF-IDF
        5. Cosine similarity
        6. Filtre non-skincare
        7. Règles médicales
        8. Ingrédients évités
        9. Knowledge Graph
        10. Ranking final
    """

    # ========================================================
    # Chargement des modèles
    # ========================================================

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    product_matrix = joblib.load(
        PRODUCT_MATRIX_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    # ========================================================
    # Sécurité
    # ========================================================

    safety = evaluate_profile_safety(
        profile
    )

    # ========================================================
    # Profil
    # ========================================================

    profile_text = build_profile_text(
        profile
    )

    if not profile_text.strip():

        empty = metadata.iloc[
            0:0
        ].copy()

        empty[
            "similarity_score"
        ] = pd.Series(
            dtype=float
        )

        empty[
            "knowledge_score"
        ] = pd.Series(
            dtype=float
        )

        empty[
            "final_score"
        ] = pd.Series(
            dtype=float
        )

        return {
            "recommendations": empty,
            "safety": safety,
            "knowledge_graph": {
                "synergies": [],
                "interactions": [],
            },
        }

    # ========================================================
    # User vector
    # ========================================================

    user_vector = vectorizer.transform(
        [profile_text]
    )

    # ========================================================
    # Cosine similarity
    # ========================================================

    similarity_scores = (
        cosine_similarity(
            user_vector,
            product_matrix
        ).flatten()
    )

    # ========================================================
    # Filtre non-skincare
    # ========================================================

    valid_mask = pd.Series(
        True,
        index=metadata.index
    )

    if "is_non_skincare" in metadata.columns:

        valid_mask &= (
            metadata[
                "is_non_skincare"
            ].fillna(False) != True
        )

    # ========================================================
    # Règles médicales
    # ========================================================

    valid_mask &= metadata.apply(
        lambda product:
        is_product_safe(
            product,
            profile
        ),
        axis=1
    )

    # ========================================================
    # Ingrédients évités
    # ========================================================

    avoid_ingredients = profile.get(
        "avoid_ingredients",
        []
    )

    if avoid_ingredients:

        avoided_mask = metadata.apply(
            lambda product:
            contains_avoided_ingredient(
                product,
                avoid_ingredients
            ),
            axis=1
        )

        valid_mask &= ~avoided_mask

    # ========================================================
    # Knowledge Graph
    # ========================================================

    knowledge_scores = []

    for _, product in metadata.iterrows():

        knowledge_score = (
            calculate_knowledge_score(
                product,
                profile
            )
        )

        knowledge_scores.append(
            knowledge_score
        )

    knowledge_scores = pd.Series(
        knowledge_scores,
        index=metadata.index,
        dtype=float
    )

    # ========================================================
    # Score final
    # ========================================================

    final_scores = (
        similarity_scores
        + knowledge_scores.to_numpy()
    )

    # ========================================================
    # Supprimer produits interdits
    # ========================================================

    final_scores[
        ~valid_mask.to_numpy()
    ] = -1

    # ========================================================
    # Ranking
    # ========================================================

    ranked_indices = (
        final_scores
        .argsort()[::-1]
    )

    ranked_indices = [
        index
        for index in ranked_indices
        if final_scores[index] >= 0
    ]

    ranked_indices = ranked_indices[
        :top_n
    ]

    # ========================================================
    # Résultat
    # ========================================================

    recommendations = metadata.iloc[
        ranked_indices
    ].copy()

    # Score TF-IDF / cosine
    recommendations[
        "similarity_score"
    ] = similarity_scores[
        ranked_indices
    ]

    # Score Knowledge Graph
    recommendations[
        "knowledge_score"
    ] = knowledge_scores.iloc[
        ranked_indices
    ].to_numpy()

    # Score final
    recommendations[
        "final_score"
    ] = final_scores[
        ranked_indices
    ]

    recommendations = (
        recommendations
        .reset_index(drop=True)
    )

    # ========================================================
    # Analyse globale du Knowledge Graph
    # ========================================================

    global_synergies = []
    global_interactions = []

    profile_ingredients = (
        get_profile_ingredients(
            profile
        )
    )

    for ingredient in profile_ingredients:

        for other in profile_ingredients:

            if ingredient == other:
                continue

            if have_synergy(
                ingredient,
                other
            ):

                pair = (
                    ingredient,
                    other
                )

                reverse_pair = (
                    other,
                    ingredient
                )

                if (
                    pair not in global_synergies
                    and
                    reverse_pair not in global_synergies
                ):

                    global_synergies.append(
                        pair
                    )

            if have_interaction(
                ingredient,
                other
            ):

                pair = (
                    ingredient,
                    other
                )

                reverse_pair = (
                    other,
                    ingredient
                )

                if (
                    pair not in global_interactions
                    and
                    reverse_pair not in global_interactions
                ):

                    global_interactions.append(
                        pair
                    )

    # ========================================================
    # Retour
    # ========================================================

    return {
        "recommendations": recommendations,

        "safety": safety,

        "knowledge_graph": {
            "synergies":
                global_synergies,

            "interactions":
                global_interactions,
        },
    }