from pathlib import Path
import ast
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "parafendri_products_clean.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "recommendation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
PRODUCT_MATRIX_PATH = MODEL_DIR / "product_matrix.joblib"
METADATA_PATH = MODEL_DIR / "products_metadata.joblib"


# ============================================================
# UTILITAIRE
# ============================================================

def parse_list(value):
    """
    Transforme une chaîne du type :

        "['acide_salicylique', 'niacinamide']"

    en :

        ['acide_salicylique', 'niacinamide']
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value:
        return []

    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

    except (ValueError, SyntaxError):
        pass

    return []


# ============================================================
# TEXTE PRODUIT
# ============================================================

def build_product_text(row):

    parts = []

    # --------------------------------------------------------
    # Nom
    # --------------------------------------------------------

    product_name = row.get("product_name")

    if pd.notna(product_name):
        parts.append(str(product_name))

    # --------------------------------------------------------
    # Catégorie
    # --------------------------------------------------------

    category = row.get("category")

    if pd.notna(category):
        parts.append(
            f"category_{category}"
        )

    # --------------------------------------------------------
    # Routine
    # --------------------------------------------------------

    routine_step = row.get("routine_step")

    if pd.notna(routine_step):
        parts.append(
            f"routine_{routine_step}"
        )

    # --------------------------------------------------------
    # Skin type
    # --------------------------------------------------------

    skin_type = row.get("skin_type")

    if pd.notna(skin_type):
        parts.append(
            f"skin_type_{skin_type}"
        )

    # --------------------------------------------------------
    # Concerns
    # --------------------------------------------------------

    concerns = parse_list(
        row.get("concerns")
    )

    for concern in concerns:

        parts.append(
            f"concern_{concern}"
        )

    # --------------------------------------------------------
    # Actifs
    # --------------------------------------------------------

    active_ingredients = parse_list(
        row.get("active_ingredients")
    )

    for ingredient in active_ingredients:

        parts.append(
            f"active_{ingredient}"
        )

    return " ".join(parts)


# ============================================================
# TRAINING
# ============================================================

def train_model():

    print(
        f"Chargement du dataset : {DATA_PATH}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Produits chargés : {len(df)}"
    )

    # --------------------------------------------------------
    # Exclure les produits hors skincare
    # --------------------------------------------------------

    if "is_non_skincare" in df.columns:

        df = df[
            df["is_non_skincare"] != True
        ].copy()

    print(
        f"Produits utilisés : {len(df)}"
    )

    # --------------------------------------------------------
    # Construire représentation textuelle
    # --------------------------------------------------------

    df["product_text"] = df.apply(
        build_product_text,
        axis=1
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\wÀ-ÿ]+\b"
    )

    product_matrix = vectorizer.fit_transform(
        df["product_text"]
    )

    print(
        f"Vocabulaire : {len(vectorizer.vocabulary_)} tokens"
    )

    print(
        f"Matrice produits : {product_matrix.shape}"
    )

    # --------------------------------------------------------
    # Sauvegarde vectorizer
    # --------------------------------------------------------

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    # --------------------------------------------------------
    # Sauvegarde matrice
    # --------------------------------------------------------

    joblib.dump(
        product_matrix,
        PRODUCT_MATRIX_PATH
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_columns = [
        "product_name",
        "brand",
        "price_tnd",
        "availability",
        "image_url",
        "product_url",
        "category",
        "routine_step",
        "skin_type",
        "concerns",
        "ingredients_clean",
        "active_ingredients",
        "has_ingredients_data",
        "is_non_skincare"
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    metadata = df[
        metadata_columns
    ].reset_index(drop=True)

    joblib.dump(
        metadata,
        METADATA_PATH
    )

    print("\n======================================")
    print("✅ MODÈLE DE RECOMMANDATION ENTRAÎNÉ")
    print("======================================")
    print(f"Vectorizer : {VECTORIZER_PATH}")
    print(f"Matrix     : {PRODUCT_MATRIX_PATH}")
    print(f"Metadata   : {METADATA_PATH}")


if __name__ == "__main__":
    train_model()