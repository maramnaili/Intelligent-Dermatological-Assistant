"""
AI-S2-07 — Intent Classifier

Pipeline:
    TF-IDF -> Logistic Regression

Le modèle est entraîné à partir de :
    data/chatbot/chatbot_dataset.csv

Colonnes attendues :
    text
    intent

Le chatbot supporte des exemples en français et en anglais,
mais conserve exactement les mêmes intents.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "chatbot"
    / "chatbot_dataset.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "chatbot"
)

MODEL_PATH = (
    MODEL_DIR
    / "intent_classifier.joblib"
)


# ============================================================
# TRAINING
# ============================================================

def train_intent_classifier():

    print("=" * 60)
    print("AI-S2-07 — Training Intent Classifier")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "text",
        "intent",
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le dataset : {missing}"
        )

    # Suppression des lignes invalides
    df = df.dropna(
        subset=["text", "intent"]
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["text"] != "")
        &
        (df["intent"] != "")
    ]

    print(
        f"Nombre total d'exemples : {len(df)}"
    )

    print(
        f"Nombre d'intents : "
        f"{df['intent'].nunique()}"
    )

    print("\nDistribution :")
    print(
        df["intent"]
        .value_counts()
    )

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------
    #
    # Ce split ne sert qu'à ESTIMER la performance du modèle
    # (accuracy, macro-F1 ci-dessous). Avec un dataset de cette
    # taille (173 exemples / 7 intents), 20% de côté représente
    # ~35 exemples — assez pour qu'un mot aussi basique que
    # "Hello" ou "Hi" tombe entièrement dans le test et ne soit
    # JAMAIS vu à l'entraînement. Le modèle sauvegardé plus bas
    # n'est donc PAS celui entraîné ici : il est réentraîné sur
    # l'intégralité des données (cf. bloc "FINAL FIT" plus bas).

    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # --------------------------------------------------------
    # TF-IDF + LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=5000,
                sublinear_tf=True,
            ),
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    print("\nTraining (evaluation split)...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------
    #
    # Mesurée sur le modèle entraîné SANS les 20% de test, donc
    # représentative de ce que verrait un nouvel exemple jamais
    # vu. C'est la seule raison d'être de ce split.

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print("\nEvaluation")
    print("-" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
    )

    print(
        "Target Macro-F1 : >= 0.80"
    )

    if macro_f1 >= 0.80:
        print(
            "Status : TARGET REACHED"
        )
    else:
        print(
            "Status : TARGET NOT REACHED"
        )

    # --------------------------------------------------------
    # FINAL FIT — réentraînement sur 100% des données
    # --------------------------------------------------------
    #
    # Le modèle évalué ci-dessus n'a vu que 80% des exemples.
    # Pour la production, on réentraîne un pipeline identique
    # sur l'intégralité de X/y : c'est CE modèle qui est
    # sauvegardé et chargé par predict_intent(). Les métriques
    # affichées plus haut restent une estimation honnête de sa
    # performance (elles ne sont pas recalculées sur ce modèle
    # final, précisément pour ne pas se juger sur ses propres
    # données d'entraînement).

    print(
        "\nRéentraînement final sur "
        "100% des données (pour la "
        "sauvegarde)..."
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=5000,
                sublinear_tf=True,
            ),
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "model_path": str(MODEL_PATH),
    }


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def load_intent_classifier():

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Le modèle d'intention n'existe pas. "
                "Lancez d'abord l'entraînement."
            )

        _model = joblib.load(
            MODEL_PATH
        )

    return _model


# ============================================================
# PREDICT
# ============================================================

def predict_intent(text):

    if not isinstance(text, str):
        raise ValueError(
            "text doit être une chaîne de caractères."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Le message ne peut pas être vide."
        )

    model = load_intent_classifier()

    intent = model.predict(
        [text]
    )[0]

    probabilities = (
        model.predict_proba(
            [text]
        )[0]
    )

    confidence = float(
        probabilities.max()
    )

    return {
        "intent": intent,
        "confidence": confidence,
    }


# ============================================================
# CLI TRAINING
# ============================================================

if __name__ == "__main__":

    train_intent_classifier()