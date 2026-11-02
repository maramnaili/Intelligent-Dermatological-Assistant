# evaluate_train.py

import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# 1. CHEMINS
# ============================================================

DATA_PATH = "data/chatbot/chatbot_dataset.csv"
MODEL_PATH = "models/chatbot/intent_classifier.joblib"


# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# Vérification des colonnes
if "text" not in df.columns or "intent" not in df.columns:
    raise ValueError(
        "Le dataset doit contenir les colonnes 'text' et 'intent'."
    )

# Suppression des lignes invalides
df = df.dropna(subset=["text", "intent"])

X = df["text"]
y = df["intent"]


# ============================================================
# 3. CHARGEMENT DU MODÈLE DÉJÀ ENTRAÎNÉ
# ============================================================

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("ÉVALUATION DU MODÈLE SUR LE JEU D'ENTRAÎNEMENT")
print("=" * 60)

print(f"Nombre total d'exemples : {len(X)}")


# ============================================================
# 4. PRÉDICTIONS SUR LES DONNÉES D'ENTRAÎNEMENT
# ============================================================

y_pred = model.predict(X)


# ============================================================
# 5. CALCUL DES MÉTRIQUES
# ============================================================

accuracy = accuracy_score(y, y_pred)

macro_f1 = f1_score(
    y,
    y_pred,
    average="macro"
)


# ============================================================
# 6. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\nRésultats sur le jeu d'entraînement :")
print("-" * 60)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro-F1  : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")


# ============================================================
# 7. RAPPORT DÉTAILLÉ PAR INTENTION
# ============================================================

print("\nRapport de classification :")
print("-" * 60)

print(
    classification_report(
        y,
        y_pred,
        digits=4
    )
)


# ============================================================
# 8. INTERPRÉTATION
# ============================================================

print("=" * 60)
print("INTERPRÉTATION")
print("=" * 60)

if accuracy >= 0.95:
    print(
        "Accuracy très élevée sur le jeu d'entraînement."
    )
    print(
        "Le modèle reconnaît très bien les exemples qu'il a vus."
    )
else:
    print(
        "L'Accuracy sur le jeu d'entraînement n'est pas extrêmement élevée."
    )

print(
    "\nPour détecter un éventuel surapprentissage, "
    "comparez ces résultats avec ceux obtenus sur le jeu de test."
)

print(
    "\nUn écart important entre Train et Test peut indiquer "
    "un surapprentissage."
)