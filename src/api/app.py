"""
src/api/app.py

Run (depuis la racine du projet) :
    python -m src.api.app
"""

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from src.recommendation.recommender import recommend_products
from src.recommendation.routine_builder import build_routine
from src.chatbot.chat_pipeline import reset_session
from src.agent.orchestrator import orchestrate


from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Par défaut, Flask trie les clés JSON par ordre alphabétique
# ("evening" passe avant "morning", les champs produit sont
# réarrangés alphabétiquement). On désactive ce tri pour que l'ordre
# d'insertion des dicts Python soit respecté dans la réponse.
app.json.sort_keys = False


# Champs produit exposés dans les réponses JSON, dans cet ordre.
# Les autres colonnes du dataset (ingredients_clean, category,
# has_ingredients_data, is_non_skincare, similarity_score...) sont
# internes au moteur et ne présentent aucun intérêt côté client :
# on les retire plutôt que de les laisser traîner en fin de payload.
PRODUCT_FIELDS = [
    "product_name",
    "brand",
    "price_tnd",
    "routine_step",
    "skin_type",
    "concerns",
    "active_ingredients",
    "availability",
    "product_url",
    "image_url",
]


def _select_product_fields(product: dict) -> dict:
    return {key: product[key] for key in PRODUCT_FIELDS if key in product}


def _clean_value(value):
    """JSON ne connaît ni NaN ni les types numpy."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _dataframe_to_records(df: pd.DataFrame) -> list:
    records = df.to_dict("records")
    return [
        _select_product_fields({key: _clean_value(value) for key, value in record.items()})
        for record in records
    ]


def _product_to_json(product):
    if product is None:
        return None
    cleaned = {key: _clean_value(value) for key, value in product.items()}
    return _select_product_fields(cleaned)


@app.route("/ml/recommend", methods=["POST"])
def recommend():
    """
    Body JSON attendu (profil utilisateur, tous les champs sont optionnels) :
        {
            "skin_type": "grasse",
            "concerns": ["acne", "taches"],
            "preferred_ingredients": ["niacinamide"],
            "avoid_ingredients": [],
            "pregnant": false,
            "breastfeeding": false,
            "rosacea": false,
            "ultra_sensitive": false,
            "dermatological_treatment": false
        }

    Query param optionnel : ?top_n=5 (défaut 5)
    """
    profile = request.get_json(silent=True)

    if not isinstance(profile, dict):
        return jsonify({"error": "Un body JSON (objet) représentant le profil est requis."}), 400

    try:
        top_n = int(request.args.get("top_n", 5))
    except ValueError:
        return jsonify({"error": "top_n doit être un entier."}), 400

    try:
        result = recommend_products(profile, top_n=top_n)
    except FileNotFoundError:
        return jsonify({
            "error": (
                "Modèle non entraîné. Lancer "
                "`python scripts/train_model.py` puis relancer Flask."
            )
        }), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Erreur moteur de recommandation: {exc}"}), 500

    return jsonify({
        "profile": profile,
        "top_n": top_n,
        "recommendations": _dataframe_to_records(result["recommendations"]),
        "safety": result["safety"],
    }), 200


@app.route("/ml/routine", methods=["POST"])
def routine():
    """
    Même profil que /ml/recommend, mais renvoie une routine AM/PM
    assemblée plutôt qu'un simple top-N.
    """
    profile = request.get_json(silent=True)

    if not isinstance(profile, dict):
        return jsonify({"error": "Un body JSON (objet) représentant le profil est requis."}), 400

    try:
        result = build_routine(profile)
    except FileNotFoundError:
        return jsonify({
            "error": (
                "Modèle non entraîné. Lancer "
                "`python scripts/train_model.py` puis relancer Flask."
            )
        }), 503
    except Exception as exc:
        return jsonify({"error": f"Erreur construction de routine: {exc}"}), 500

    morning = {step: _product_to_json(product) for step, product in result["morning"].items()}
    evening = {step: _product_to_json(product) for step, product in result["evening"].items()}

    return jsonify({
        "profile": profile,
        "routine": {"morning": morning, "evening": evening},
        "safety": result["safety"],
    }), 200


@app.route("/ml/chat", methods=["POST"])
def chat():
    """
    Chat Pipeline

    Body JSON attendu :
        {
            "message": "J'ai la peau grasse avec de l'acné",
            "session_id": "optionnel, généré si absent"
        }

    Le session_id doit être renvoyé par le client à chaque appel
    suivant pour conserver le profil construit au fil de la
    conversation (type de peau, préoccupations, ingrédients,
    dernier produit recommandé).

    NOTE : le stockage de session est en mémoire côté serveur pour
    l'instant (cf. chat_pipeline.py). À remplacer par Redis/DB avant
    la mise en prod ou le déploiement multi-workers.
    """
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({"error": "Un body JSON (objet) est requis."}), 400

    message = body.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Le champ 'message' (texte non vide) est requis."}), 400

    session_id = body.get("session_id")

    try:
        result = orchestrate(
            message,
            session_id=session_id,
            language=body.get("language"),
        )
    except FileNotFoundError:
        return jsonify({
            "error": (
                "Modèle non entraîné. Lancer "
                "`python scripts/train_model.py` et "
                "`python -m src.chatbot.intent_classifier` puis relancer Flask."
            )
        }), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Erreur du pipeline de chat: {exc}"}), 500

    return jsonify(result), 200


@app.route("/ml/rag/ask", methods=["POST"])
def rag_ask():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Un body JSON (objet) est requis."}), 400

    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Le champ 'message' (texte non vide) est requis."}), 400

    try:
        result = orchestrate(
            message,
            session_id=body.get("session_id"),
            force_rag=True,
            language=body.get("language"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Erreur du service RAG: {exc}"}), 500
    status_code = 503 if result.get("mode") == "rag_unavailable" else 200
    return jsonify(result), status_code


@app.route("/ml/agent/ask", methods=["POST"])
def agent_ask():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Un body JSON (objet) est requis."}), 400

    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Le champ 'message' (texte non vide) est requis."}), 400

    try:
        result = orchestrate(
            message,
            session_id=body.get("session_id"),
            language=body.get("language"),
        )
    except FileNotFoundError:
        return jsonify({
            "error": (
                "Modèle non entraîné. Lancer `python scripts/train_model.py` "
                "puis relancer Flask."
            )
        }), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Erreur de l'orchestrateur: {exc}"}), 500
    status_code = 503 if result.get("mode") == "rag_unavailable" else 200
    return jsonify(result), status_code


@app.route("/ml/chat/reset", methods=["POST"])
def chat_reset():
    """
    Réinitialise le profil de session (nouveau départ de conversation).

    Body JSON attendu :
        {"session_id": "..."}
    """
    body = request.get_json(silent=True)

    if not isinstance(body, dict) or not body.get("session_id"):
        return jsonify({"error": "Le champ 'session_id' est requis."}), 400

    reset_session(body["session_id"])

    return jsonify({"status": "reset", "session_id": body["session_id"]}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)