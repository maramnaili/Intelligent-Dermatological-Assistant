"""
Chat Pipeline

Pipeline:

User message
    ↓
Intent classification
    ↓
Entity extraction
    ↓
Profile update
    ↓
Recommendation / Routine / Explanation
    ↓
Response template
    ↓
Final response

Supported languages:
    French
    English

Existing intents:
    greeting
    goodbye
    profile_info
    product_recommendation
    routine_request
    routine_explanation
    routine_modification
"""

import math
import re
import uuid


# ============================================================
# IMPORTS
# ============================================================

from src.chatbot.intent_classifier import (
    predict_intent,
)

from src.chatbot.entity_extractor import (
    extract_entities,
)

from src.chatbot.response_templates import (
    render_response,
)

from src.recommendation.recommender import (
    recommend_products,
    parse_list,
)

from src.recommendation.routine_builder import (
    build_routine,
)

from src.recommendation.safety_rules import (
    detect_medical_alerts,
)

from src.recommendation.knowledge_graph import (
    normalize_ingredient,
    RETINOL,
    SALICYLIC_ACID,
    GLYCOLIC_ACID,
    NIACINAMIDE,
    AZELAIC_ACID,
    VITAMIN_C,
    CERAMIDES,
    PANTHENOL,
    HYALURONIC_ACID,
)

EXPLICIT_ROUTINE_REQUEST_PATTERNS = (
    r"\b(?:build|create|make|generate|adapt|update|recommend)\b"
    r".{0,120}\b(?:my|a|an|the)?\s*(?:morning\s+and\s+evening\s+)?"
    r"(?:skin\s*care\s*)?routine\b",
    r"\b(?:propose(?:-moi)?|recommande(?:z)?|recommander|recommandation|"
    r"crée|creer|créer|construis|génère|genere|prépare|prepare|adapte|fais)\b"
    r".{0,100}\b(?:ma|une|la)?\s*routine\b",
    r"\b(?:add|include|integrate|ajoute|ajouter|intègre|integre|intégrer|"
    r"integrer)\b.{0,100}\b(?:my|ma|une|la)?\s*(?:morning|matin|routine)\b",
)


def _is_explicit_routine_request(text):
    return any(
        re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        for pattern in EXPLICIT_ROUTINE_REQUEST_PATTERNS
    )


# ============================================================
# 1. LANGUAGE DETECTION
# ============================================================

FRENCH_WORDS = {
    "bonjour",
    "salut",
    "bonsoir",
    "coucou",
    "merci",
    "peau",
    "grasse",
    "sèche",
    "seche",
    "sensible",
    "normale",
    "mixte",
    "produit",
    "produits",
    "routine",
    "crème",
    "creme",
    "sérum",
    "serum",
    "nettoyant",
    "hydratant",
    "pourquoi",
    "comment",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "vous",
    "avec",
    "pour",
    "sans",
    "j'ai",
    "je",
    "ma",
    "mon",
    "mes",
}

ENGLISH_WORDS = {
    "hello",
    "hi",
    "hey",
    "goodbye",
    "bye",
    "thanks",
    "thank",
    "morning",
    "afternoon",
    "evening",
    "skin",
    "oily",
    "dry",
    "sensitive",
    "normal",
    "combination",
    "product",
    "products",
    "routine",
    "cream",
    "serum",
    "cleanser",
    "moisturizer",
    "sunscreen",
    "why",
    "how",
    "what",
    "which",
    "can",
    "could",
    "recommend",
    "recommendation",
    "have",
    "my",
    "your",
    "with",
    "for",
    "without",
    "build",
    "skincare",
    "create",
    "make",
    "give",
    "show",
    
}


def detect_language(text):
    """
    Détecte la langue principale du message.

    Retourne:
        "fr" ou "en"
    """

    text = text.lower().strip()

    # Cas évidents
    if text in {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "goodbye",
        "bye",
        "see you",
        "see you soon",
    }:
        return "en"

    if text in {
        "bonjour",
        "salut",
        "bonsoir",
        "coucou",
        "au revoir",
        "à bientôt",
    }:
        return "fr"

    words = set(
        re.findall(
            r"[a-zA-ZÀ-ÿ'-]+",
            text
        )
    )

    fr_score = len(
        words & FRENCH_WORDS
    )

    en_score = len(
        words & ENGLISH_WORDS
    )

    if en_score > fr_score:
        return "en"

    return "fr"


# ============================================================
# 2. EXACT INTENT DETECTION
# ============================================================
#
# Certains messages sont suffisamment explicites pour ne pas
# laisser le classifieur ML décider.
#
# Cela évite par exemple:
#     "goodbye" -> greeting
#     "hi"       -> goodbye
#
# Le classifieur reste utilisé pour les autres messages.
# ============================================================

ENGLISH_EXACT_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

FRENCH_EXACT_GREETINGS = {
    "bonjour",
    "salut",
    "bonsoir",
    "coucou",
}

ENGLISH_EXACT_GOODBYES = {
    "bye",
    "goodbye",
    "see you",
    "see you soon",
    "take care",
}

FRENCH_EXACT_GOODBYES = {
    "au revoir",
    "à bientôt",
    "a bientôt",
}


def detect_exact_intent(text):
    """
    Détecte les messages très courts et non ambigus.

    Retourne:
        ("greeting", "en")
        ("greeting", "fr")
        ("goodbye", "en")
        ("goodbye", "fr")
        ou
        (None, None)
    """

    normalized = (
        text
        .strip()
        .lower()
    )

    if normalized in ENGLISH_EXACT_GREETINGS:
        return "greeting", "en"

    if normalized in FRENCH_EXACT_GREETINGS:
        return "greeting", "fr"

    if normalized in ENGLISH_EXACT_GOODBYES:
        return "goodbye", "en"

    if normalized in FRENCH_EXACT_GOODBYES:
        return "goodbye", "fr"

    return None, None


# ============================================================
# 3. SESSIONS
# ============================================================

SESSIONS = {}


def _new_profile():
    """
    Crée un profil utilisateur vide.
    """

    return {
        "skin_type": None,
        "concerns": [],
        "preferred_ingredients": [],
        "avoid_ingredients": [],
        "last_product": None,
    }


def get_session(session_id):
    """
    Récupère ou crée une session.
    """

    if session_id not in SESSIONS:
        SESSIONS[session_id] = _new_profile()

    return SESSIONS[session_id]


def reset_session(session_id):
    """
    Réinitialise une session.
    """

    SESSIONS[session_id] = _new_profile()

    return SESSIONS[session_id]


# ============================================================
# 4. CONCERN EXTRACTION
# ============================================================

CONCERN_KEYWORDS = {

    "acne": [
        "acné",
        "acne",
        "acne-prone",
        "pimples",
        "breakouts",
        "blemishes",
        "blackheads",
        "boutons",
        "imperfections",
        "points noirs",
    ],

    "taches": [
        "tache",
        "taches",
        "tâche",
        "tâches",
        "pigmentation",
        "hyperpigmentation",
        "dark spot",
        "dark spots",
    ],

    "rides": [
        "ride",
        "rides",
        "ridules",
        "wrinkles",
        "fine lines",
        "aging",
        "ageing",
    ],

    "rougeurs": [
        "rougeur",
        "rougeurs",
        "redness",
        "rosacea",
        "rosacée",
        "rosacee",
    ],

    "secheresse": [
        "sécheresse",
        "secheresse",
        "dryness",
        "flaky",
        "flakiness",
        "tiraillement",
    ],

    "sensibilite": [
        "sensibilité",
        "sensibilite",
        "sensitive",
        "sensitivity",
        "irritation",
        "irritated",
    ],

    "eczema": [
        "eczéma",
        "eczema",
        "dermatite atopique",
        "atopic dermatitis",
    ],

    "deshydratation": [
        "déshydratation",
        "deshydratation",
        "dehydration",
        "dehydrated",
    ],

    "pores": [
        "pores",
        "pores dilatés",
        "pores dilates",
        "large pores",
        "enlarged pores",
    ],
}


def extract_concerns(text):
    """
    Extrait les préoccupations présentes dans le message.
    """

    normalized_text = (
        text
        .lower()
        .strip()
    )

    found = []

    for concern, keywords in CONCERN_KEYWORDS.items():

        for keyword in keywords:

            if keyword in normalized_text:

                if concern not in found:
                    found.append(
                        concern
                    )

                break

    return found


# ============================================================
# 5. SKIN TYPE NORMALIZATION
# ============================================================

SKIN_TYPE_MAP = {

    "grasse": "grasse",
    "oily": "grasse",

    "sèche": "seche",
    "seche": "seche",
    "dry": "seche",

    "mixte": "mixte",
    "combination": "mixte",

    "sensible": "sensible",
    "sensitive": "sensible",

    "normale": "normale",
    "normal": "normale",

    "ultra-sensible": "sensible",
    "ultra-sensitive": "sensible",

}


# ============================================================
# 6. PROFILE UPDATE
# ============================================================

def _merge_list(existing, values):
    """
    Ajoute les nouvelles valeurs sans doublons.
    """

    for value in values:

        if value not in existing:
            existing.append(value)

    return existing


def update_profile_from_entities(
    profile,
    entities,
    concerns,
):
    """
    Met à jour le profil utilisateur à partir des entités
    et préoccupations extraites.
    """

    # --------------------------------------------------------
    # Skin type
    # --------------------------------------------------------

    skin_types = entities.get(
        "skin_types",
        []
    )

    if skin_types:

        latest_skin_type = (
            skin_types[-1]
        )

        normalized_skin_type = (
            SKIN_TYPE_MAP.get(
                latest_skin_type
            )
        )

        if normalized_skin_type:

            profile[
                "skin_type"
            ] = normalized_skin_type

    # --------------------------------------------------------
    # Ingredients
    # --------------------------------------------------------

    ingredients = entities.get(
        "ingredients",
        []
    )

    normalized_ingredients = []

    for ingredient in ingredients:

        normalized = normalize_ingredient(
            ingredient
        )

        if normalized:
            normalized_ingredients.append(
                normalized
            )

    _merge_list(
        profile[
            "preferred_ingredients"
        ],
        normalized_ingredients,
    )

    # --------------------------------------------------------
    # Concerns
    # --------------------------------------------------------

    _merge_list(
        profile[
            "concerns"
        ],
        concerns,
    )

    return profile


# ============================================================
# 7. PROFILE FOR RECOMMENDATION ENGINE
# ============================================================

def _profile_for_engine(profile):
    """
    Prépare le profil dans le format attendu par
    recommend_products() et build_routine().
    """

    return {
        "skin_type": profile.get(
            "skin_type"
        ),

        "concerns": profile.get(
            "concerns",
            []
        ),

        "preferred_ingredients": profile.get(
            "preferred_ingredients",
            []
        ),

        "avoid_ingredients": profile.get(
            "avoid_ingredients",
            []
        ),
    }


# ============================================================
# 8. PROFILE TEXT
# ============================================================

SKIN_TYPE_FR = {

    "grasse":
        "une peau grasse",

    "seche":
        "une peau sèche",

    "mixte":
        "une peau mixte",

    "sensible":
        "une peau sensible",

    "normale":
        "une peau normale",
}


SKIN_TYPE_EN = {

    "grasse":
        "oily skin",

    "seche":
        "dry skin",

    "mixte":
        "combination skin",

    "sensible":
        "sensitive skin",

    "normale":
        "normal skin",
}


CONCERN_FR = {

    "acne":
        "l'acné",

    "taches":
        "les taches pigmentaires",

    "rides":
        "les rides",

    "rougeurs":
        "les rougeurs",

    "secheresse":
        "la sécheresse",

    "sensibilite":
        "la sensibilité",

    "deshydratation":
        "la déshydratation",

    "pores":
        "les pores dilatés",
}


CONCERN_EN = {

    "acne":
        "acne",

    "taches":
        "pigmentation and dark spots",

    "rides":
        "wrinkles",

    "rougeurs":
        "redness",

    "secheresse":
        "dryness",

    "sensibilite":
        "sensitivity",

    "deshydratation":
        "dehydration",

    "pores":
        "enlarged pores",
}


def build_skin_profile_text(
    profile,
    language="fr",
):
    """
    Construit une description lisible du profil.
    """

    skin_type = profile.get(
        "skin_type"
    )

    concerns = profile.get(
        "concerns",
        []
    )

    if language == "en":

        base = SKIN_TYPE_EN.get(
            skin_type,
            "your skin"
        )

        if not concerns:
            return base

        labels = [
            CONCERN_EN.get(
                concern,
                concern
            )
            for concern in concerns
        ]

        if len(labels) == 1:

            concern_text = labels[0]

        elif len(labels) == 2:

            concern_text = (
                f"{labels[0]} and "
                f"{labels[1]}"
            )

        else:

            concern_text = (
                ", ".join(
                    labels[:-1]
                )
                + " and "
                + labels[-1]
            )

        return (
            f"{base} with "
            f"{concern_text}"
        )

    # French

    base = SKIN_TYPE_FR.get(
        skin_type,
        "votre peau"
    )

    if not concerns:
        return base

    labels = [
        CONCERN_FR.get(
            concern,
            concern
        )
        for concern in concerns
    ]

    if len(labels) == 1:

        concern_text = labels[0]

    elif len(labels) == 2:

        concern_text = (
            f"{labels[0]} et "
            f"{labels[1]}"
        )

    else:

        concern_text = (
            ", ".join(
                labels[:-1]
            )
            + " et "
            + labels[-1]
        )

    return (
        f"{base} avec "
        f"{concern_text}"
    )


# ============================================================
# 9. ROUTINE STEP DISPLAY
# ============================================================

ROUTINE_STEP_DISPLAY = {

    "nettoyage": {
        "fr": "Nettoyage",
        "en": "Cleansing",
    },

    "serum": {
        "fr": "Sérum",
        "en": "Serum",
    },

    "hydratant": {
        "fr": "Hydratant",
        "en": "Moisturizer",
    },

    "protection_solaire": {
        "fr": "Protection solaire",
        "en": "Sun Protection",
    },

    "demaquillage": {
        "fr": "Démaquillage",
        "en": "Makeup Removal",
    },

    "traitement": {
        "fr": "Traitement",
        "en": "Treatment",
    },

    "reparation": {
        "fr": "Réparation",
        "en": "Repair",
    },
}


def _display_step_name(
    step,
    language="en",
):
    """
    Traduit une étape interne vers la langue d'affichage.
    """

    if not step:
        return step

    key = str(
        step
    ).strip().lower()

    entry = ROUTINE_STEP_DISPLAY.get(
        key
    )

    if entry is None:
        return step

    return entry.get(
        language,
        entry["en"]
    )


# ============================================================
# 10. PRODUCT HELPERS
# ============================================================

def _clean_value(value):
    """
    Nettoie les valeurs None / NaN / chaînes vides.
    """

    if value is None:
        return None

    if isinstance(
        value,
        float
    ) and math.isnan(value):

        return None

    if isinstance(
        value,
        str
    ):

        value = value.strip()

        if value == "":
            return None

        if value.lower() in {
            "nan",
            "none",
            "null",
        }:
            return None

        return value

    return value


def _product_summary(product):
    """
    Convertit un produit en dictionnaire standard.

    Compatible avec:
        - dict
        - pandas Series
    """

    if product is None:
        return None

    return {
        "product_name":
            _clean_value(
                product.get(
                    "product_name"
                )
            ),

        "brand":
            _clean_value(
                product.get(
                    "brand"
                )
            ),

        "product_url":
            _clean_value(
                product.get(
                    "product_url"
                )
            ),

        "price":
            _clean_value(
                product.get(
                    "price_tnd"
                )
            ),

        "routine_step":
            _clean_value(
                product.get(
                    "routine_step"
                )
            ),

        "active_ingredients":
            parse_list(
                product.get(
                    "active_ingredients"
                )
            ),
    }


# ============================================================
# 11. EXPLAINABLE AI
# ============================================================

INGREDIENT_DISPLAY_FR = {

    RETINOL:
        "le rétinol",

    SALICYLIC_ACID:
        "l'acide salicylique",

    GLYCOLIC_ACID:
        "l'acide glycolique",

    NIACINAMIDE:
        "la niacinamide",

    AZELAIC_ACID:
        "l'acide azélaïque",

    VITAMIN_C:
        "la vitamine C",

    CERAMIDES:
        "les céramides",

    PANTHENOL:
        "le panthénol",

    HYALURONIC_ACID:
        "l'acide hyaluronique",
}


INGREDIENT_DISPLAY_EN = {

    RETINOL:
        "retinol",

    SALICYLIC_ACID:
        "salicylic acid",

    GLYCOLIC_ACID:
        "glycolic acid",

    NIACINAMIDE:
        "niacinamide",

    AZELAIC_ACID:
        "azelaic acid",

    VITAMIN_C:
        "vitamin C",

    CERAMIDES:
        "ceramides",

    PANTHENOL:
        "panthenol",

    HYALURONIC_ACID:
        "hyaluronic acid",
}


INGREDIENT_ROLES_FR = {

    RETINOL:
        "le renouvellement cellulaire et l'anti-âge",

    SALICYLIC_ACID:
        "l'exfoliation et l'action anti-imperfections",

    GLYCOLIC_ACID:
        "l'exfoliation et l'unification du teint",

    NIACINAMIDE:
        "la régulation du sébum et l'apaisement",

    AZELAIC_ACID:
        "l'action anti-inflammatoire et éclaircissante",

    VITAMIN_C:
        "l'action antioxydante et éclaircissante",

    CERAMIDES:
        "la réparation de la barrière cutanée",

    PANTHENOL:
        "l'apaisement et l'hydratation",

    HYALURONIC_ACID:
        "l'hydratation",
}


INGREDIENT_ROLES_EN = {

    RETINOL:
        "cell renewal and anti-aging",

    SALICYLIC_ACID:
        "exfoliation and anti-blemish action",

    GLYCOLIC_ACID:
        "exfoliation and skin tone improvement",

    NIACINAMIDE:
        "sebum regulation and soothing",

    AZELAIC_ACID:
        "anti-inflammatory and brightening action",

    VITAMIN_C:
        "antioxidant and brightening action",

    CERAMIDES:
        "skin barrier repair",

    PANTHENOL:
        "soothing and hydration",

    HYALURONIC_ACID:
        "hydration",
}


def explain_product(
    product,
    profile,
    language="fr",
):
    """
    Génère une explication XAI simple à partir des actifs
    connus dans le Knowledge Graph.

    Aucun nouvel actif ou règle médicale n'est créé ici.
    """

    if not product:
        return None

    ingredients = product.get(
        "active_ingredients",
        []
    )

    known = []

    for ingredient in ingredients:

        normalized = normalize_ingredient(
            ingredient
        )

        if (
            normalized
            and normalized in INGREDIENT_ROLES_FR
            and normalized not in known
        ):
            known.append(
                normalized
            )

    if not known:
        return None

    product_name = product.get(
        "product_name"
    )

    if not product_name:
        return None

    if language == "en":

        names = [
            INGREDIENT_DISPLAY_EN[i]
            for i in known
        ]

        # Décompose les rôles composés et supprime les doublons.
        # Exemple :
        #   niacinamide -> "sebum regulation and soothing"
        #   panthenol   -> "soothing and hydration"
        # devient :
        #   "sebum regulation, soothing and hydration"
        role_parts = []

        for ingredient in known:
            role = INGREDIENT_ROLES_EN[ingredient]

            for part in role.split(" and "):
                part = part.strip()

                if part and part.lower() not in {
                    existing.lower()
                    for existing in role_parts
                }:
                    role_parts.append(part)

        roles = role_parts

        if len(names) == 1:

            ingredients_text = names[0]

        else:

            ingredients_text = (
                ", ".join(
                    names[:-1]
                )
                + " and "
                + names[-1]
            )

        if len(roles) == 1:

            roles_text = roles[0]

        else:

            roles_text = (
                ", ".join(
                    roles[:-1]
                )
                + " and "
                + roles[-1]
            )

        return (
            f"It contains "
            f"{ingredients_text}, associated with "
            f"{roles_text}."
        )

    names = [
        INGREDIENT_DISPLAY_FR[i]
        for i in known
    ]

    # Décompose les rôles composés et supprime les doublons.
    # Exemple :
    #   acide hyaluronique -> "l'hydratation"
    #   panthénol          -> "l'apaisement et l'hydratation"
    # devient :
    #   "l'hydratation et l'apaisement"
    role_parts = []

    for ingredient in known:
        role = INGREDIENT_ROLES_FR[ingredient]

        for part in role.split(" et "):
            part = part.strip()

            if part and part.lower() not in {
                existing.lower()
                for existing in role_parts
            }:
                role_parts.append(part)

    roles = role_parts

    if len(names) == 1:

        ingredients_text = names[0]

    else:

        ingredients_text = (
            ", ".join(
                names[:-1]
            )
            + " et "
            + names[-1]
        )

    if len(roles) == 1:

        roles_text = roles[0]

    else:

        roles_text = (
            ", ".join(
                roles[:-1]
            )
            + " et "
            + roles[-1]
        )

    return (
        f"Il contient "
        f"{ingredients_text}, associés à "
        f"{roles_text}."
    )


def _format_routine_response(
    intro,
    routine,
    language="fr",
    profile=None,
):
    """
    Construit l'affichage complet de la routine.

    Le pipeline doit retourner une seule réponse lisible :
    - introduction
    - routine du matin
    - routine du soir
    - produit + lien
    - explication XAI sous chaque produit lorsque disponible
    """

    if language == "en":
        morning_title = "Morning routine"
        evening_title = "Evening routine"
        no_link = "Link: not available"
    else:
        morning_title = "Routine du matin"
        evening_title = "Routine du soir"
        no_link = "Lien : non disponible"

    sections = [intro.strip()]

    for period, title in (
        ("morning", morning_title),
        ("evening", evening_title),
    ):
        products = routine.get(period, {})

        if not products:
            continue

        lines = [f"{title} :"]

        for step, product in products.items():
            if not product:
                continue

            product_name = product.get("product_name")
            if not product_name:
                continue

            lines.append("")
            lines.append(f"{step} :")
            lines.append(f"    {product_name}")

            brand = product.get("brand")
            if brand and str(brand).strip().lower() not in {
                str(product_name).strip().lower()
            }:
                lines.append(f"    Brand: {brand}")

            url = product.get("product_url")
            if url:
                if language == "en":
                    lines.append(f"    Link: {url}")
                else:
                    lines.append(f"    Lien : {url}")
            else:
                lines.append(f"    {no_link}")

            explanation = explain_product(
                product,
                profile or {},
                language,
            )

            if explanation:
                lines.append(f"    Explanation: {explanation}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ============================================================
# 12. HANDLERS
# ============================================================

def _handle_greeting(
    profile,
    entities,
    concerns,
    language,
):
    """
    Greeting.
    """

    return (
        render_response(
            "greeting",
            language=language,
        ),
        None,
    )


def _handle_goodbye(
    profile,
    entities,
    concerns,
    language,
):
    """
    Goodbye.
    """

    return (
        render_response(
            "goodbye",
            language=language,
        ),
        None,
    )


def _handle_profile_info(
    profile,
    entities,
    concerns,
    language,
):
    """
    Affiche les informations connues sur le profil.
    """

    profile_text = (
        build_skin_profile_text(
            profile,
            language,
        )
    )

    context = {
        "skin_profile_text":
            profile_text
    }

    return (
        render_response(
            "profile_info",
            context,
            language,
        ),
        None,
    )


def _handle_routine_request(
    profile,
    entities,
    concerns,
    language,
):
    """
    Construit la routine AM/PM.
    """

    engine_profile = (
        _profile_for_engine(
            profile
        )
    )

    try:

        result = build_routine(
            engine_profile
        )

    except FileNotFoundError:

        if language == "en":

            return (
                "The recommendation model is not "
                "available yet.",
                None,
            )

        return (
            "Le moteur de recommandation "
            "n'est pas encore disponible.",
            None,
        )

    except Exception as exc:

        # On ne masque pas une erreur de programmation.
        raise exc

    morning_result = (
        result.get(
            "morning",
            {}
        )
    )

    evening_result = (
        result.get(
            "evening",
            {}
        )
    )

    # --------------------------------------------------------
    # Important:
    #
    # build_routine() peut retourner:
    #     dict -> product
    #
    # On conserve donc le produit complet.
    # --------------------------------------------------------

    morning = {}

    for step, product in (
        morning_result.items()
        if isinstance(
            morning_result,
            dict
        )
        else []
    ):

        summary = _product_summary(
            product
        )

        morning[
            _display_step_name(
                step,
                language,
            )
        ] = summary

    evening = {}

    for step, product in (
        evening_result.items()
        if isinstance(
            evening_result,
            dict
        )
        else []
    ):

        summary = _product_summary(
            product
        )

        evening[
            _display_step_name(
                step,
                language,
            )
        ] = summary

    data = {
        "routine": {
            "morning": morning,
            "evening": evening,
        },

        "safety":
            result.get(
                "safety"
            ),
    }

    # --------------------------------------------------------
    # Reference product for XAI / next question
    # --------------------------------------------------------

    reference_product = None

    preferred_steps = (
        [
            "Treatment",
            "Serum",
            "Moisturizer",
            "Repair",
            "Cleansing",
            "Sun Protection",
            "Makeup Removal",
        ]
        if language == "en"
        else
        [
            "Traitement",
            "Sérum",
            "Hydratant",
            "Réparation",
            "Nettoyage",
            "Protection solaire",
            "Démaquillage",
        ]
    )

    for step in preferred_steps:

        if step in evening:

            if evening[step] is not None:

                reference_product = (
                    evening[step]
                )

                break

        if step in morning:

            if morning[step] is not None:

                reference_product = (
                    morning[step]
                )

                break

    if reference_product:

        profile[
            "last_product"
        ] = reference_product

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    profile_text = (
        build_skin_profile_text(
            profile,
            language,
        )
    )

    response = render_response(
        "routine_request",
        {
            "skin_profile_text":
                profile_text
        },
        language,
    )

    # Construire la réponse complète : introduction + routine AM/PM
    # + lien + explication sous chaque produit.
    response = _format_routine_response(
        response,
        data["routine"],
        language,
        profile,
    )

    return response, data


def _handle_product_recommendation(
    profile,
    entities,
    concerns,
    language,
):
    """
    Recommande un produit à partir du profil.
    """

    engine_profile = (
        _profile_for_engine(
            profile
        )
    )

    result = recommend_products(
        engine_profile,
        top_n=1,
    )

    recommendations = (
        result.get(
            "recommendations"
        )
    )

    if (
        recommendations is None
        or recommendations.empty
    ):

        if language == "en":

            return (
                "I couldn't find a suitable "
                "product for your current profile.",
                None,
            )

        return (
            "Je n'ai pas trouvé de produit "
            "correspondant à votre profil actuel.",
            None,
        )

    product = (
        recommendations
        .iloc[0]
        .to_dict()
    )

    summary = _product_summary(
        product
    )

    profile[
        "last_product"
    ] = summary

    context = {
        "product_name":
            summary.get(
                "product_name"
            ),

        "brand":
            summary.get(
                "brand"
            ),
    }

    response = render_response(
        "product_recommendation",
        context,
        language,
    )

    # XAI automatique
    explanation = explain_product(
        summary,
        profile,
        language,
    )

    if explanation:

        response = (
            response
            + " "
            + explanation
        )

    return response, {
        "product":
            summary,

        "safety":
            result.get(
                "safety"
            ),
    }


def _handle_routine_explanation(
    profile,
    entities,
    concerns,
    language,
):
    """
    Explique le dernier produit recommandé.
    """

    product = profile.get(
        "last_product"
    )

    context = {}

    if product:

        context[
            "product_name"
        ] = product.get(
            "product_name"
        )

    response = render_response(
        "routine_explanation",
        context,
        language,
    )

    explanation = explain_product(
        product,
        profile,
        language,
    )

    if explanation:

        response = (
            response
            + " "
            + explanation
        )

    return response, None


def _handle_routine_modification(
    profile,
    entities,
    concerns,
    language,
):
    """
    Propose une alternative au dernier produit recommandé.

    Cette logique utilise uniquement les champs déjà présents
    dans le profil et le moteur de recommandation.
    """

    current = profile.get(
        "last_product"
    )

    if not current:

        response = render_response(
            "routine_modification",
            {},
            language,
        )

        return response, None

    engine_profile = (
        _profile_for_engine(
            profile
        )
    )

    current_ingredients = set(
        current.get(
            "active_ingredients",
            []
        )
    )

    existing_avoid = set(
        engine_profile.get(
            "avoid_ingredients",
            []
        )
    )

    engine_profile[
        "avoid_ingredients"
    ] = list(
        existing_avoid
        |
        current_ingredients
    )

    # Filet plus large (20 au lieu de 5) : on a besoin de pouvoir
    # filtrer ensuite sur le routine_step du produit à remplacer,
    # donc il faut davantage de candidats à parcourir.
    result = recommend_products(
        engine_profile,
        top_n=20,
    )

    recommendations = (
        result.get(
            "recommendations"
        )
    )

    if (
        recommendations is None
        or recommendations.empty
    ):

        response = render_response(
            "routine_modification",
            {
                "product_name":
                    current.get(
                        "product_name"
                    )
            },
            language,
        )

        return response, None

    current_name = current.get(
        "product_name"
    )

    current_step = current.get(
        "routine_step"
    )

    alternative = None

    # Passe 1 : même étape de routine que le produit remplacé,
    # et nom différent. C'est ce qui garantit qu'on propose bien
    # une alternative "nettoyant pour nettoyant", "traitement
    # pour traitement", etc., et non un produit d'une tout autre
    # étape.
    for _, row in recommendations.iterrows():

        if row.get("product_name") == current_name:
            continue

        if (
            current_step
            and row.get("routine_step") != current_step
        ):
            continue

        alternative = row.to_dict()
        break

    # Passe 2 (repli) : si aucune alternative n'existe dans la
    # même étape (catalogue restreint), on accepte tout produit
    # différent plutôt que de ne rien proposer du tout.
    if alternative is None:

        for _, row in recommendations.iterrows():

            if row.get("product_name") != current_name:
                alternative = row.to_dict()
                break

    if alternative is None:

        response = render_response(
            "routine_modification",
            {
                "product_name": current_name
            },
            language,
        )

        return response, None

    alternative_summary = (
        _product_summary(
            alternative
        )
    )

    profile[
        "last_product"
    ] = alternative_summary

    intro = render_response(
        "routine_modification",
        {
            "product_name": current_name
        },
        language,
    )

    # On construit explicitement la phrase qui nomme la NOUVELLE
    # alternative : sans ça, le message ne contenait que le nom
    # de l'ANCIEN produit (via le template) plus une explication
    # générique sur les ingrédients, sans jamais dire ce qui est
    # réellement proposé à la place.
    alt_name = alternative_summary.get("product_name")
    alt_brand = alternative_summary.get("brand")
    alt_url = alternative_summary.get("product_url")

    if language == "en":

        alt_line = f"New suggestion: {alt_name}"

        if alt_brand:
            alt_line += f" ({alt_brand})"

        alt_line += "."

        if alt_url:
            alt_line += f" Link: {alt_url}"

    else:

        alt_line = f"Nouvelle suggestion : {alt_name}"

        if alt_brand:
            alt_line += f" ({alt_brand})"

        alt_line += "."

        if alt_url:
            alt_line += f" Lien : {alt_url}"

    response = intro + " " + alt_line

    explanation = explain_product(
        alternative_summary,
        profile,
        language,
    )

    if explanation:

        response = (
            response
            + " "
            + explanation
        )

    return response, {
        "alternative_product":
            alternative_summary
    }


# ============================================================
# 13. INTENT HANDLERS
# ============================================================

INTENT_HANDLERS = {

    "greeting":
        _handle_greeting,

    "goodbye":
        _handle_goodbye,

    "profile_info":
        _handle_profile_info,

    "routine_request":
        _handle_routine_request,

    "product_recommendation":
        _handle_product_recommendation,

    "routine_explanation":
        _handle_routine_explanation,

    "routine_modification":
        _handle_routine_modification,
}


# ============================================================
# 14. AUTO ROUTINE
# ============================================================

AUTO_ROUTINE_INTENTS = {
    "greeting",
    "profile_info",
}


def _is_profile_actionable(profile):
    """
    Le profil est exploitable pour une routine lorsque
    le type de peau et au moins une préoccupation sont connus.
    """

    return (
        bool(
            profile.get(
                "skin_type"
            )
        )
        and
        bool(
            profile.get(
                "concerns"
            )
        )
    )


# ============================================================
# 15. MEDICAL REFERRAL / SAFETY CHECK
# ============================================================

MEDICAL_ALERT_PATTERNS = {

    "acne_severe": [
        "severe acne",
        "acne severe",
        "acne is severe",
        "my acne is severe",
        "acné sévère",
        "acné est sévère",
        "mon acné est sévère",
    ],

    "melanoma_suspicion": [
        "suspected melanoma",
        "suspect melanoma",
        "i suspect melanoma",
        "melanoma",
        "suspicion melanoma",
        "suspicion de melanoma",
        "suspicion de mélanome",
        "mélanome",
        "melanome",
    ],

    "rosacea_severe": [
        "severe rosacea",
        "rosacea severe",
        "rosacée sévère",
        "rosacee severe",
    ],

    "inflammatory_plaques": [
        "significant inflammatory plaques",
        "important inflammatory plaques",
        "inflammatory plaques",
        "plaques inflammatoires importantes",
        "plaques inflammatoires",
    ],

    "significant_hair_loss": [
        "significant hair loss",
        "severe hair loss",
        "important hair loss",
        "chute de cheveux importante",
        "forte chute de cheveux",
        "chute de cheveux sévère",
    ],
}


MEDICAL_ALERT_MESSAGES_EN = {
    "acne_severe":
        "Severe acne detected. We recommend consulting a dermatologist.",
    "melanoma_suspicion":
        "A possible melanoma concern was detected. We recommend consulting a dermatologist.",
    "rosacea_severe":
        "Severe rosacea detected. We recommend consulting a dermatologist.",
    "important_inflammatory_plaques":
        "Significant inflammatory plaques detected. We recommend consulting a dermatologist.",
    "significant_hair_loss":
        "Significant hair loss detected. We recommend consulting a dermatologist.",
}


MEDICAL_ALERT_MESSAGES_FR = {
    "acne_severe":
        "Acné sévère détectée : une consultation médicale est recommandée.",
    "melanoma_suspicion":
        "Une suspicion de mélanome a été détectée : une consultation médicale est recommandée.",
    "rosacea_severe":
        "Rosacée sévère détectée : une consultation médicale est recommandée.",
    "important_inflammatory_plaques":
        "Plaques inflammatoires importantes détectées : une consultation médicale est recommandée.",
    "significant_hair_loss":
        "Chute de cheveux importante détectée : une consultation médicale est recommandée.",
}


def _apply_medical_flags_from_text(
    profile,
    text,
):
    """
    Détecte les cinq situations d'orientation médicale
    définies dans safety_rules.py à partir du message utilisateur.

    Cette étape ne pose aucun diagnostic. Elle prépare simplement
    les indicateurs attendus par detect_medical_alerts().
    """

    normalized = (
        text
        .lower()
        .strip()
    )

    for reason, patterns in MEDICAL_ALERT_PATTERNS.items():

        if any(
            pattern in normalized
            for pattern in patterns
        ):

            if reason == "acne_severe":
                profile["acne_severity"] = "severe"

            elif reason == "melanoma_suspicion":
                profile["melanoma_suspicion"] = True

            elif reason == "rosacea_severe":
                profile["rosacea_severity"] = "severe"

            elif reason == "inflammatory_plaques":
                profile["inflammatory_plaques"] = True

            elif reason == "significant_hair_loss":
                profile["significant_hair_loss"] = True


def _medical_alert_response(
    alert,
    language,
):
    """
    Retourne le message médical dans la langue détectée.
    """

    reason = alert.get(
        "reason"
    )

    if language == "en":
        return MEDICAL_ALERT_MESSAGES_EN.get(
            reason,
            "A medical concern was detected. We recommend consulting a dermatologist.",
        )

    return MEDICAL_ALERT_MESSAGES_FR.get(
        reason,
        "Une situation nécessitant un avis médical a été détectée : une consultation médicale est recommandée.",
    )


def _medical_alert_data(
    alerts,
):
    return [
        {
            "reason": alert.get("reason"),
            "recommend_consultation": alert.get(
                "recommend_consultation",
                False,
            ),
            "renyou_appointment": alert.get(
                "renyou_appointment",
                False,
            ),
        }
        for alert in alerts
    ]


# ============================================================
# 16. UNSUPPORTED MESSAGE VALIDATION
# ============================================================

# --------------------------------------------------------
# CORRECTIF (bug routine_modification / "treatment"):
#
# Avant correctif, cette liste était maintenue entièrement à la
# main et ne contenait ni "treatment" ni "traitement" ni d'autres
# libellés d'étapes de routine. Un message comme
# "I'd like a different treatment" ne matchait donc aucun mot-clé,
# _contains_skincare_signal() renvoyait False,
# _validate_intent_prediction() rejetait la prédiction ML
# (pourtant correcte: routine_modification) et le pipeline
# retombait sur la réponse générique "unsupported".
#
# Correctif en 2 parties :
#   1. Ajout explicite des mots manquants (treatment/traitement,
#      repair/réparation, cleansing/démaquillage, plan, different...).
#   2. Fusion dynamique avec les libellés déjà définis dans
#      ROUTINE_STEP_DISPLAY (section 9) et CONCERN_KEYWORDS
#      (section 4), pour qu'un futur ajout d'étape ou de concern
#      n'oublie plus de mettre à jour cette whitelist séparément.
# --------------------------------------------------------

SKINCARE_DOMAIN_KEYWORDS = {
    "skin", "peau", "skincare", "skin-care",
    "routine", "product", "products", "produit", "produits",
    "cream", "crème", "creme", "serum", "sérum",
    "cleanser", "nettoyant", "moisturizer", "hydratant",
    "sunscreen", "solaire", "spf",
    "acne", "acné", "pimples", "boutons", "imperfections",
    "pores", "redness", "rougeurs", "rosacea", "rosacée",
    "dryness", "sécheresse", "dehydration", "déshydratation",
    "sensitive", "sensible", "oily", "grasse", "dry", "sèche",
    "combination", "mixte", "normal", "normale",
    "niacinamide", "retinol", "rétinol", "salicylic",
    "salicylique", "hyaluronic", "hyaluronique",
    "glycolic", "glycolique", "azelaic", "azélaïque",
    "vitamin c", "vitamine c", "ceramides", "céramides",
    "panthenol", "panthénol",
    "recommend", "recommendation", "recommander", "recommandation",
    "explain", "why", "pourquoi", "expliquer",
    "modify", "modifier", "alternative",

    # --- Ajouts du correctif ---
    "treatment", "traitement",
    "repair", "réparation", "reparation",
    "cleansing", "démaquillage", "demaquillage", "makeup removal",
    "plan", "different", "différent", "autre",
}

# Fusion dynamique: tout libellé de ROUTINE_STEP_DISPLAY (fr/en)
# et tout mot-clé de CONCERN_KEYWORDS vient compléter la liste
# ci-dessus automatiquement, sans duplication manuelle à entretenir.
SKINCARE_DOMAIN_KEYWORDS |= {
    label.lower()
    for entry in ROUTINE_STEP_DISPLAY.values()
    for label in entry.values()
}

SKINCARE_DOMAIN_KEYWORDS |= {
    keyword.lower()
    for keywords in CONCERN_KEYWORDS.values()
    for keyword in keywords
}


def _contains_skincare_signal(text):
    normalized = text.lower().strip()
    return any(
        keyword in normalized
        for keyword in SKINCARE_DOMAIN_KEYWORDS
    )


def _looks_like_greeting_or_goodbye(text, intent):
    normalized = text.lower().strip()
    words = set(re.findall(r"[a-zA-ZÀ-ÿ'-]+", normalized))

    if intent == "greeting":
        return any(word in words for word in {
            "hi", "hello", "hey", "bonjour", "salut",
            "bonsoir", "coucou",
        })

    if intent == "goodbye":
        return any(word in words for word in {
            "bye", "goodbye", "take", "care", "see",
            "revoir", "bientôt",
        })

    return False


def _validate_intent_prediction(text, intent):
    """
    Vérifie qu'une prédiction ML reste cohérente avec le domaine.

    On ne crée pas d'intent `unknown` dans le dataset : un message
    hors domaine est simplement traité comme message non supporté.
    """

    if intent not in INTENT_HANDLERS:
        return False

    # greeting/goodbye doivent avoir un signal lexical explicite
    # lorsqu'ils n'ont pas été reconnus par detect_exact_intent().
    if intent in {"greeting", "goodbye"}:
        return _looks_like_greeting_or_goodbye(
            text,
            intent,
        )

    # Tous les autres intents doivent être liés au domaine skincare.
    return _contains_skincare_signal(text)


def _unsupported_response(language):
    if language == "en":
        return (
            "I'm specialized in skincare, skincare routines, "
            "and product recommendations. Please ask me something "
            "related to your skin or skincare."
        )

    return (
        "Je suis spécialisé dans les soins de la peau, les routines "
        "skincare et les recommandations de produits. Posez-moi une "
        "question en rapport avec votre peau ou vos soins."
    )


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def process_message(
    text,
    session_id=None,
    language=None,
):
    """
    Traite un message utilisateur de bout en bout.
    """

    if not isinstance(
        text,
        str,
    ):
        raise ValueError(
            "Le message doit être une chaîne."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Le message ne peut pas être vide."
        )

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    if session_id is None:

        session_id = str(
            uuid.uuid4()
        )

    profile = get_session(
        session_id
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    if language not in {"fr", "en"}:
        language = detect_language(text)

    # --------------------------------------------------------
    # Exact intent
    # --------------------------------------------------------

    exact_intent, exact_language = (
        detect_exact_intent(
            text
        )
    )

    if exact_intent is not None:

        intent = exact_intent

        language = (
            exact_language
        )

        confidence = 1.0

    else:

        if _is_explicit_routine_request(text):
            intent = "routine_request"
            confidence = 1.0
        else:
            # ----------------------------------------------------
            # ML intent classification
            # ----------------------------------------------------

            intent_result = predict_intent(
                text
            )

            intent = intent_result[
                "intent"
            ]

            confidence = intent_result.get(
                "confidence"
            )

            if confidence is None:
                confidence = 0.0

        # Le classifieur peut forcer une intention erronée sur un
        # message hors domaine. On valide donc sa prédiction au
        # niveau pipeline, sans ajouter `unknown` au dataset.
        if not _validate_intent_prediction(
            text,
            intent,
        ):
            intent = "unsupported"

    # --------------------------------------------------------
    # Entity extraction
    # --------------------------------------------------------

    entities = extract_entities(
        text
    )

    # --------------------------------------------------------
    # Concern extraction
    # --------------------------------------------------------

    concerns = extract_concerns(
        text
    )

    # --------------------------------------------------------
    # Profile update
    # --------------------------------------------------------

    update_profile_from_entities(
        profile,
        entities,
        concerns,
    )

    # --------------------------------------------------------
    # Medical referral / safety check
    # --------------------------------------------------------
    # This check has priority over normal chatbot actions.
    # It does not diagnose the user; it only detects predefined
    # high-risk signals and recommends medical consultation.

    _apply_medical_flags_from_text(
        profile,
        text,
    )

    medical_alerts = detect_medical_alerts(
        profile
    )

    # --------------------------------------------------------
    # Handler
    # --------------------------------------------------------

    auto_generated = False

    if medical_alerts:

        response = _medical_alert_response(
            medical_alerts[0],
            language,
        )

        data = {
            "medical_alerts":
                _medical_alert_data(
                    medical_alerts
                ),
            "recommend_consultation": True,
        }

    else:

    # --------------------------------------------------------
    # Handler
    #
    # A profile update alone must NOT automatically generate
    # a skincare routine. The user must explicitly request
    # a routine. This keeps messages such as:
    #     "I have dry and sensitive skin"
    # as profile information only.
    #
    # Explicit requests such as:
    #     "Build me a routine"
    #     "Give me my routine"
    # are still handled by routine_request.
    # --------------------------------------------------------

        handler = INTENT_HANDLERS.get(
            intent
        )

        if handler is None:

            response = _unsupported_response(
                language
            )

            data = None

        else:

            response, data = handler(
                profile,
                entities,
                concerns,
                language,
            )

    # --------------------------------------------------------
    # Goodbye resets the session
    # --------------------------------------------------------

    if intent == "goodbye":

        reset_session(
            session_id
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "session_id":
            session_id,

        "intent":
            intent,

        "language":
            language,

        "auto_generated":
            auto_generated,

        "confidence":
            confidence,

        "entities":
            entities,

        "concerns":
            concerns,

        "profile": {

            "skin_type":
                profile.get(
                    "skin_type"
                ),

            "concerns":
                profile.get(
                    "concerns",
                    []
                ),

            "preferred_ingredients":
                profile.get(
                    "preferred_ingredients",
                    []
                ),

            "last_product":
                profile.get(
                    "last_product"
                ),
        },

        "response":
            response,

        "medical_alerts":
            _medical_alert_data(
                medical_alerts
            ),

        "data":
            data,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    session_id = None

    tests = [

        "hi",

        "hello",

        "hey",

        "Bonjour",

        "I have oily skin and acne",

        "I have severe acne",

        "I have severe rosacea",

        "I have significant hair loss",

        "I have inflammatory plaques",

        "I suspect melanoma",

        "I have oily skin",

        "Can you recommend a product?",

        "Give me my routine",

        "Why this product?",

        "I'd like a different treatment",

        "goodbye",
    ]

    for message in tests:

        print()
        print(
            "=" * 70
        )

        print(
            "USER:",
            message,
        )

        result = process_message(
            message,
            session_id,
        )

        session_id = result[
            "session_id"
        ]

        print(
            "LANGUAGE:",
            result[
                "language"
            ],
        )

        print(
            "INTENT:",
            result[
                "intent"
            ],
        )

        print(
            "CONFIDENCE:",
            round(
                result[
                    "confidence"
                ],
                4,
            ),
        )

        print(
            "ENTITIES:",
            result[
                "entities"
            ],
        )

        print(
            "CONCERNS:",
            result[
                "concerns"
            ],
        )

        print(
            "PROFILE:",
            result[
                "profile"
            ],
        )

        print(
            "BOT:",
            result[
                "response"
            ],
        )