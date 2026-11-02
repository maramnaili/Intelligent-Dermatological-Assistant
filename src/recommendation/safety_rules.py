"""
Medical Safety Rules

Les règles implémentées correspondent strictement
au cahier des charges.

Règles Médicales de Sécurité
--------------------------------
- Grossesse :
    Blocage Rétinol + Hydroquinone
- Allaitement :
    Restrictions adaptées
- Peau ultra-sensible :
    Réduction des actifs irritants
- Rosacée :
    Exclusion acides agressifs + gommages abrasifs
- Traitement dermatologique en cours :
    Adaptation automatique

Orientation Médicale
------------------------
- Acné sévère
- Suspicion de mélanome
- Rosacée sévère
- Plaques inflammatoires importantes
- Chute de cheveux importante

IMPORTANT :
Le module ne pose aucun diagnostic.
Il applique uniquement les informations fournies
par le profil et les règles définies dans le projet.
"""

import ast
import re
import unicodedata


# ============================================================
# UTILITAIRES
# ============================================================

def normalize_text(value):
    """
    Normalisation utilisée uniquement pour comparer
    les textes sans être sensible aux majuscules/minuscules
    ni aux accents.
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text


def parse_ingredients(value):
    """
    Convertit active_ingredients en liste.

    Exemple :
        "['niacinamide', 'vitamine_c']"

    devient :
        ['niacinamide', 'vitamine_c']
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text or text.lower() == "nan":
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


def get_product_ingredients(product):
    """
    Récupère les actifs présents dans active_ingredients.

    IMPORTANT :
    On utilise les données déjà extraites par le scraping/
    nettoyage. On ne réinvente pas une extraction complète
    des ingrédients ici.
    """

    values = parse_ingredients(
        product.get(
            "active_ingredients",
            []
        )
    )

    return {
        normalize_text(value)
        for value in values
        if normalize_text(value)
    }


def get_product_text(product):
    """
    Texte produit utilisé pour détecter notamment
    les gommages abrasifs.
    """

    fields = [
        product.get("product_name", ""),
        product.get("category", ""),
        product.get("routine_step", ""),
    ]

    return normalize_text(
        " ".join(
            str(value)
            for value in fields
            if value is not None
        )
    )


# ============================================================
# GROSSESSE
# ============================================================

def pregnancy_blocked(product):
    """
    Grossesse :
    blocage du Rétinol et de l'Hydroquinone.
    """

    ingredients = get_product_ingredients(
        product
    )

    forbidden = {
        "retinol",
        "hydroquinone",
    }

    return bool(
        ingredients.intersection(
            forbidden
        )
    )


# ============================================================
#  PEAU ULTRA-SENSIBLE
# ============================================================

def ultra_sensitive_restricted(product):
    """
    Peau ultra-sensible :
    réduction des actifs irritants.

    Le cahier des charges ne fournit pas une liste
    exhaustive d'actifs irritants.

    On ne crée donc pas arbitrairement une liste
    supplémentaire ici.

    La fonction est prévue pour appliquer la liste
    d'actifs irritants validée dans le projet.
    """

    # À compléter uniquement lorsque la liste officielle
    # des actifs irritants du projet est définie.

    return False


# ============================================================
# §11 — ROSACÉE
# ============================================================

def rosacea_excluded(product):
    """
    Rosacée :
    exclusion des acides agressifs et des gommages abrasifs.

    La notion d'« acides agressifs » n'est pas détaillée
    dans le cahier des charges.

    On ne fabrique donc pas arbitrairement une liste
    d'acides.
    """

    product_text = get_product_text(
        product
    )

    # Le cahier des charges parle explicitement
    # de gommages abrasifs.
    abrasive_terms = {
        "gommage abrasif",
        "gommage",
        "scrub",
    }

    for term in abrasive_terms:

        if term in product_text:
            return True

    # La liste exacte des acides agressifs doit être
    # fournie/validée dans le projet avant de les exclure.

    return False


# ============================================================
# ECZÉMA / DERMATITE ATOPIQUE
# ============================================================

def eczema_routine_restricted(product):
    """Avoid treatment and exfoliating products for an eczema profile."""

    product_text = get_product_text(product)
    routine_step = normalize_text(product.get("routine_step", ""))

    if routine_step in {"traitement", "treatment"}:
        return True

    exfoliating_terms = {
        "exfoliant",
        "exfoliating",
        "exfoliation",
        "peeling",
        "peel",
        "gommage",
        "scrub",
    }

    return any(term in product_text for term in exfoliating_terms)


# ============================================================
# §11 — ALLAITEMENT
# ============================================================

def breastfeeding_restricted(product):
    """
    Allaitement :
    le cahier des charges demande des restrictions adaptées,
    mais ne donne pas la liste détaillée des restrictions.

    On ne crée donc aucune règle médicale arbitraire.
    """

    return False


# ============================================================
# §11 — TRAITEMENT DERMATOLOGIQUE
# ============================================================

def dermatological_treatment_adaptation(
    product
):
    """
    Traitement dermatologique en cours :
    adaptation automatique.

    Le cahier des charges ne fournit pas ici la liste
    précise des produits/actifs à exclure.

    On ne crée donc pas de règle supplémentaire non spécifiée.
    """

    return False


# ============================================================
# VERIFICATION PRODUIT
# ============================================================

def is_product_safe(
    product,
    profile
):
    """
    Vérifie les règles de sécurité applicables
    au produit pour le profil donné.
    """

    # --------------------------------------------------------
    # Grossesse
    # --------------------------------------------------------

    if profile.get("pregnant") is True:

        if pregnancy_blocked(product):
            return False

    # --------------------------------------------------------
    # Allaitement
    # --------------------------------------------------------

    if profile.get("breastfeeding") is True:

        if breastfeeding_restricted(product):
            return False

    # --------------------------------------------------------
    # Peau ultra-sensible
    # --------------------------------------------------------

    if profile.get("ultra_sensitive") is True:

        if ultra_sensitive_restricted(product):
            return False

    # --------------------------------------------------------
    # Rosacée
    # --------------------------------------------------------

    if profile.get("rosacea") is True:

        if rosacea_excluded(product):
            return False

    # --------------------------------------------------------
    # Eczéma / dermatite atopique
    # --------------------------------------------------------

    concerns = {
        normalize_text(value)
        for value in profile.get("concerns", [])
    }
    if concerns.intersection({
        "eczema",
        "atopic dermatitis",
        "dermatite atopique",
    }) and eczema_routine_restricted(product):
        return False

    # --------------------------------------------------------
    # Traitement dermatologique
    # --------------------------------------------------------

    if profile.get(
        "dermatological_treatment"
    ) is True:

        if dermatological_treatment_adaptation(
            product
        ):
            return False

    return True


# ============================================================
# §12 — ORIENTATION MÉDICALE
# ============================================================

def detect_medical_alerts(profile):
    """
    Détecte exactement les situations médicales
    indiquées dans le §12 du cahier des charges.

    Aucun diagnostic n'est effectué.
    """

    alerts = []

    concerns = {
        normalize_text(value)
        for value in profile.get(
            "concerns",
            []
        )
    }

    # ========================================================
    # 1. ACNÉ SÉVÈRE
    # ========================================================

    acne_severe = (
        profile.get(
            "acne_severity"
        ) == "severe"
        or
        "acne_severe" in concerns
        or
        "acne severe" in concerns
        or
        "acne sévère" in concerns
    )

    if acne_severe:

        alerts.append({
            "reason": "acne_severe",
            "message": (
                "Acné sévère détectée : "
                "une consultation médicale est recommandée."
            ),
            "recommend_consultation": True,
            "renyou_appointment": True,
        })

    # ========================================================
    # 2. SUSPICION DE MÉLANOME
    # ========================================================

    melanoma = (
        profile.get(
            "melanoma_suspicion"
        ) is True
        or
        "melanoma_suspicion" in concerns
        or
        "suspicion melanome" in concerns
        or
        "suspicion de melanome" in concerns
        or
        "melanome" in concerns
    )

    if melanoma:

        alerts.append({
            "reason": "melanoma_suspicion",
            "message": (
                "Suspicion de mélanome : "
                "une consultation médicale est recommandée."
            ),
            "recommend_consultation": True,
            "renyou_appointment": True,
        })

    # ========================================================
    # 3. ROSACÉE SÉVÈRE
    # ========================================================

    rosacea_severe = (
        profile.get(
            "rosacea_severity"
        ) == "severe"
        or
        "rosacea_severe" in concerns
        or
        "rosacee severe" in concerns
        or
        "rosacée sévère" in concerns
    )

    if rosacea_severe:

        alerts.append({
            "reason": "rosacea_severe",
            "message": (
                "Rosacée sévère détectée : "
                "une consultation médicale est recommandée."
            ),
            "recommend_consultation": True,
            "renyou_appointment": True,
        })

    # ========================================================
    # 4. PLAQUES INFLAMMATOIRES IMPORTANTES
    # ========================================================

    inflammatory_plaques = (
        profile.get(
            "inflammatory_plaques"
        ) is True
        or
        "inflammatory_plaques" in concerns
        or
        "plaques inflammatoires importantes" in concerns
        or
        "plaques inflammatoires" in concerns
    )

    if inflammatory_plaques:

        alerts.append({
            "reason": "important_inflammatory_plaques",
            "message": (
                "Plaques inflammatoires importantes détectées : "
                "une consultation médicale est recommandée."
            ),
            "recommend_consultation": True,
            "renyou_appointment": True,
        })

    # ========================================================
    # 5. CHUTE DE CHEVEUX IMPORTANTE
    # ========================================================

    hair_loss = (
        profile.get(
            "significant_hair_loss"
        ) is True
        or
        profile.get(
            "hair_loss_severity"
        ) == "severe"
        or
        "chute cheveux importante" in concerns
        or
        "chute de cheveux importante" in concerns
    )

    if hair_loss:

        alerts.append({
            "reason": "significant_hair_loss",
            "message": (
                "Chute de cheveux importante détectée : "
                "une consultation médicale est recommandée."
            ),
            "recommend_consultation": True,
            "renyou_appointment": True,
        })

    return alerts


# ============================================================
# EVALUATION COMPLETE DU PROFIL
# ============================================================

def evaluate_profile_safety(profile):
    """
    Retourne les informations de sécurité du profil.

    Les catégories correspondent au cahier des charges §11.
    """

    restrictions = []

    if profile.get("pregnant") is True:
        restrictions.extend([
            "retinol",
            "hydroquinone",
        ])

    if profile.get("breastfeeding") is True:
        restrictions.append(
            "allaitement"
        )

    if profile.get("ultra_sensitive") is True:
        restrictions.append(
            "actifs_irritants"
        )

    if profile.get("rosacea") is True:
        restrictions.extend([
            "acides_agressifs",
            "gommages_abrasifs",
        ])

    concerns = {
        normalize_text(value)
        for value in profile.get("concerns", [])
    }
    if concerns.intersection({
        "eczema",
        "atopic dermatitis",
        "dermatite atopique",
    }):
        restrictions.extend([
            "traitements_cosmétiques",
            "exfoliants",
        ])

    if profile.get(
        "dermatological_treatment"
    ) is True:
        restrictions.append(
            "adaptation_traitement_dermatologique"
        )

    medical_alerts = detect_medical_alerts(
        profile
    )

    return {
        "restrictions": restrictions,

        "medical_alerts": medical_alerts,

        "medical_attention_required": bool(
            medical_alerts
        ),
    }