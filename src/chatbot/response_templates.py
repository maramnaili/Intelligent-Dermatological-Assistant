"""
AI-S2-09 — Response Templates

7 intents actuellement utilisés par le chatbot.
Chaque intent possède 4 templates en français
et 4 templates en anglais.
"""

import random


RESPONSE_TEMPLATES = {

    # ========================================================
    # GREETING
    # ========================================================

    "greeting": {

        "fr": [
            "Bonjour ! Je suis là pour vous accompagner dans vos soins de la peau. Que souhaitez-vous savoir ?",
            "Bonjour ! Comment puis-je vous aider avec votre routine skincare ?",
            "Ravie de vous aider ! Que souhaitez-vous savoir sur vos soins de la peau ?",
            "Bonjour ! Je peux vous aider avec votre profil cutané, votre routine et vos recommandations de produits.",
        ],

        "en": [
            "Hello! I'm here to help you with your skincare. What would you like to know?",
            "Hi! How can I help you with your skincare routine?",
            "Nice to meet you! What would you like to know about your skincare?",
            "Hello! I can help you with your skin profile, routine, and product recommendations.",
        ],
    },


    # ========================================================
    # ROUTINE REQUEST
    # ========================================================

    "routine_request": {

        "fr": [
            "Avec plaisir ! Je peux construire une routine personnalisée pour {skin_profile_text}.",
            "Bien sûr ! Je vais vous proposer une routine adaptée à {skin_profile_text}.",
            "Voici une routine personnalisée en fonction de {skin_profile_text}.",
            "Je vais vous aider à construire une routine adaptée à votre profil.",
        ],

        "en": [
            "Of course! I can build a personalized routine for {skin_profile_text}.",
            "Sure! I'll suggest a skincare routine adapted to {skin_profile_text}.",
            "Here is a personalized routine based on {skin_profile_text}.",
            "I'll help you build a skincare routine adapted to your profile.",
        ],
    },


    # ========================================================
    # PROFILE INFO
    # ========================================================

    "profile_info": {

        "fr": [
            "Merci pour ces précisions ! Je vais tenir compte de votre profil cutané : {skin_profile_text}.",
            "D'accord ! Je prends en compte {skin_profile_text} pour personnaliser vos recommandations.",
            "Votre profil a bien été pris en compte : {skin_profile_text}.",
            "Merci ! Ces informations vont m'aider à personnaliser vos recommandations.",
        ],

        "en": [
            "Thanks for sharing! I'll take your skin profile into account: {skin_profile_text}.",
            "Got it! I'll use {skin_profile_text} to personalize your recommendations.",
            "Your skin profile has been taken into account: {skin_profile_text}.",
            "Thanks! This information will help me personalize your recommendations.",
        ],
    },


    # ========================================================
    # PRODUCT RECOMMENDATION
    # ========================================================

    "product_recommendation": {

        "fr": [
            "Je vous recommande {product_name} de {brand}.",
            "Le produit que je vous recommande est {product_name} de {brand}.",
            "Pour votre profil, je vous recommande {product_name} de {brand}.",
            "Voici une recommandation adaptée à votre profil : {product_name} de {brand}.",
            "Je vous recommande {product_name}.",
            "Voici une recommandation adaptée à votre profil : {product_name}.",
        ],

        "en": [
            "I recommend {product_name} by {brand}.",
            "The product I recommend for you is {product_name} by {brand}.",
            "Based on your profile, I recommend {product_name} by {brand}.",
            "Here is a recommendation adapted to your profile: {product_name} by {brand}.",
            "I recommend {product_name}.",
            "Here is a recommendation adapted to your profile: {product_name}.",
        ],
    },


    # ========================================================
    # ROUTINE EXPLANATION
    # ========================================================

    "routine_explanation": {

        "fr": [
            "Je peux vous expliquer pourquoi ce produit correspond à votre profil : {product_name}.",
            "Ce produit a été sélectionné en fonction de votre profil et de vos besoins.",
            "La recommandation de {product_name} est liée aux caractéristiques de votre profil.",
            "Je peux vous expliquer le lien entre ce produit et votre profil.",
        ],

        "en": [
            "I can explain why this product matches your profile: {product_name}.",
            "This product was selected based on your profile and skincare needs.",
            "The recommendation of {product_name} is related to your skin profile and needs.",
            "I can explain how this product relates to your profile.",
        ],
    },


    # ========================================================
    # ROUTINE MODIFICATION
    # ========================================================

    "routine_modification": {

        "fr": [
            "Bien sûr ! Je peux rechercher une alternative à {product_name}.",
            "D'accord ! Je vais chercher une autre option pour remplacer {product_name}.",
            "Je vais vous proposer une alternative à {product_name}.",
            "Je peux rechercher un autre produit pour cette étape.",
        ],

        "en": [
            "Of course! I can look for an alternative to {product_name}.",
            "Sure! I'll look for another option to replace {product_name}.",
            "I'll suggest an alternative to {product_name}.",
            "I can look for another product for this step.",
        ],
    },


    # ========================================================
    # GOODBYE
    # ========================================================

    "goodbye": {

        "fr": [
            "Au revoir ! Prenez soin de votre peau.",
            "À bientôt ! Prenez soin de vous.",
            "Avec plaisir ! Prenez soin de votre peau et à bientôt !",
            "Merci pour votre visite. À bientôt !",
        ],

        "en": [
            "Goodbye! Take care of your skin.",
            "See you soon! Take care.",
            "You're welcome! Take care of your skin and see you soon!",
            "Thanks for stopping by. See you soon!",
        ],
    },
}


# ============================================================
# RENDER
# ============================================================

def render_response(
    intent,
    context=None,
    language="fr"
):

    context = context or {}

    # Retire les clés dont la valeur est vide ou None : un champ
    # comme "brand" peut arriver à None (marque inconnue dans le
    # dataset). Si on le laisse dans le contexte, str.format()
    # l'interpole littéralement ("... by None.") au lieu de
    # simplement échouer. En le retirant, un template qui exige
    # {brand} lève KeyError et on passe à une variante qui ne
    # l'exige pas (cf. product_recommendation, qui a maintenant
    # des variantes sans marque pour ce cas précis).
    context = {
        key: value
        for key, value in context.items()
        if value not in (None, "")
    }

    if language not in {
        "fr",
        "en",
    }:
        language = "fr"

    intent_templates = (
        RESPONSE_TEMPLATES.get(
            intent
        )
    )

    if not intent_templates:
        return (
            "Je suis spécialisé dans les soins de la peau, "
            "les routines skincare et les recommandations de produits."
            "Pouvez-vous me poser une question concernant ces sujets ?"
            if language == "fr"
            else
            "I'm specialized in skincare, "
            "skincare routines, and product recommendations."
            "Can you ask me a question about these topics?"
        )

    templates = intent_templates.get(
        language,
        intent_templates.get("fr", [])
    )

    candidates = list(
        templates
    )

    random.shuffle(
        candidates
    )

    for template in candidates:

        try:

            return template.format(
                **context
            )

        except KeyError:

            continue

    return templates[-1]