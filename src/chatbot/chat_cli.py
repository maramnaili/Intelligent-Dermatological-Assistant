"""
CLI de test du chatbot RenYou.

AI-S2-10 — Chat Pipeline

La routine affiche, dans la langue détectée par le pipeline
(result["language"], "fr" ou "en") :

    Morning / Matin:
        Cleansing / Nettoyage
        Serum / Sérum
        Moisturizer / Hydratant
        Sun Protection / Protection solaire

    Evening / Soir:
        Makeup Removal / Démaquillage
        Cleansing / Nettoyage
        Treatment / Traitement
        Repair / Réparation

Ces libellés doivent rester synchronisés avec
ROUTINE_STEP_DISPLAY dans src/chatbot/chat_pipeline.py : c'est
ce module qui traduit routine_step (valeurs françaises du
dataset) vers ces mêmes libellés, dans la langue de la
conversation.

Le prix n'est pas affiché.
Le lien produit provenant de `product_url`
est affiché lorsqu'il est disponible.
"""

from src.chatbot.chat_pipeline import (
    process_message,
)


# ============================================================
# TEXTS (FR / EN)
# ============================================================

TEXTS = {

    "en": {
        "morning_header": "Morning routine:",
        "evening_header": "Evening routine:",
        "no_product": "(No recommended product available)",
        "link_label": "Link:",
        "link_unavailable": "(not available)",
        "recommended_product": "Recommended product:",
        "alternative_product": "Alternative product:",
    },

    "fr": {
        "morning_header": "Routine du matin :",
        "evening_header": "Routine du soir :",
        "no_product": "(Aucun produit recommandé disponible)",
        "link_label": "Lien :",
        "link_unavailable": "(non disponible)",
        "recommended_product": "Produit recommandé :",
        "alternative_product": "Produit alternatif :",
    },
}


# Ordre d'affichage des étapes, par langue. Doit correspondre
# exactement aux libellés produits par ROUTINE_STEP_DISPLAY
# côté chat_pipeline.py.
ROUTINE_STEPS = {

    "en": {
        "morning": [
            "Cleansing",
            "Serum",
            "Moisturizer",
            "Sun Protection",
        ],
        "evening": [
            "Makeup Removal",
            "Cleansing",
            "Treatment",
            "Repair",
        ],
    },

    "fr": {
        "morning": [
            "Nettoyage",
            "Sérum",
            "Hydratant",
            "Protection solaire",
        ],
        "evening": [
            "Démaquillage",
            "Nettoyage",
            "Traitement",
            "Réparation",
        ],
    },
}


def _texts(language):
    return TEXTS.get(
        language,
        TEXTS["en"]
    )


def _steps(language):
    return ROUTINE_STEPS.get(
        language,
        ROUTINE_STEPS["en"]
    )


# ============================================================
# DISPLAY ROUTINE
# ============================================================

def _display_product(
    product,
    language="en"
):

    texts = _texts(language)

    if not product:

        print(
            f"    {texts['no_product']}"
        )

        return

    product_name = product.get(
        "product_name"
    )

    brand = product.get(
        "brand"
    )

    product_url = product.get(
        "product_url"
    )

    # --------------------------------------------------------
    # Product name
    # --------------------------------------------------------

    if brand:

        print(
            f"    {product_name} "
            f"({brand})"
        )

    else:

        print(
            f"    {product_name}"
        )

    # --------------------------------------------------------
    # Product URL
    # --------------------------------------------------------

    if product_url:

        print(
            f"    {texts['link_label']} "
            f"{product_url}"
        )

    else:

        print(
            f"    {texts['link_label']} "
            f"{texts['link_unavailable']}"
        )


def display_routine(
    routine,
    language="en"
):

    texts = _texts(language)
    steps = _steps(language)

    print()
    print(
        f"  {texts['morning_header']}"
    )

    morning = routine.get(
        "morning",
        {}
    )

    for step in steps["morning"]:

        print(
            f"    {step}:"
        )

        _display_product(
            morning.get(
                step
            ),
            language
        )

    print()

    print(
        f"  {texts['evening_header']}"
    )

    evening = routine.get(
        "evening",
        {}
    )

    for step in steps["evening"]:

        print(
            f"    {step}:"
        )

        _display_product(
            evening.get(
                step
            ),
            language
        )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result
):
    """
    Affiche uniquement la réponse finale produite par le pipeline.

    Le pipeline (chat_pipeline.py) construit déjà la réponse complète,
    notamment la routine avec les explications de chaque produit.
    Le CLI ne doit donc pas réafficher result["data"]["routine"],
    result["data"]["product"] ou result["data"]["alternative_product"],
    sinon le contenu apparaît deux fois.
    """

    print()
    print(
        f"Assistant : "
        f"{result['response']}"
    )


# ============================================================
# HELP
# ============================================================

def display_help():

    print()
    print(
        "Commandes disponibles :"
    )

    print(
        "  /reset -> réinitialise la conversation"
    )

    print(
        "  /debug -> active/désactive le debug"
    )

    print(
        "  /help -> affiche cette aide"
    )

    print(
        "  /quit -> quitte le programme"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=== RenYou AI Skin Assistant — CLI de test ==="
    )

    print(
        "Mode direct — aucun serveur Flask requis."
    )

    display_help()

    session_id = None

    debug = False

    while True:

        try:

            text = input(
                "Vous : "
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            print(
                "Au revoir !"
            )

            break

        text = text.strip()

        if not text:
            continue

        # ----------------------------------------------------
        # COMMANDS
        # ----------------------------------------------------

        if text == "/quit":

            print(
                "Au revoir !"
            )

            break

        if text == "/help":

            display_help()
            continue

        if text == "/reset":

            session_id = None

            print(
                "Conversation réinitialisée."
            )

            continue

        if text == "/debug":

            debug = not debug

            print(
                f"Debug : "
                f"{'ON' if debug else 'OFF'}"
            )

            continue

        # ----------------------------------------------------
        # PROCESS MESSAGE
        # ----------------------------------------------------

        try:

            result = process_message(
                text,
                session_id
            )

            session_id = result[
                "session_id"
            ]

            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            display_result(
                result
            )

            # ------------------------------------------------
            # DEBUG
            # ------------------------------------------------

            if debug:

                print()

                print(
                    "[DEBUG]"
                )

                print(
                    "Intent:",
                    result["intent"]
                )

                print(
                    "Language:",
                    result["language"]
                )

                print(
                    "Confidence:",
                    result["confidence"]
                )

                print(
                    "Entities:",
                    result["entities"]
                )

                print(
                    "Concerns:",
                    result["concerns"]
                )

                print(
                    "Profile:",
                    result["profile"]
                )

        except Exception as exc:

            print()

            print(
                "Erreur :",
                str(exc)
            )

            print()


if __name__ == "__main__":

    main()