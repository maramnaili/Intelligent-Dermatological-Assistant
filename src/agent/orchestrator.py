"""Rule-first request routing and bounded orchestration."""

import re
import uuid

from src.chatbot.chat_pipeline import (
    ENGLISH_WORDS,
    FRENCH_WORDS,
    _apply_medical_flags_from_text,
    _medical_alert_data,
    _medical_alert_response,
    detect_language,
    get_session,
    process_message,
)
from src.recommendation.safety_rules import detect_medical_alerts
from src.rag.service import RAGUnavailableError, answer as rag_answer


MAX_TOOL_CALLS = 2
KNOWLEDGE_CUES = (
    "what is",
    "what's",
    "what are",
    "explain",
    "difference between",
    "what is the difference",
    "how does",
    "how to use",
    "what does",
    "what are the benefits",
    "c'est quoi",
    "qu'est-ce",
    "explique",
    "différence entre",
    "quelle est la différence",
    "quels sont",
    "comment utiliser",
    "comment fonctionne",
    "à quoi sert",
    "que signifie",
    "que dit",
    "que disent",
    "que rapporte",
)
SKINCARE_CUES = (
    "aha", "bha", "spf", "sunscreen", "skin barrier",
    "barrière cutanée", "retinol", "rétinol", "niacinamide",
    "salicylic acid", "acide salicylique", "glycolic acid",
    "acide glycolique", "hyaluronic acid", "acide hyaluronique",
    "vitamin c", "vitamine c", "azelaic acid", "acide azélaïque",
    "ceramide", "céramide", "panthenol", "panthénol", "ingredient",
    "ingrédient", "actif cosmétique", "cosmetic",
    "acne", "acné", "acne-prone", "acne scars", "cicatrices d'acné",
    "eczema", "eczéma", "atopic dermatitis", "dermatite atopique",
    "rosacea", "rosacée", "melasma", "mélasma", "psoriasis",
    "scabies", "gale", "dark spots", "taches",
)
ROUTINE_CUES = (
    "routine", "build me", "build a", "create a routine", "propose",
    "recommend", "recommendation", "recommande", "recommandation",
    "propose-moi", "propose moi",
)
ROUTINE_REQUEST_CLAUSE_PATTERNS = (
    r"[,;]?\s*(?:(?:and|et|then|puis|ensuite)\s+)?"
    r"(?:(?:please|s'il vous plaît|s’il vous plaît)\s+)?"
    r"(?:build|create|make|generate|adapt|update|recommend|"
    r"propose(?:-moi)?|recommande(?:-moi)?|recommander|"
    r"crée|creer|construis|génère|genere|adapte|fais)\b"
    r"[^.!?]*\broutine\b[^.!?]*[.!?]?\s*$",
    r"[,;]?\s*(?:and|et)\s+"
    r"(?:how\s+(?:can|should)\s+i\b|comment\b)"
    r"[^.!?]*\broutine\b[^.!?]*[.!?]?\s*$",
)


def _looks_like_knowledge_question(message):
    normalized = message.casefold()
    has_question_cue = any(cue in normalized for cue in KNOWLEDGE_CUES)
    has_skincare_cue = any(cue in normalized for cue in SKINCARE_CUES)
    return has_question_cue and has_skincare_cue


def route_message(message):
    """Return ML, RAG, or mixed based on transparent lexical rules."""
    knowledge = _looks_like_knowledge_question(message)
    personalized = any(cue in message.casefold() for cue in ROUTINE_CUES)
    if knowledge and personalized:
        return "mixed"
    if knowledge:
        return "rag"
    return "ml"


def _document_question(message):
    question = message
    for pattern in ROUTINE_REQUEST_CLAUSE_PATTERNS:
        question = re.sub(pattern, "", question, flags=re.IGNORECASE).strip()
    if question and question[-1] not in "?.!":
        question += "?"
    return question or message


def _resolve_language(message, previous_language, requested_language):
    if requested_language is not None and (
        not isinstance(requested_language, str)
        or requested_language not in {"fr", "en"}
    ):
        raise ValueError("language doit être 'fr' ou 'en'.")
    if requested_language in {"fr", "en"}:
        return requested_language
    words = set(re.findall(r"[a-zA-ZÀ-ÿ'-]+", message.lower()))
    french_score = len(words & FRENCH_WORDS)
    english_score = len(words & ENGLISH_WORDS)
    if french_score > english_score:
        return "fr"
    if english_score > french_score:
        return "en"
    if previous_language in {"fr", "en"}:
        return previous_language
    return detect_language(message)


def _safe_rag_result(message, session_id, language):
    profile = get_session(session_id)
    _apply_medical_flags_from_text(profile, message)
    alerts = detect_medical_alerts(profile)
    if alerts:
        return {
            "session_id": session_id,
            "intent": "medical_alert",
            "language": language,
            "auto_generated": False,
            "confidence": 1.0,
            "entities": {},
            "concerns": [],
            "profile": {},
            "response": _medical_alert_response(alerts[0], language),
            "medical_alerts": _medical_alert_data(alerts),
            "mode": "safety",
            "sources": [],
            "tools_used": ["apply_safety_rules"],
        }

    try:
        result = rag_answer(message, language=language)
    except RAGUnavailableError:
        if language == "en":
            text = (
                "The document assistant is not ready yet. Start Ollama and "
                "index reviewed bilingual PDFs; the existing RenYou chat is "
                "still available for skincare routines and recommendations."
            )
        else:
            text = (
                "L'assistant documentaire n'est pas encore prêt. Démarrez "
                "Ollama et indexez des PDF bilingues validés ; le chat RenYou "
                "existant reste disponible pour les routines et recommandations."
            )
        return {
            "session_id": session_id,
            "intent": "knowledge_question",
            "language": language,
            "auto_generated": False,
            "confidence": 1.0,
            "entities": {},
            "concerns": [],
            "profile": {},
            "response": text,
            "medical_alerts": [],
            "mode": "rag_unavailable",
            "sources": [],
            "tools_used": ["retrieve_knowledge"],
            "tool_errors": ["retrieve_knowledge: unavailable"],
        }
    return {
        "session_id": session_id,
        "intent": "knowledge_question",
        "language": language,
        "auto_generated": False,
        "confidence": 1.0,
        "entities": {},
        "concerns": [],
        "profile": {},
        "response": result["answer"],
        "medical_alerts": [],
        "mode": result["mode"],
        "sources": result["sources"],
        "tools_used": ["retrieve_knowledge"],
    }


def orchestrate(message, session_id=None, force_rag=False, language=None):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Le message ne peut pas être vide.")
    if session_id is not None and (
        not isinstance(session_id, str) or len(session_id) > 256
    ):
        raise ValueError("session_id doit être un texte de 256 caractères maximum.")

    session_id = session_id or str(uuid.uuid4())
    profile = get_session(session_id)
    language = _resolve_language(message, profile.get("language"), language)
    profile["language"] = language
    route = "rag" if force_rag else route_message(message)
    allowed_tools = {
        "retrieve_knowledge": lambda: _safe_rag_result(
            _document_question(message), session_id, language
        ),
        "chat_ml": lambda: process_message(
            message,
            session_id=session_id,
            language=language,
        ),
    }
    calls = []

    if route == "ml":
        calls.append("chat_ml")
        result = allowed_tools["chat_ml"]()
        result.update(mode="ml", sources=[], tools_used=calls)
        return result

    if route == "rag":
        calls.append("retrieve_knowledge")
        return allowed_tools["retrieve_knowledge"]()

    calls.extend(["chat_ml", "retrieve_knowledge"])
    if len(calls) > MAX_TOOL_CALLS:
        raise RuntimeError("La limite d'appels d'outils a été dépassée.")

    ml_result = allowed_tools["chat_ml"]()
    if ml_result.get("medical_alerts"):
        ml_result.update(mode="safety", sources=[], tools_used=["chat_ml"])
        return ml_result

    knowledge_result = allowed_tools["retrieve_knowledge"]()
    if knowledge_result.get("mode") == "safety":
        return knowledge_result
    ml_response = ml_result.get("response", "").strip()
    knowledge_response = knowledge_result.get("response", "").strip()
    if ml_response and knowledge_response:
        separator = "\n\n" if language == "en" else "\n\n"
        ml_result["response"] = knowledge_response + separator + ml_response
    elif knowledge_response:
        ml_result["response"] = knowledge_response

    ml_result.update(
        mode="mixed",
        sources=knowledge_result.get("sources", []),
        tools_used=calls,
    )
    if knowledge_result.get("mode") == "rag_unavailable":
        ml_result["tool_errors"] = ["retrieve_knowledge: unavailable"]
    return ml_result
