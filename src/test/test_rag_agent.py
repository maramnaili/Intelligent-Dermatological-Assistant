import unittest
from unittest.mock import patch

from src.agent.orchestrator import _document_question, orchestrate, route_message
from src.chatbot.chat_pipeline import extract_concerns, process_message
from src.recommendation.safety_rules import is_product_safe
from src.rag.service import _answer_with_matches, answer, retrieve, split_text


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_size_and_overlap(self):
        text = " ".join(f"mot{i}" for i in range(100))
        chunks = split_text(text, chunk_size=80, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        self.assertTrue(
            any(
                set(chunks[index].split()) & set(chunks[index + 1].split())
                for index in range(len(chunks) - 1)
            )
        )

    def test_invalid_chunk_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=10, overlap=10)


class RouterTests(unittest.TestCase):
    def test_document_question_excludes_only_the_routine_request(self):
        self.assertEqual(
            _document_question(
                "Pour mes taches et mon mélasma, que dit l’étude sur les "
                "protections teintées, et comment l’intégrer à ma routine du matin ?"
            ),
            "Pour mes taches et mon mélasma, que dit l’étude sur les protections teintées?",
        )

    def test_routes_documentary_question_to_rag(self):
        self.assertEqual(
            route_message("What is the difference between AHA and BHA?"),
            "rag",
        )
        self.assertEqual(
            route_message("Explique la différence entre AHA et BHA"),
            "rag",
        )

    def test_routes_compound_question_to_both_tools(self):
        self.assertEqual(
            route_message(
                "Explique l'acide salicylique puis propose-moi une routine"
            ),
            "mixed",
        )
        self.assertEqual(
            route_message(
                "J'ai une peau grasse avec de l'acné. "
                "Explique les recommandations EuroGuiDerm puis propose une routine."
            ),
            "mixed",
        )
        self.assertEqual(
            route_message(
                "I have sensitive, eczema-prone skin. Explain the barrier-care "
                "evidence and help me adapt my RenYou routine."
            ),
            "mixed",
        )
        self.assertEqual(
            route_message(
                "Pour mes taches et mon mélasma, que dit l'étude sur les "
                "protections teintées, et comment l'intégrer à ma routine du matin ?"
            ),
            "mixed",
        )

    def test_dermatology_questions_stay_in_the_skincare_knowledge_domain(self):
        self.assertEqual(
            route_message("Que dit EuroGuiDerm sur l'acné ?"),
            "rag",
        )
        self.assertEqual(
            route_message(
                "What does the review say about atopic dermatitis skin care?"
            ),
            "rag",
        )

    def test_keeps_existing_routine_request_on_ml_path(self):
        self.assertEqual(route_message("Give me my routine"), "ml")

    @patch("src.chatbot.chat_pipeline.build_routine")
    @patch("src.chatbot.chat_pipeline.predict_intent")
    def test_explicit_routine_request_survives_misclassified_mixed_prompt(
        self, predict_intent, build_routine
    ):
        predict_intent.return_value = {
            "intent": "profile_info",
            "confidence": 0.9,
        }
        build_routine.return_value = {
            "morning": {},
            "evening": {},
            "safety": {},
        }

        result = process_message(
            "I have oily skin and acne. What does the guideline say? "
            "Please build my morning and evening RenYou routine.",
            session_id="explicit-routine-integration-test",
            language="en",
        )

        self.assertEqual(result["intent"], "routine_request")
        self.assertIsNotNone(result["data"])
        self.assertIn("routine", result["data"])
        build_routine.assert_called_once()

    @patch("src.agent.orchestrator.rag_answer")
    def test_existing_medical_alert_prevents_rag_generation(self, rag_answer):
        result = orchestrate(
            "What is retinol? I suspect melanoma.",
            session_id="medical-alert-test",
            force_rag=True,
        )

        self.assertEqual(result["mode"], "safety")
        self.assertTrue(result["medical_alerts"])
        self.assertIn("melanoma", result["response"].lower())
        rag_answer.assert_not_called()

    @patch("src.chatbot.chat_pipeline.predict_intent")
    @patch("src.agent.orchestrator.rag_answer")
    def test_severe_acne_in_natural_word_order_blocks_rag_and_routine(
        self, rag_answer, predict_intent
    ):
        predict_intent.return_value = {
            "intent": "profile_info",
            "confidence": 0.9,
        }

        result = orchestrate(
            "My acne is severe and painful. What does EuroGuiDerm say, "
            "and please build my RenYou routine?",
            session_id="severe-acne-word-order-test",
        )

        self.assertEqual(result["mode"], "safety")
        self.assertTrue(result["medical_alerts"])
        self.assertEqual(result["tools_used"], ["chat_ml"])
        self.assertNotIn("routine", result.get("data", {}))
        rag_answer.assert_not_called()

    @patch("src.agent.orchestrator.rag_answer")
    def test_ambiguous_follow_up_keeps_the_session_language(self, rag_answer):
        rag_answer.return_value = {"answer": "Answer", "sources": [], "mode": "rag"}
        session_id = "language-continuity-test"

        orchestrate("What is AHA?", session_id=session_id, force_rag=True)
        orchestrate("ok", session_id=session_id, force_rag=True)

        self.assertEqual(rag_answer.call_args.kwargs["language"], "en")

    @patch("src.agent.orchestrator.process_message")
    def test_existing_chat_receives_session_language(self, process_message):
        process_message.return_value = {"language": "en", "response": "Okay"}
        session_id = "ml-language-continuity-test"
        orchestrate("What is AHA?", session_id=session_id, force_rag=True)

        result = orchestrate("ok", session_id=session_id)

        self.assertEqual(process_message.call_args.kwargs["language"], "en")
        self.assertEqual(result["mode"], "ml")

    @patch("src.agent.orchestrator.rag_answer")
    def test_rag_response_includes_real_sources_and_tool_trace(self, rag_answer):
        rag_answer.return_value = {
            "answer": "Réponse documentée [1].",
            "sources": [{"title": "Guide", "page_or_section": "3"}],
            "mode": "rag",
        }
        result = orchestrate("What is AHA?", force_rag=True)

        self.assertEqual(result["mode"], "rag")
        self.assertEqual(result["tools_used"], ["retrieve_knowledge"])
        self.assertEqual(result["sources"][0]["title"], "Guide")
        rag_answer.assert_called_once()

    @patch("src.agent.orchestrator.process_message")
    @patch("src.agent.orchestrator.rag_answer")
    def test_mixed_request_is_bounded_and_combines_results(
        self, rag_answer, process_message
    ):
        process_message.return_value = {
            "response": "Routine proposée.",
            "medical_alerts": [],
            "data": {"routine": {}},
        }
        rag_answer.return_value = {
            "answer": "Explication documentée.",
            "sources": [{"title": "Guide"}],
            "mode": "rag",
        }
        result = orchestrate(
            "Explique l'acide salicylique puis propose-moi une routine",
            session_id="test-session",
        )

        self.assertEqual(result["mode"], "mixed")
        self.assertEqual(
            result["tools_used"], ["chat_ml", "retrieve_knowledge"]
        )
        self.assertIn("Explication documentée.", result["response"])
        self.assertIn("Routine proposée.", result["response"])
        self.assertEqual(result["data"], {"routine": {}})
        self.assertEqual(
            rag_answer.call_args.args[0],
            "Explique l'acide salicylique?",
        )


class EczemaSafetyTests(unittest.TestCase):
    def test_extracts_eczema_and_atopic_dermatitis(self):
        self.assertEqual(
            extract_concerns(
                "I have eczema and previously diagnosed atopic dermatitis."
            ),
            ["eczema"],
        )
        self.assertEqual(
            extract_concerns("J'ai une dermatite atopique."),
            ["eczema"],
        )

    def test_excludes_treatment_and_exfoliation_for_eczema_profile(self):
        profile = {"concerns": ["eczema"]}

        self.assertFalse(is_product_safe(
            {
                "product_name": "Exfoliant AHA",
                "routine_step": "traitement",
            },
            profile,
        ))
        self.assertTrue(is_product_safe(
            {
                "product_name": "Gentle cleanser",
                "routine_step": "nettoyage",
            },
            profile,
        ))


class RetrievalTests(unittest.TestCase):
    @patch("src.rag.service.embed_texts")
    @patch("src.rag.service.get_collection")
    def test_retrieval_uses_approved_cross_language_matches(
        self, get_collection, embed_texts
    ):
        collection = get_collection.return_value
        collection.get.return_value = {"ids": ["document-1-0"]}
        collection.query.return_value = {
            "documents": [["AHA exfoliants"]],
            "metadatas": [[
                {
                    "title": "Validated guide",
                    "language": "fr",
                    "trust_level": "approved",
                }
            ]],
            "distances": [[0.2]],
        }
        embed_texts.return_value = [[0.1, 0.9]]

        matches = retrieve("What are AHAs?", language="en")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["metadata"]["language"], "fr")
        self.assertEqual(
            collection.query.call_args.kwargs["where"],
            {"trust_level": "approved"},
        )

    @patch("src.rag.service.embed_texts")
    @patch("src.rag.service.get_collection")
    def test_explicit_melasma_query_drops_unrelated_retrieved_documents(
        self, get_collection, embed_texts
    ):
        collection = get_collection.return_value
        collection.get.return_value = {
            "ids": [f"document-{index}" for index in range(30)]
        }
        documents = [
            "Treatment pathways for scabies in dermatology."
            for _ in range(5)
        ]
        documents.extend(
            "Tinted sunscreen was compared for melasma patients."
            for _ in range(6)
        )
        collection.query.return_value = {
            "documents": [documents],
            "metadatas": [[
                {
                    "title": "Scabies care pathway",
                    "doi": "10.0000/scabies",
                    "trust_level": "approved",
                }
                for _ in range(5)
            ] + [
                {
                    "title": "Visible-light protection in melasma",
                    "doi": "10.0000/melasma",
                    "trust_level": "approved",
                }
                for _ in range(6)
            ]],
            "distances": [[
                0.1, 0.11, 0.12, 0.13, 0.14,
                0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
            ]],
        }
        embed_texts.return_value = [[0.1, 0.9]]

        matches = retrieve("Que dit cette étude sur le mélasma ?", language="fr")

        self.assertEqual(len(matches), 5)
        self.assertEqual(matches[0]["metadata"]["doi"], "10.0000/melasma")
        self.assertEqual(
            collection.query.call_args.kwargs["n_results"],
            25,
        )

    @patch("src.rag.service.retrieve")
    @patch("src.rag.service._ollama_request")
    def test_generation_uses_previously_retrieved_matches(
        self, ollama_request, retrieve_mock
    ):
        matches = [{
            "text": "The study compared a tinted sunscreen to an untinted sunscreen.",
            "metadata": {
                "title": "Melasma study",
                "doi": "10.1111/example",
            },
            "distance": 0.1,
        }]
        ollama_request.return_value = {
            "message": {"content": "The study compared two sunscreens."}
        }

        result = _answer_with_matches(
            "What did the study compare?",
            "en",
            matches,
        )

        self.assertEqual(result["sources"][0]["doi"], "10.1111/example")
        prompt = ollama_request.call_args.args[1]["messages"][1]["content"]
        self.assertIn(matches[0]["text"], prompt)
        retrieve_mock.assert_not_called()

    @patch("src.rag.service._ollama_request")
    @patch("src.rag.service.retrieve")
    def test_prescriptive_generation_is_replaced_with_safe_text(
        self, retrieve_mock, ollama_request
    ):
        retrieve_mock.return_value = [{
            "text": "Trial findings about treatments.",
            "metadata": {
                "title": "Review",
                "source": "PeerJ",
                "doi": "10.7717/example",
                "trust_level": "approved",
            },
            "distance": 0.1,
        }]
        ollama_request.return_value = {
            "message": {
                "content": "You should use laser treatments for acne scars."
            }
        }

        result = answer("What did the review find?", language="en")

        self.assertIn("I can’t turn them into a treatment recommendation", result["answer"])
        self.assertEqual(result["sources"][0]["doi"], "10.7717/example")

    @patch("src.rag.service._ollama_request")
    @patch("src.rag.service.retrieve")
    def test_documented_guideline_recommendations_are_not_mistaken_for_personal_advice(
        self, retrieve_mock, ollama_request
    ):
        retrieve_mock.return_value = [{
            "text": "The guideline recommends benzoyl peroxide as one treatment option.",
            "metadata": {
                "title": "Acne guideline",
                "source": "EuroGuiDerm",
                "doi": "10.1111/example",
                "trust_level": "approved",
            },
            "distance": 0.1,
        }]
        ollama_request.return_value = {
            "message": {
                "content": (
                    "The guideline reports benzoyl peroxide as one of its "
                    "treatment recommendations [1]."
                )
            }
        }

        result = answer("What does the guideline report?", language="en")

        self.assertIn(
            "one of its treatment recommendations",
            result["answer"],
        )
        self.assertNotIn(
            "I can’t turn them into a treatment recommendation",
            result["answer"],
        )

    @patch("src.rag.service._ollama_request")
    @patch("src.rag.service.retrieve")
    def test_generated_dosage_and_routine_are_replaced_with_safe_text(
        self, retrieve_mock, ollama_request
    ):
        retrieve_mock.return_value = [{
            "text": "Clinical findings about eczema treatments.",
            "metadata": {
                "title": "Atopic dermatitis review",
                "source": "Cureus",
                "doi": "10.7759/cureus.example",
                "trust_level": "approved",
            },
            "distance": 0.1,
        }]
        ollama_request.return_value = {
            "message": {
                "content": (
                    "A RenYou morning and evening routine for someone with "
                    "eczema might include mometasone 0.1%. It should be applied "
                    "no more than 3-4 times per week."
                )
            }
        }

        result = answer("What does the review say? Build me a routine.", language="en")

        self.assertIn(
            "I can’t turn them into a treatment recommendation",
            result["answer"],
        )
        self.assertNotIn("mometasone", result["answer"])

    @patch("src.rag.service.embed_texts")
    @patch("src.rag.service.get_collection")
    def test_no_approved_documents_does_not_call_ollama(
        self, get_collection, embed_texts
    ):
        get_collection.return_value.get.return_value = {"ids": []}

        self.assertEqual(retrieve("What are AHAs?"), [])
        embed_texts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
