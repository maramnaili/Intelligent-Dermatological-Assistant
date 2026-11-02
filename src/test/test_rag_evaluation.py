import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag.evaluate import DEFAULT_CASES, _build_samples, _load_cases


class EvaluationDatasetTests(unittest.TestCase):
    def test_benchmark_has_french_and_english_reference_cases(self):
        cases = _load_cases(DEFAULT_CASES)

        self.assertEqual(len(cases), 6)
        self.assertEqual(
            {case["language"] for case in cases},
            {"fr", "en"},
        )
        self.assertTrue(all(case["expected_source_dois"] for case in cases))

    @patch("src.rag.evaluate._answer_with_matches")
    @patch("src.rag.evaluate.retrieve")
    def test_samples_reuse_one_retrieval_for_answer_and_evaluation(
        self, retrieve_mock, generate_mock
    ):
        matches = [{
            "text": "The sunscreen study compared tinted and untinted products.",
            "metadata": {"doi": "10.1111/jocd.70450"},
        }]
        retrieve_mock.return_value = matches
        generate_mock.return_value = {
            "answer": "The study compared two sunscreen types.",
        }
        cases = [{
            "id": "melasma-en",
            "language": "en",
            "question": "What did the sunscreen study compare?",
            "expected_source_dois": ["10.1111/jocd.70450"],
            "reference": "It compared tinted and untinted sunscreens.",
        }]

        samples, retrieval_checks = _build_samples(cases, top_k=5)

        self.assertEqual(samples[0]["user_input"], cases[0]["question"])
        self.assertEqual(
            generate_mock.call_args.args,
            (cases[0]["question"], "en", matches),
        )
        self.assertEqual(
            samples[0]["retrieved_contexts"],
            ["The sunscreen study compared tinted and untinted products."],
        )
        self.assertEqual(samples[0]["reference"], cases[0]["reference"])
        self.assertTrue(retrieval_checks[0]["expected_source_hit"])
        retrieve_mock.assert_called_once_with(
            cases[0]["question"],
            language="en",
            top_k=5,
        )


if __name__ == "__main__":
    unittest.main()
