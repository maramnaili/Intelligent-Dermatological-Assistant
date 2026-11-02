"""Run a local RAGAS evaluation against the RenYou knowledge collection."""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from src.rag.service import _answer_with_matches, retrieve


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CASES = BASE_DIR / "data" / "rag" / "evaluation_cases.json"
DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "rag-evaluation"
DEFAULT_TOP_K = 5


def _load_cases(path):
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Le jeu d'évaluation est vide ou invalide : {path}")

    required = {
        "id",
        "language",
        "question",
        "expected_source_dois",
        "reference",
    }
    for index, case in enumerate(cases):
        missing = required - case.keys()
        if missing:
            raise ValueError(
                f"Cas {index} incomplet, champs manquants : {sorted(missing)}"
            )
        if case["language"] not in {"fr", "en"}:
            raise ValueError(f"Langue invalide pour le cas {case['id']}.")
    return cases


def _build_samples(cases, top_k):
    samples = []
    retrieval_checks = []
    for case_index, case in enumerate(cases, 1):
        print(
            f"Préparation {case_index}/{len(cases)} : {case['id']}",
            flush=True,
        )
        matches = retrieve(
            case["question"],
            language=case["language"],
            top_k=top_k,
        )
        generated = _answer_with_matches(
            case["question"],
            case["language"],
            matches,
        )
        retrieved_dois = sorted({
            match["metadata"].get("doi")
            for match in matches
            if match["metadata"].get("doi")
        })
        expected_dois = case["expected_source_dois"]
        retrieval_checks.append({
            "id": case["id"],
            "language": case["language"],
            "expected_source_dois": expected_dois,
            "retrieved_source_dois": retrieved_dois,
            "expected_source_hit": bool(set(expected_dois) & set(retrieved_dois)),
            "ragas_response": generated["answer"],
        })
        samples.append({
            "user_input": case["question"],
            "response": generated["answer"],
            "retrieved_contexts": [match["text"] for match in matches],
            "reference": case["reference"],
        })
    return samples, retrieval_checks


def _build_evaluator_models():
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "Dépendances RAGAS manquantes. Installez-les avec "
            "'.\\.env\\Scripts\\python.exe -m pip install -r requirements-ragas.txt'."
        ) from exc

    base_url = os.environ.get("RENYOU_OLLAMA_URL", "http://localhost:11434")
    llm_model = os.environ.get(
        "RENYOU_RAGAS_LLM_MODEL",
        os.environ.get("RENYOU_OLLAMA_CHAT_MODEL", "qwen2.5:3b"),
    )
    embedding_model = os.environ.get(
        "RENYOU_RAGAS_EMBEDDING_MODEL",
        os.environ.get(
            "RENYOU_OLLAMA_EMBEDDING_MODEL",
            "qwen3-embedding:0.6b",
        ),
    )
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.environ.get("RENYOU_RAGAS_API_KEY", "ollama"),
        timeout=120,
        max_retries=0,
    )
    evaluator_llm = llm_factory(
        llm_model,
        client=client,
        temperature=0,
    )
    evaluator_embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=client,
    )
    run_config = RunConfig(
        timeout=120,
        max_retries=1,
        max_wait=5,
    )
    return (
        client,
        evaluator_llm,
        evaluator_embeddings,
        llm_model,
        embedding_model,
        run_config,
    )


def run_evaluation(cases_path=DEFAULT_CASES, report_dir=DEFAULT_REPORT_DIR, top_k=DEFAULT_TOP_K):
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    try:
        from ragas import EvaluationDataset
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS n'est pas installé. Installez requirements-ragas.txt."
        ) from exc

    cases = _load_cases(cases_path)
    samples, retrieval_checks = _build_samples(cases, top_k)
    (
        evaluator_client,
        evaluator_llm,
        evaluator_embeddings,
        llm_model,
        embedding_model,
        run_config,
    ) = (
        _build_evaluator_models()
    )
    evaluation_dataset = EvaluationDataset.from_list(samples)
    metric_scorers = [
        (
            Faithfulness(llm=evaluator_llm, run_config=run_config),
            ("user_input", "response", "retrieved_contexts"),
        ),
        (
            AnswerRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                strictness=1,
                run_config=run_config,
            ),
            ("user_input", "response"),
        ),
        (
            ContextPrecision(llm=evaluator_llm, run_config=run_config),
            ("user_input", "reference", "retrieved_contexts"),
        ),
        (
            ContextRecall(llm=evaluator_llm, run_config=run_config),
            ("user_input", "retrieved_contexts", "reference"),
        ),
    ]

    async def score_dataset():
        try:
            scores = []
            for sample_index, sample in enumerate(evaluation_dataset.samples, 1):
                print(
                    f"RAGAS {sample_index}/{len(evaluation_dataset.samples)} : "
                    f"{cases[sample_index - 1]['id']}",
                    flush=True,
                )
                row_scores = {}
                sample_data = {
                    "user_input": sample.user_input,
                    "response": sample.response,
                    "retrieved_contexts": sample.retrieved_contexts,
                    "reference": sample.reference,
                }
                for scorer, required_fields in metric_scorers:
                    print(f"  - {scorer.name}", flush=True)
                    score = await scorer.ascore(
                        **{
                            field: sample_data[field]
                            for field in required_fields
                        }
                    )
                    row_scores[scorer.name] = score.value
                scores.append(row_scores)
            return scores
        finally:
            await evaluator_client.close()

    per_case_scores = asyncio.run(score_dataset())
    metric_names = list(per_case_scores[0])
    summary_scores = {
        name: sum(row[name] for row in per_case_scores) / len(per_case_scores)
        for name in metric_names
    }
    per_case_results = [
        {"id": case["id"], "language": case["language"], **scores}
        for case, scores in zip(cases, per_case_scores)
    ]
    language_scores = {}
    for language in ("fr", "en"):
        language_rows = [
            row for row in per_case_results if row["language"] == language
        ]
        language_scores[language] = {
            name: sum(row[name] for row in language_rows) / len(language_rows)
            for name in metric_names
        }

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"ragas-{timestamp}.json"
    report = {
        "framework": "RAGAS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "language_counts": {
            language: sum(case["language"] == language for case in cases)
            for language in ("fr", "en")
        },
        "top_k": top_k,
        "ragas_version": version("ragas"),
        "generation_model": os.environ.get(
            "RENYOU_OLLAMA_CHAT_MODEL", "qwen2.5:3b"
        ),
        "evaluation_llm_model": llm_model,
        "evaluation_embedding_model": embedding_model,
        "scores": summary_scores,
        "scores_by_language": language_scores,
        "per_case_scores": per_case_results,
        "retrieval_checks": retrieval_checks,
        "expected_source_hit_rate": (
            sum(row["expected_source_hit"] for row in retrieval_checks)
            / len(retrieval_checks)
        ),
        "limitations": [
            "The bilingual benchmark is small and manually curated.",
            "RAGAS scores depend on the local evaluator model and are not a clinical validation.",
            "Expected source DOI checks are reported separately from RAGAS metrics.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k doit être compris entre 1 et 20.")

    report_path, report = run_evaluation(
        cases_path=args.cases,
        report_dir=args.report_dir,
        top_k=args.top_k,
    )
    print(f"Rapport RAGAS écrit : {report_path}")
    print(json.dumps({
        "case_count": report["case_count"],
        "language_counts": report["language_counts"],
        "scores": report["scores"],
        "expected_source_hit_rate": report["expected_source_hit_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
