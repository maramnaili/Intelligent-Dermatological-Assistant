# AI Skin Assistant

This is a bilingual (French and English) skincare assistant that combines product recommendations, AM/PM routine building, a conversational chatbot, and retrieval-augmented answers grounded in a reviewed local document collection.


## What it does

- **Product recommendations:** represents product and profile attributes with TF-IDF and ranks products by cosine similarity.
- **Routine building:** assembles available recommendations into morning and evening skincare steps.
- **Safety filtering:** applies project-defined product restrictions and escalation messages for selected profile flags or reported symptoms.
- **French/English chat:** extracts intents and profile details, keeps an in-memory conversation profile, and returns localized responses.
- **Document-grounded answers (RAG):** retrieves approved passages from local PDFs or JATS XML, then generates a French or English answer with source details using Ollama and ChromaDB.
- **Agent orchestration:** routes requests to the chatbot, document retrieval, or both, using bounded, rule-based routing.
- **Evaluation and exploration:** includes unit tests, RAG evaluation cases, an optional RAGAS evaluator, and Jupyter notebooks.

The HTTP API and CLI are intended for local development and evaluation. The API uses Flask's development server and in-memory chat sessions; neither should be treated as production-ready.

## Architecture

```text
src/
  api/                 Flask API
  agent/               Rule-based ML/RAG request orchestration
  chatbot/             Intent model, entity extraction, chat pipeline and CLI
  rag/                 PDF/XML ingestion, ChromaDB retrieval and evaluation
  recommendation/      Product ranking, routine builder and safety rules
  test/                Agent/RAG tests and a browser-based API tester
scripts/
  scraper_parafendri.py  Fetches public product catalogue pages
  clean_dataset.py       Cleans/enriches the scraped CSV
  train_model.py         Builds product recommendation artifacts
data/
  chatbot/             Bilingual intent training examples
  knowledge/           Reviewed reading documents and rights register
  rag/                 RAG evaluation cases
notebooks/              Exploratory analysis and visualizations
reports/                 Evaluation output
```

The main runtime components are:

1. `src/api/app.py` exposes the Flask endpoints.
2. `src/agent/orchestrator.py` routes questions to the conversational pipeline and/or document assistant.
3. `src/chatbot/` handles language, intent, entities, conversation state, and routine requests.
4. `src/recommendation/` ranks products, assembles routines, and applies product safety checks.
5. `src/rag/` ingests permitted documents, retrieves approved passages from ChromaDB, and generates cited answers with Ollama.

## Requirements

- Python 3.10 or later
- pip
- Internet access for the optional product scraper
- [Ollama](https://ollama.com/) and the configured local models for RAG answers and ingestion
- Tesseract OCR installed on the system only when indexing scanned PDFs that require OCR

The project dependencies are listed in [`requirements.txt`](requirements.txt). The RAGAS evaluation uses the same dependency file and a running Ollama service.

## Installation

Run commands from the project root.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, invoke the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare product data and models

The raw scraped product catalogue, cleaned catalogue, and generated model files are excluded from Git. A fresh checkout therefore needs product data and trained artifacts before product recommendation and routine endpoints can work.

1. Fetch the public product catalogue:

   ```bash
   python scripts/scraper_parafendri.py
   ```

   This writes `data/raw/products/parafendri_products_raw.csv`. The scraper sends requests to Para Fendri's public website; respect its terms, robots policy, and request load.

2. Clean and enrich the catalogue:

   ```bash
   python scripts/clean_dataset.py
   ```

   This writes `data/processed/parafendri_products_clean.csv`.

3. Train the product recommender:

   ```bash
   python scripts/train_model.py
   ```

   This creates artifacts under `models/recommendation/`.

4. Train the chatbot intent classifier:

   ```bash
   python -m src.chatbot.intent_classifier
   ```

   This creates `models/chatbot/intent_classifier.joblib`.

The scraper and training steps are separate: running the scraper alone does not produce the cleaned data or trained models.

## Run the application

### Interactive CLI

After training both models:

```bash
python -m src.chatbot.chat_cli
```

Enter French or English messages. The CLI supports `/help`, `/reset`, `/debug`, and `/quit`.

### HTTP API

Start the local Flask development server:

```bash
python -m src.api.app
```

The server listens on `http://127.0.0.1:5000`. Check that it is running:

```bash
curl http://127.0.0.1:5000/health
```

Example routine request:

```bash
curl -X POST http://127.0.0.1:5000/ml/routine -H "Content-Type: application/json" -d '{"skin_type":"grasse","concerns":["acne"],"preferred_ingredients":["niacinamide"],"avoid_ingredients":[],"pregnant":false,"breastfeeding":false,"rosacea":false,"ultra_sensitive":false,"dermatological_treatment":false}'
```


Example chat request:

```bash
curl -X POST http://127.0.0.1:5000/ml/chat -H "Content-Type: application/json" -d '{"message":"I have oily skin and acne. Can you suggest a routine?","language":"en"}'
```

Pass the returned `session_id` in later chat requests to retain the conversation profile. Sessions are held in server memory and are lost when the process restarts.

## API endpoints

All request bodies are JSON.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Basic process health check |
| `POST` | `/ml/recommend` | Rank products for a profile; optional `?top_n=5` |
| `POST` | `/ml/routine` | Build available morning and evening routine steps |
| `POST` | `/ml/chat` | Continue a conversational skincare session |
| `POST` | `/ml/chat/reset` | Reset the profile for a `session_id` |
| `POST` | `/ml/rag/ask` | Force a document-grounded question |
| `POST` | `/ml/agent/ask` | Route a question through the agent orchestrator |

Chat and agent requests accept `message`, optional `session_id`, and optional `language` (`"fr"` or `"en"`). A recommendation profile can include `skin_type`, `concerns`, `preferred_ingredients`, `avoid_ingredients`, `pregnant`, `breastfeeding`, `rosacea`, `ultra_sensitive`, and `dermatological_treatment`. Use normalized catalogue values where applicable (for example, `skin_type: "grasse"` and `concerns: ["acne"]`) for best results.

## Optional: enable the document assistant (RAG)

The document assistant needs a local Ollama server and an indexed ChromaDB collection. Its defaults are:

- Ollama URL: `http://localhost:11434`
- Chat model: `qwen2.5:3b`
- Embedding model: `qwen3-embedding:0.6b`
- ChromaDB directory: `data/rag/chroma`
- Collection: `renyou_knowledge`

Install/start Ollama, then download the default models:

```bash
ollama pull qwen2.5:3b
ollama pull qwen3-embedding:0.6b
```

Index a document from the reviewed knowledge collection (example):

```bash
python -m src.rag.ingest "data/knowledge/EuroGuiderm_acne_2026_CC-BY.pdf" --language en --source "Journal of the European Academy of Dermatology and Venereology" --document-type guideline --trust-level approved --last-reviewed 2026-11-02 --rights-evidence "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13308670/fullTextXML" --reuse-license "CC BY 4.0" --reuse-license-url "https://creativecommons.org/licenses/by/4.0/"
```


The service also supports these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RENYOU_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `RENYOU_OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Answer generation model |
| `RENYOU_OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model |
| `RENYOU_CHROMA_PATH` | `data/rag/chroma` | Persistent vector database directory |
| `RENYOU_CHROMA_COLLECTION` | `renyou_knowledge` | ChromaDB collection name |
| `RENYOU_RAG_MAX_DISTANCE` | `0.55` | Maximum accepted cosine distance |

On PowerShell, set a variable for the current terminal with `$env:RENYOU_OLLAMA_URL = "http://localhost:11434"`. On macOS/Linux, use `export RENYOU_OLLAMA_URL="http://localhost:11434"`.

To run the local RAGAS evaluation after the collection is populated and Ollama is available:

```bash
python -m src.rag.evaluate
```

Each evaluation case is retrieved once. The same passages are supplied to answer
generation and then scored as the retrieved contexts, so the generated answer
and RAGAS metrics use a consistent retrieval result.

Evaluation cases live in `data/rag/evaluation_cases.json`; reports are written under `reports/rag-evaluation/`. The benchmark is small and manually curated. Its scores are not clinical validation.

## Tests and notebooks

Run the API-agent and RAG unit tests from the project root:

```bash
python -m unittest discover -s src/test -p "test_*.py"
```

Some RAG evaluation and runtime paths require Ollama and an indexed collection; tests that exercise those paths may need a configured local service. The notebooks in `notebooks/` are for exploratory analysis and visualization, and may depend on the generated product data described above.


