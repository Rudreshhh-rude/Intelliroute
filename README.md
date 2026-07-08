# IntelliRoute: Adaptive Enterprise Knowledge & Action Assistant

Welcome to **IntelliRoute**! This is a production-grade, adaptive GenAI assistant built to solve the real-world engineering challenges of enterprise document query tools: controlling LLM transaction costs, providing robust security guardrails, matching the right retrieval strategy to each query, and tracking system metrics in real time.

Unlike standard RAG apps that just query a vector database and pass the results to a single LLM, IntelliRoute makes smart runtime decisions in code to optimize speed, quality, and cost.

---

## 🏗️ Architecture & Core Modules

Here is a breakdown of what I built under the hood and why:

### 1. Ingestion Pipeline
*   **Multi-Source Extractor**: Handles unstructured PDFs page-by-page (using `pypdf`) and processes structured Markdown/YAML outline configurations.
*   **Hierarchical Parser**: Parses headers recursively using a stack-based heading parser to construct hierarchical `TreeNode` outline index structures.

### 2. Hybrid Retrieval Engine
*   **Broad Semantic Search**: Uses persistent ChromaDB vector store collections with `gemini-embedding-2` embeddings.
*   **Structural PageIndex Retriever**: Bypasses embeddings entirely for structured manuals. It traverses the document outline index tree using LLM directory decisions, with an **Auto-Dive Optimization** to fast-track through wrapper folders.
*   **Batch Reranking**: Groups matching snippets and scores relevance in a single API call to bypass requests-per-minute rate limit bottlenecks.

### 3. Model Router (The Centerpiece)
*   At runtime, a classifier audits the incoming query.
*   **Simple Queries** (factual retrieval, definitions) are routed to the fast, cost-effective `gemini-2.5-flash` model.
*   **Complex Queries** (analytics, comparisons, code generation) are routed to the premium `gemini-2.5-pro` model.
*   All token exchanges, latencies, and dollar costs are logged per transaction.

### 4. Security Guardrails
*   **Heuristic Filters**: Fast regex scanning checks for common prompt injection phrases (e.g., "ignore previous instructions") at the input level.
*   **Semantic LLM Audit**: Standard queries bypass further checks, while suspicious queries undergo a lightweight LLM check. Unsafe attempts are blocked in under 0.6 seconds at $0.00 search cost.

### 5. Semantic Cache & Token Rate Limiter
*   **Semantic Cache**: Indexes query-response pairs in a dedicated Chroma collection using a cosine similarity threshold of `0.85`. Cache hits load in under 0.01 seconds and bypass LLM invocation costs completely.
*   **Token Bucket Limiter**: Thread-safe rate limiter per API key (default: 60-token burst, 1-token/sec refill rate) to prevent system abuse.

### 6. Observability Dashboard SPA
*   A clean, dark-mode single-page dashboard served directly from the FastAPI backend root. It displays real-time session cost widgets, average response latencies, active document listings, and live transaction log grids.

### 7. Evaluation Suite
*   An automated script (`run_evaluation.py`) that uploads a test manual, runs 15 evaluation queries (simple, complex, tool, and unsafe prompt injections), and outputs an execution report.

---

## 🛠️ Tech Stack & Persistence Decisions

*   **Backend Framework**: FastAPI (highly modular, dependency-injected).
*   **Vector Store**: ChromaDB (locally persistent file storage).
*   **Primary DB**: SQLite (`database.db`). We chose a persistent SQLite database over volatile in-memory Python dictionaries to persist metrics, API keys, and chat histories across server restarts.
*   **Unified SPA Delivery**: Served directly from FastAPI to avoid complex CORS configurations, environment variable mismatch bugs, and cross-origin roundtrip latencies.

---

## 🚀 Local Quickstart

### 1. Configure Credentials
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_DB_PATH=backend/chromadb_store
MODEL_NAME_FLASH=gemini-2.5-flash
MODEL_NAME_PRO=gemini-2.5-pro
EMBEDDING_MODEL=text-embedding-004
```

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Rudreshhh-rude/Intelliroute.git
cd Intelliroute

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```bash
python -m uvicorn backend.app.main:app --port 8000
```
Open **`http://localhost:8000/`** in your browser.

### 4. Run the Evaluation Suite
With the server running on port 8000, execute the benchmark:
```bash
python backend/app/tests/run_evaluation.py
```
This will test the routing, cache, and guardrails, and save a summary report to `evaluation_report.md`.
