import os
import json
import shutil
import tempfile
import time
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import services
from backend.app.core.config import settings
from backend.app.services.ingestion import (
    parse_markdown,
    parse_pdf,
    build_markdown_structural_tree,
    RecursiveCharacterTextSplitter,
    Document
)
from backend.app.services.retrieval import VectorRetriever, PageIndexRetriever, RelevanceReranker, RetrievalRouter
from backend.app.services.routing import ModelRouter, observability_registry, RouteMetrics
from backend.app.services.agent import AgentEngine, ConversationMemory
from backend.app.services.guardrails import Guardrails
from backend.app.services.cache import SemanticCache
from backend.app.services.rate_limiter import RateLimiter
from backend.app.services import database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intelliroute.main")

app = FastAPI(
    title="IntelliRoute API",
    description="Adaptive Enterprise Knowledge & Action Assistant",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database
api_key = settings.gemini_api_key
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing!")
database.init_db(api_key)

vector_retriever = VectorRetriever(
    persist_directory=settings.chroma_db_path,
    api_key=api_key,
    embedding_model=settings.embedding_model
)
page_retriever = PageIndexRetriever(api_key=api_key)
reranker = RelevanceReranker(api_key=api_key)
retrieval_router = RetrievalRouter(api_key=api_key)
agent_engine = AgentEngine(api_key=api_key)
guardrails = Guardrails(api_key=api_key)
semantic_cache = SemanticCache(api_key=api_key, threshold=0.85)

# Burst limit 60, refill rate 1 request/sec
rate_limiter = RateLimiter(capacity=60.0, refill_rate=1.0)

# List of currently ingested documents in memory for display
ingested_files_registry: List[Dict[str, Any]] = []


# --- API Models ---

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(required_roles: List[str]):
    def dependency(x_api_key: Optional[str] = Security(API_KEY_HEADER)):
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="API Key is missing. Please provide X-API-Key header."
            )
        key_record = database.get_api_key(x_api_key)
        if not key_record:
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key."
            )
        if key_record["status"] != "active":
            raise HTTPException(
                status_code=403,
                detail="API Key has been revoked."
            )
        if key_record["role"] not in required_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: {required_roles}."
            )
        return key_record
    return dependency


# --- Endpoint 1: Upload and Ingest Document ---

@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    key_record: dict = Depends(verify_api_key(["Writer", "Admin"]))
):
    """Uploads and ingests a Markdown, PDF, or YAML document into Vector DB and PageIndex."""

    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()
    
    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        start_time = time.time()
        chunks = []
        has_tree = False
        tree_path = None
        
        # 1. Parse and chunk based on extension
        if ext == ".md":
            text = ""
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split using split_documents to preserve metadata
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            doc = Document(text=text, metadata={"source": filename, "file_type": "md"})
            chunks = splitter.split_documents([doc])
            
            # Create structural tree outline
            tree = build_markdown_structural_tree(temp_path)
            tree_json_path = os.path.join(tempfile.gettempdir(), f"{filename}_tree.json")
            with open(tree_json_path, "w", encoding="utf-8") as f:
                json.dump(tree.model_dump(), f, indent=2)
            # Load page_retriever tree index
            page_retriever.load_tree_index(tree_json_path)
            has_tree = True
            tree_path = tree_json_path
            
        elif ext == ".pdf":
            chunks = parse_pdf(temp_path)
            # For PDFs, tree outlines can be mapped linearly
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
            
        # 2. Add chunks to Vector DB
        if chunks:
            vector_retriever.add_documents(chunks)
            
        duration = time.time() - start_time
        
        # Add to global registry
        file_info = {
            "filename": filename,
            "file_type": ext[1:].upper(),
            "chunks_count": len(chunks),
            "has_structural_tree": has_tree,
            "duration_sec": round(duration, 2),
            "timestamp": time.time()
        }
        ingested_files_registry.append(file_info)
        
        return {
            "status": "success",
            "message": f"Successfully ingested {filename}",
            "file_details": file_info
        }
        
    except Exception as e:
        logger.error(f"Failed to ingest document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- Endpoint 2: Unified Chat Endpoint ---

@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    key_record: dict = Depends(verify_api_key(["Reader", "Writer", "Admin"]))
):
    """Processes queries using rate-limiting, semantic cache, guardrails, dual-retrieval routing, and agent loops."""
    session_id = request.session_id
    memory = ConversationMemory(session_id=session_id)

    # 1. API Key & Rate Limiting Enforcement
    client_key = key_record["key"]
    is_allowed, remaining_tokens = rate_limiter.is_allowed(client_key)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Burst limit: 60, Refill: 1 req/sec."
        )

    query = request.query
    start_time = time.time()

    # 2. Semantic Cache Check
    cached_response, similarity = semantic_cache.lookup(query)
    if cached_response:
        # Construct metrics for cache hit transaction
        metrics = RouteMetrics(
            query=query,
            classified_complexity="simple",
            chosen_model="semantic_cache",
            reasoning=f"Semantic cache hit (Similarity: {similarity:.4f})",
            prompt_tokens=0,
            completion_tokens=0,
            latency_sec=time.time() - start_time,
            cost_usd=0.0,
            timestamp=time.time()
        )
        observability_registry.log_transaction(metrics)
        return {
            "answer": cached_response,
            "cache_hit": True,
            "strategy": "CACHED",
            "metrics": metrics.model_dump(),
            "routing_reasoning": metrics.reasoning
        }

    # 3. Guardrails Input Check
    is_unsafe, safety_reason = guardrails.check_input_injection(query)
    if is_unsafe:
        # Log unsafe transaction
        metrics = RouteMetrics(
            query=query,
            classified_complexity="simple",
            chosen_model="guardrails",
            reasoning=f"Blocked by input security guardrail: {safety_reason}",
            prompt_tokens=0,
            completion_tokens=0,
            latency_sec=time.time() - start_time,
            cost_usd=0.0,
            timestamp=time.time()
        )
        observability_registry.log_transaction(metrics)
        return {
            "answer": f"Request blocked: {safety_reason}",
            "cache_hit": False,
            "strategy": "BLOCKED",
            "metrics": metrics.model_dump(),
            "routing_reasoning": safety_reason
        }

    # 4. Retrieval Routing Strategy
    has_tree = page_retriever.tree is not None
    try:
        strategy, routing_reason = retrieval_router.route(query, has_tree=has_tree)
    except Exception:
        strategy, routing_reason = "vector", "Failed to run router. Falling back to vector search."

    # 5. Dual Retrieval Retrieval & Relevance Reranking
    retrieved_docs = []
    
    if strategy in ["vector", "both"]:
        try:
            vector_docs = vector_retriever.query(query, n_results=4)
            retrieved_docs.extend(vector_docs)
        except Exception as e:
            logger.error(f"Vector retriever query failed: {e}")
            
    if strategy in ["page_index", "both"] and has_tree:
        try:
            # page_retriever.traverse returns tree nodes
            traverse_res = page_retriever.traverse(page_retriever.tree, query, max_depth=4)
            # Convert traversed node sections to document references
            page_docs = page_retriever.get_sections_as_documents(traverse_res)
            retrieved_docs.extend(page_docs)
        except Exception as e:
            logger.error(f"PageIndex retrieval failed: {e}")

    # Reranking Retrieved Documents
    if retrieved_docs:
        try:
            reranked_docs = reranker.rerank(query, retrieved_docs, top_n=3)
        except Exception:
            reranked_docs = retrieved_docs[:3]
    else:
        reranked_docs = []

    # Format context text block
    context_block = ""
    for idx, doc in enumerate(reranked_docs):
        context_block += f"Reference [{idx+1}] (Source: {doc.metadata.get('source', 'unknown')}):\n{doc.text}\n\n"

    # 6. Execute Agent Engine Inference Loop
    try:
        response_text, metrics, tool_calls = agent_engine.run_agent_loop(
            query=query,
            context=context_block,
            memory=memory
        )
    except Exception as e:
        # API Error recovery fallback
        response_text = f"An API inference error occurred: {e}."
        metrics = RouteMetrics(
            query=query,
            classified_complexity="simple",
            chosen_model="error_fallback",
            reasoning=f"System inference exception: {e}",
            prompt_tokens=0,
            completion_tokens=0,
            latency_sec=time.time() - start_time,
            cost_usd=0.0,
            timestamp=time.time()
        )
        tool_calls = None
        observability_registry.log_transaction(metrics)

    # 7. Update cache and memory
    if response_text and "inference error" not in response_text.lower():
        semantic_cache.update(query, response_text)
        memory.add_message("user", query)
        memory.add_message("model", response_text)
        # Periodic memory trimming check
        memory.trim_and_summarize(api_key=api_key, max_messages=8)

    return {
        "answer": response_text,
        "cache_hit": False,
        "strategy": strategy.upper(),
        "metrics": metrics.model_dump(),
        "routing_reasoning": metrics.reasoning,
        "tool_calls": tool_calls
    }


# --- Endpoint 3: Observability Dashboard Summary ---

@app.get("/api/metrics")
async def get_metrics_summary(
    key_record: dict = Depends(verify_api_key(["Admin"]))
):
    """Retrieves session observability overview statistics."""
    return database.get_metrics_summary()


# --- Endpoint 4: Detailed Transaction Log ---

@app.get("/api/metrics/transactions")
async def get_detailed_transactions(
    limit: int = Query(50, ge=1, le=100),
    key_record: dict = Depends(verify_api_key(["Admin"]))
):
    """Retrieves the list of detailed query transaction routing logs from SQLite."""
    return database.get_detailed_transactions(limit=limit)


# --- Endpoint 5: Ingested Documents List ---

@app.get("/api/documents")
async def get_ingested_documents(
    key_record: dict = Depends(verify_api_key(["Reader", "Writer", "Admin"]))
):
    """Returns the list of ingested manual files."""
    return ingested_files_registry


# --- Endpoint 6: Embedded Front-end Single Page Application Dashboard ---

@app.get("/", response_class=HTMLResponse)
async def get_dashboard_ui():
    """Renders a stunning dark-mode glassmorphic observability and chat interface."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IntelliRoute Workspace & Observability Panel</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #020617;
            --card-bg: rgba(15, 23, 42, 0.45);
            --card-border: rgba(51, 65, 85, 0.35);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Glassmorphic Navbar */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--card-border);
            z-index: 10;
        }

        header h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-badge {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.2);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--success);
        }

        /* Grid Layout */
        main {
            display: grid;
            grid-template-columns: 320px 1fr 380px;
            flex: 1;
            height: calc(100vh - 75px);
            overflow: hidden;
        }

        /* Sidebar Sections */
        .sidebar {
            background: rgba(15, 23, 42, 0.25);
            border-right: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            gap: 1.5rem;
            overflow-y: auto;
        }

        h2 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #c7d2fe;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* File Upload Zone */
        .upload-card {
            backdrop-filter: blur(8px);
            background: var(--card-bg);
            border: 2px dashed var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-card:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .upload-card p {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        .upload-card input {
            display: none;
        }

        /* File list */
        .file-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .file-item {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }

        .file-item-name {
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 180px;
        }

        .file-item-badge {
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        /* Chat Panel */
        .chat-container {
            display: flex;
            flex-direction: column;
            background: rgba(2, 6, 23, 0.5);
            position: relative;
            height: 100%;
            overflow: hidden;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .message {
            display: flex;
            flex-direction: column;
            max-width: 80%;
            border-radius: 12px;
            padding: 1rem;
            font-size: 0.95rem;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: #ffffff;
            border-bottom-right-radius: 2px;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .message.assistant {
            align-self: flex-start;
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            border-bottom-left-radius: 2px;
        }

        .message-reasoning {
            margin-top: 0.75rem;
            font-size: 0.8rem;
            color: #a5b4fc;
            background: rgba(99, 102, 241, 0.1);
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            border-left: 2px solid var(--primary);
        }

        .message-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.4rem;
            display: flex;
            gap: 1rem;
        }

        /* Chat input */
        .chat-input-area {
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.4);
            border-top: 1px solid var(--card-border);
            display: flex;
            gap: 0.75rem;
        }

        .chat-input {
            flex: 1;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.85rem 1.2rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        .send-btn {
            background: var(--primary);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0 1.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .send-btn:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        /* Metrics Observability Sidebar */
        .metrics-sidebar {
            background: rgba(15, 23, 42, 0.25);
            border-left: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            gap: 1.5rem;
            overflow-y: auto;
        }

        /* KPI Card grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }

        .kpi-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .kpi-title {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        .kpi-val {
            font-size: 1.25rem;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Transaction log list */
        .transaction-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .tx-card {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            font-size: 0.8rem;
        }

        .tx-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .tx-badge {
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.65rem;
            text-transform: uppercase;
        }

        .tx-badge.flash {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
        }

        .tx-badge.pro {
            background: rgba(245, 158, 11, 0.2);
            color: #fde047;
        }

        .tx-badge.cached {
            background: rgba(16, 185, 129, 0.2);
            color: #6ee7b7;
        }

        .tx-query {
            font-weight: 500;
            color: var(--text-main);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .tx-stats {
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 0.7rem;
        }

        .tx-cost {
            color: var(--success);
            font-weight: 600;
        }
    </style>
</head>
<body>

    <header>
        <h1>IntelliRoute Assistant Panel</h1>
    </header>

    <main>
        <!-- Sidebar - Ingestion -->
        <section class="sidebar">
            <div>
                <h2>Manuals Ingestion</h2>
                <div class="upload-card" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" onchange="handleFileUpload(event)">
                    <p style="font-weight: 500; color: #a5b4fc; margin-top: 0.25rem;">Drop manual file here</p>
                    <p>Supports .md or .pdf</p>
                </div>
            </div>

            <div>
                <h2>Ingested Documents</h2>
                <div class="file-list" id="fileList">
                    <p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 1rem;">No manuals loaded yet.</p>
                </div>
            </div>
        </section>

        <!-- Center Panel - Chat -->
        <section class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    Hello! I am IntelliRoute, your Adaptive Enterprise Assistant. Upload document manuals on the left, and ask me any configurations or technical setup questions!
                </div>
            </div>
            
            <div class="chat-input-area">
                <input type="text" class="chat-input" id="chatInput" placeholder="Ask a question about the configurations..." onkeydown="if(event.key === 'Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </section>

        <!-- Sidebar - Observability Metrics -->
        <section class="metrics-sidebar">
            <div>
                <h2>Session Metrics</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <span class="kpi-title">Total Cost</span>
                        <span class="kpi-val" id="kpiCost">$0.0000</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-title">Avg Latency</span>
                        <span class="kpi-val" id="kpiLatency">0.00s</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-title">Total Calls</span>
                        <span class="kpi-val" id="kpiCalls">0</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-title">Cache Rate</span>
                        <span class="kpi-val" id="kpiCache">0%</span>
                    </div>
                </div>
            </div>

            <div>
                <h2>Router Transaction Log</h2>
                <div class="transaction-list" id="transactionList">
                    <p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 1rem;">No transactions logged.</p>
                </div>
            </div>
        </section>
    </main>

    <script>
        const apiKey = "{DEFAULT_API_KEY}";
        function getApiKey() {
            return apiKey;
        }

        // Fetch current documents and metrics on load
        window.onload = () => {
            fetchDocuments();
            fetchMetrics();
        };

        // Ingestion handler
        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append("file", file);

            // Show temporary loading indicator in file list
            const fileList = document.getElementById("fileList");
            const loadingItem = document.createElement("div");
            loadingItem.className = "file-item";
            loadingItem.innerHTML = `<span class="file-item-name">Ingesting ${file.name}...</span>`;
            fileList.prepend(loadingItem);

            try {
                const response = await fetch("/api/ingest", {
                    method: "POST",
                    headers: {
                        "X-API-Key": getApiKey()
                    },
                    body: formData
                });
                const result = await response.json();
                
                if (result.status === "success") {
                    fetchDocuments();
                } else {
                    alert("Ingestion error: " + result.message);
                }
            } catch (err) {
                alert("Upload failed: " + err.message);
                fetchDocuments();
            }
        }

        // Fetch documents
        async function fetchDocuments() {
            const response = await fetch("/api/documents", {
                headers: {
                    "X-API-Key": getApiKey()
                }
            });
            const data = await response.json();
            const fileList = document.getElementById("fileList");
            
            if (data.length === 0) {
                fileList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 1rem;">No manuals loaded yet.</p>';
                return;
            }

            fileList.innerHTML = data.map(doc => `
                <div class="file-item">
                    <div>
                        <div class="file-item-name" title="${doc.filename}">${doc.filename}</div>
                        <div style="font-size: 0.7rem; color: var(--text-muted)">${doc.chunks_count} chunks</div>
                    </div>
                    <span class="file-item-badge">${doc.file_type}</span>
                </div>
            `).join('');
        }

        // Fetch metrics
        async function fetchMetrics() {
            // Overall summary
            const responseSum = await fetch("/api/metrics", {
                headers: {
                    "X-API-Key": getApiKey()
                }
            });
            const sum = await responseSum.json();
            
            // Calculate cache hit rate
            const cacheHits = sum.model_distribution.semantic_cache || 0;
            const total = sum.total_queries;
            const cacheRate = total > 0 ? Math.round((cacheHits / total) * 100) : 0;

            document.getElementById("kpiCost").innerText = `$${sum.total_cost.toFixed(4)}`;
            document.getElementById("kpiLatency").innerText = `${sum.avg_latency.toFixed(2)}s`;
            document.getElementById("kpiCalls").innerText = sum.total_queries;
            document.getElementById("kpiCache").innerText = `${cacheRate}%`;

            // Detailed transactions
            const responseTx = await fetch("/api/metrics/transactions", {
                headers: {
                    "X-API-Key": getApiKey()
                }
            });
            const txs = await responseTx.json();
            const txList = document.getElementById("transactionList");

            if (txs.length === 0) {
                txList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 1rem;">No transactions logged.</p>';
                return;
            }

            txList.innerHTML = txs.map(tx => {
                let badgeClass = "flash";
                if (tx.chosen_model.includes("pro")) badgeClass = "pro";
                if (tx.chosen_model === "semantic_cache") badgeClass = "cached";

                return `
                    <div class="tx-card">
                        <div class="tx-header">
                            <span class="tx-query" title="${tx.query}">${tx.query}</span>
                            <span class="tx-badge ${badgeClass}">${tx.chosen_model.replace("gemini-", "")}</span>
                        </div>
                        <div class="tx-stats">
                            <span>Latency: ${tx.latency_sec.toFixed(2)}s</span>
                            <span class="tx-cost">$${tx.cost_usd.toFixed(5)}</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Send chat message
        async function sendMessage() {
            const input = document.getElementById("chatInput");
            const query = input.value.trim();
            if (!query) return;

            input.value = "";
            const chatMessages = document.getElementById("chatMessages");
            
            // Add user message
            const userMsg = document.createElement("div");
            userMsg.className = "message user";
            userMsg.innerText = query;
            chatMessages.appendChild(userMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Add typing indicator
            const typingMsg = document.createElement("div");
            typingMsg.className = "message assistant";
            typingMsg.innerText = "Computing adaptive routing strategy...";
            chatMessages.appendChild(typingMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const response = await fetch("/api/chat", {
                    method: "POST",
                    headers: { 
                        "Content-Type": "application/json",
                        "X-API-Key": getApiKey()
                    },
                    body: JSON.stringify({ query: query, session_id: "default" })
                });
                const result = await response.json();
                
                // Remove typing indicator
                chatMessages.removeChild(typingMsg);

                // Add assistant response
                const assistMsg = document.createElement("div");
                assistMsg.className = "message assistant";
                
                let answerHtml = `<div>${result.answer}</div>`;
                
                // Add routing strategy details
                if (result.metrics) {
                    const routingLabel = result.cache_hit 
                        ? `⚡ Retrieved from Semantic Cache (${result.routing_reasoning})`
                        : `⚙️ Routed via ${result.strategy} search to ${result.metrics.chosen_model.replace("gemini-", "").toUpperCase()} (${result.routing_reasoning})`;
                        
                    answerHtml += `
                        <div class="message-reasoning">
                            <strong>Routing reasoning:</strong> ${routingLabel}
                        </div>
                        <div class="message-meta">
                            <span>Latency: ${result.metrics.latency_sec.toFixed(2)}s</span>
                            <span>Tokens: ${result.metrics.prompt_tokens + result.metrics.completion_tokens}</span>
                            <span>Cost: $${result.metrics.cost_usd.toFixed(5)}</span>
                        </div>
                    `;
                }

                assistMsg.innerHTML = answerHtml;
                chatMessages.appendChild(assistMsg);
                chatMessages.scrollTop = chatMessages.scrollHeight;

                // Refresh dashboard metrics
                fetchMetrics();
            } catch (err) {
                chatMessages.removeChild(typingMsg);
                const errMsg = document.createElement("div");
                errMsg.className = "message assistant";
                errMsg.innerText = "Error: " + err.message;
                chatMessages.appendChild(errMsg);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    </script>
</body>
</html>
    """
    return html_content.replace("{DEFAULT_API_KEY}", api_key or "")
