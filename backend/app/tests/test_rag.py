import os
import sys
import json
import time

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# Also check if backend is direct parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.config import settings
from backend.app.services.ingestion import (
    parse_markdown,
    build_markdown_structural_tree,
    RecursiveCharacterTextSplitter,
    TreeNode
)
from backend.app.services.retrieval import (
    VectorRetriever,
    PageIndexRetriever,
    RelevanceReranker,
    RetrievalRouter
)

def create_mock_markdown(file_path: str):
    """Creates a mock markdown file simulating an enterprise service manual."""
    content = """# IntelliRoute Enterprise Integration Guide

## Chapter 1: Introduction and System Overview
This guide provides complete specifications for integrating the IntelliRoute Adaptive Knowledge Engine.
IntelliRoute is designed for enterprise grade assistants that require high speed, low cost, and adaptive accuracy.

### 1.1 Architecture Highlights
- Multi-model routing: dynamically chooses model based on query difficulty.
- Dual retrieval routing: switches between Vector DB search and PageIndex structural tree navigation.
- Semantic cache: skips duplicate LLM calls to minimize latency and token billing.

## Chapter 2: Security & Authentication Specification
All connections to IntelliRoute APIs must be signed and authorized.

### 2.1 API Key Management
Enterprise administrators can generate API keys from the admin dashboard. Keys must be passed in the `X-API-Key` HTTP request header.
To revoke a key, update the key status to `revoked` in the DB or click the revoke button.

### 2.2 Role Based Access Control (RBAC)
There are three standard roles supported:
1. `Reader`: Can query search indices and chat.
2. `Writer`: Can upload and parse manuals/documents.
3. `Admin`: Full permissions, including API key generation and system configuration edits.

## Chapter 3: Performance Tuning and Caching
To maintain high responsiveness, caching must be properly configured.

### 3.1 Semantic Caching
The semantic cache uses a similarity threshold of 0.85 (cosine distance). If a query matches a cached question with similarity >= 0.85, the cached response is returned.

### 3.2 Token Bucket Rate Limiter
The rate limiter enforces limits per API key:
- Burst limit: 60 tokens (requests)
- Refill rate: 1 token per second
"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created mock markdown file at {file_path}")

def run_test():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mock_file = os.path.join(test_dir, "test_data", "integration_guide.md")
    create_mock_markdown(mock_file)
    
    # 1. Parse markdown for vector chunks
    print("\n--- 1. Parsing & Chunking Markdown ---")
    documents = parse_markdown(mock_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks from markdown.")
    for idx, chunk in enumerate(chunks[:2]):
        print(f"  Chunk {idx}: {chunk.text[:120].strip()}... [source: {chunk.metadata['source']}]")
        
    # 2. Build structural tree for PageIndex RAG
    print("\n--- 2. Building Structural PageIndex Tree ---")
    tree_root = build_markdown_structural_tree(mock_file)
    
    # Save the tree for observation/storage
    tree_save_path = os.path.join(test_dir, "test_data", "integration_guide_tree.json")
    with open(tree_save_path, "w", encoding="utf-8") as f:
        # Pydantic dump
        f.write(json.dumps(tree_root.model_dump(), indent=2))
    print(f"Saved structural tree to {tree_save_path}")
    print(f"Tree structure: {tree_root.title} (children count: {len(tree_root.children)})")
    for child in tree_root.children:
        print(f"  - {child.title} ({child.type}, children count: {len(child.children)})")

    # 3. Vector DB indexing
    print("\n--- 3. Initializing Vector DB and Adding Chunks ---")
    vector_retriever = VectorRetriever(
        persist_directory=settings.chroma_db_path,
        api_key=settings.gemini_api_key,
        embedding_model=settings.embedding_model
    )
    vector_retriever.add_documents(chunks)
    print("Documents successfully loaded into ChromaDB collection.")

    # 4. Initialize retrievers and router
    print("\n--- 4. Initializing Retrievers, Reranker & Router ---")
    page_retriever = PageIndexRetriever(api_key=settings.gemini_api_key)
    reranker = RelevanceReranker(api_key=settings.gemini_api_key)
    router = RetrievalRouter(api_key=settings.gemini_api_key)

    # 5. Execute Test Queries
    queries = [
        "How is the semantic cache similarity threshold configured?",
        "What does Chapter 2 cover about API Keys?",
        "Tell me about the dynamic features of IntelliRoute."
    ]

    for q_idx, query in enumerate(queries, 1):
        print(f"\n================ QUERY {q_idx} ================")
        print(f"Query: {query}")
        
        # Determine strategy via Router
        time.sleep(5)
        strategy, reasoning = router.route(query, has_tree=True)
        print(f"Routed Strategy: {strategy.upper()}")
        print(f"Reasoning: {reasoning}")
        
        # Execute based on strategy
        if strategy in ["vector", "both"]:
            print("\nExecuting Vector Search:")
            time.sleep(5)
            vect_results = vector_retriever.query(query, n_results=4)
            print(f"  Retrieved {len(vect_results)} vector chunks.")
            
            # Apply reranking
            print("  Applying Relevance Reranker...")
            time.sleep(5)
            reranked_results = reranker.rerank(query, vect_results, top_n=2)
            print("  Top Reranked Snippets:")
            for idx, r_doc in enumerate(reranked_results):
                print(f"    [{idx+1}] (from {r_doc.metadata['source']}): {r_doc.text[:150].strip()}...")
        
        if strategy in ["page_index", "both"]:
            print("\nExecuting PageIndex Structural Search:")
            time.sleep(5)
            p_result = page_retriever.traverse(tree_root, query)
            print(f"  Path taken: {p_result['path']}")
            print(f"  Final Node Title: {p_result['title']} (ID: {p_result['node_id']})")
            print("  Decision Logs:")
            for log in p_result["logs"]:
                print(f"    - {log}")
            print(f"  Retrieved Content Snippet:\n    {p_result['text'][:250].strip()}...")

if __name__ == "__main__":
    run_test()
