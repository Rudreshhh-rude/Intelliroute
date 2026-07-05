import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.config import settings
from backend.app.services.ingestion import Document
from backend.app.services.retrieval import RelevanceReranker

def main():
    print("Initializing RelevanceReranker...")
    print("Using model:", settings.model_name_flash)
    reranker = RelevanceReranker(api_key=settings.gemini_api_key)
    
    docs = [
        Document(text="The semantic cache similarity threshold is configured to 0.85.", metadata={"source": "doc1"}),
        Document(text="IntelliRoute uses dual retrieval routing and multi-model routing.", metadata={"source": "doc2"}),
        Document(text="Authentication relies on X-API-Key headers.", metadata={"source": "doc3"}),
    ]
    
    print("Calling rerank...")
    try:
        results = reranker.rerank(
            query="How is the semantic cache similarity threshold configured?",
            documents=docs,
            top_n=2
        )
        print("Success! Reranked results:")
        for idx, doc in enumerate(results):
            print(f"  [{idx}] {doc.text} (source: {doc.metadata['source']})")
    except Exception as e:
        print("Failed with error:", e)

if __name__ == "__main__":
    main()
