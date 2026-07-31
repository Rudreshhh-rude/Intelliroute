import time
import uuid
import logging
from typing import Optional, Tuple
import chromadb
from chromadb.api.types import QueryResult
from backend.app.core.config import settings
from backend.app.services.retrieval import GeminiEmbeddingFunction

logger = logging.getLogger("intelliroute.cache")

class SemanticCache:
    def __init__(self, api_key: str, threshold: float = 0.85):
        self.threshold = threshold
        # chroma client
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.embedding_function = GeminiEmbeddingFunction(
            api_key=api_key,
            model_name=settings.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="intelliroute_cache",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def lookup(self, query: str, session_id: str = "default") -> Tuple[Optional[str], Optional[float]]:
        """Looks up a query in the cache.
        
        Returns (cached_response, similarity_score) if hit, else (None, None).
        """
        try:
            # Query the collection for 1 match
            results: QueryResult = self.collection.query(
                query_texts=[query],
                n_results=1,
                where={"session_id": session_id}
            )
            if not results or not results["documents"] or len(results["documents"][0]) == 0:
                return None, None
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            
            logger.info(f"Cache lookup similarity: {similarity:.4f} (distance: {distance:.4f}) for query '{query[:30]}...'")
            
            if similarity >= self.threshold:
                cached_response = results["documents"][0][0]
                logger.info(f"Cache HIT for query '{query[:30]}...' (Score: {similarity:.4f})")
                return cached_response, similarity
                
            logger.info(f"Cache MISS (below threshold) for query '{query[:30]}...' (Score: {similarity:.4f})")
            return None, similarity
        except Exception as e:
            logger.error(f"Error looking up query in semantic cache: {e}")
            return None, None

    def update(self, query: str, response: str, session_id: str = "default"):
        """Adds a query and its completed response to the cache."""
        try:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[response],
                metadatas=[{"original_query": query, "timestamp": time.time(), "session_id": session_id}],
                ids=[doc_id]
            )
            logger.info(f"Cache updated with new entry for query '{query[:30]}...'")
        except Exception as e:
            logger.error(f"Failed to update semantic cache: {e}")
