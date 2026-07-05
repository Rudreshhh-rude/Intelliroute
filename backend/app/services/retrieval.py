import re
import json
import os
import time
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from google import genai
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from backend.app.core.config import settings
from backend.app.services.ingestion import Document, TreeNode

def _call_gemini_with_retry(client, model, contents, config=None, max_retries=3, initial_delay=5):
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Immediately raise if it's a daily limit that can't be resolved with short sleeps
                if "requestsperday" in err_msg.lower().replace("_", "").replace("-", "") or "requests per day" in err_msg.lower():
                    print("Daily request limit exceeded. Skipping retries.")
                    raise e
                if attempt < max_retries:
                    print(f"Rate limited (429/ResourceExhausted). Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise e

_genai_client = None

def get_genai_client(api_key: str) -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client

def _parse_json_response(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        nl_idx = text.find("\n")
        if nl_idx != -1:
            text = text[nl_idx+1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=input
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            raise RuntimeError(f"Error calling Gemini Embedding API: {e}")

class VectorRetriever:
    def __init__(self, persist_directory: str, api_key: str, embedding_model: str):
        # Resolve DB path relative to root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
        db_path = os.path.abspath(os.path.join(root_dir, persist_directory))
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = GeminiEmbeddingFunction(api_key=api_key, model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name="intelliroute_chunks",
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: List[Document]):
        """Adds a list of standard chunk Documents to the vector collection."""
        if not documents:
            return
            
        ids = [f"chunk_{doc.metadata.get('source')}_{doc.metadata.get('chunk_index')}" for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = []
        
        # Chroma requires metadata values to be simple strings, ints, floats, or bools
        for doc in documents:
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    clean_meta[k] = ",".join(map(str, v))
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)
        
        # Add to Chroma in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    def query(self, query_text: str, n_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Queries the vector database for similar chunks."""
        chroma_filter = {}
        if filters:
            if len(filters) == 1:
                key, val = list(filters.items())[0]
                chroma_filter = {key: val}
            elif len(filters) > 1:
                chroma_filter = {"$and": [{k: v} for k, v in filters.items()]}
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=chroma_filter if chroma_filter else None
        )
        
        docs = []
        if results and results["documents"] and results["documents"][0]:
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                docs.append(Document(text=text, metadata=meta))
        return docs


class PageIndexRetriever:
    def __init__(self, api_key: str, model_name: str = settings.model_name_flash):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.tree = None

    def load_tree_index(self, path: str):
        """Loads a TreeNode tree outline from a saved JSON file path."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tree = TreeNode.model_validate(data)

    def get_sections_as_documents(self, traverse_res: Dict[str, Any]) -> List[Document]:
        """Converts traversed leaf node content to a standard Document list."""
        if not traverse_res or not traverse_res.get("text"):
            return []
        metadata = {
            "source": traverse_res.get("title", "unknown_section"),
            "node_id": traverse_res.get("node_id", "leaf"),
            "path": traverse_res.get("path", "")
        }
        return [Document(text=traverse_res["text"], metadata=metadata)]

    def traverse(self, root: TreeNode, query: str, max_depth: int = 5) -> Dict[str, Any]:
        """Traverses the structural tree of a document using LLM decisions.
        
        Returns the retrieved content and the path taken.
        """
        current_node = root
        path = [root.title]
        logs = []

        # Auto-dive optimization: if a node has exactly one child and that child has children,
        # skip the redundant choice step and auto-select the child.
        while len(current_node.children) == 1 and current_node.children[0].children:
            logs.append(f"Auto-diving from parent '{current_node.title}' to single child '{current_node.children[0].title}'")
            current_node = current_node.children[0]
            path.append(current_node.title)

        for depth in range(max_depth):
            if not current_node.children:
                logs.append(f"Reached leaf node: {current_node.title}")
                break

            # Format the options for the LLM
            options_text = ""
            for idx, child in enumerate(current_node.children):
                options_text += f"{idx + 1}. [ID: {child.id}] {child.title} - {child.summary or 'No summary'}\n"

            prompt = f"""You are an advanced document navigation agent. You are trying to find the section or page that contains the answer to the user's query.

User Query: "{query}"

Current Section: "{current_node.title}" (ID: {current_node.id})
Summary/Preview: {current_node.summary or "N/A"}

Below are the subsections or pages directly under the current section:
{options_text}

Instructions:
- Carefully evaluate the user query against the titles and summaries of the subsections.
- Select the single most relevant subsection number (1, 2, 3, etc.) that likely contains the answer.
- If none of the subsections are relevant, or if the current section itself is the best place to answer, reply with "NONE".
- Return ONLY the number (e.g., "3") or "NONE". Do not include any explanation or extra text.
"""
            try:
                response = _call_gemini_with_retry(
                    client=self.client,
                    model=self.model_name,
                    contents=prompt
                )
                choice = response.text.strip().upper()
                logs.append(f"At '{current_node.title}', LLM chose: {choice}")

                if "NONE" in choice:
                    logs.append("LLM decided to stop traversal.")
                    break

                # Extract first digit sequence
                match = re.search(r"\d+", choice)
                if match:
                    child_idx = int(match.group(0)) - 1
                    if 0 <= child_idx < len(current_node.children):
                        current_node = current_node.children[child_idx]
                        path.append(current_node.title)
                    else:
                        logs.append(f"LLM returned invalid index {child_idx + 1}. Stopping.")
                        break
                else:
                    logs.append(f"LLM output '{choice}' could not be parsed. Stopping.")
                    break
            except Exception as e:
                logs.append(f"Error during traversal: {e}. Stopping.")
                break

        return {
            "text": current_node.content or current_node.summary or "",
            "node_id": current_node.id,
            "title": current_node.title,
            "path": " -> ".join(path),
            "logs": logs
        }


class RelevanceReranker:
    def __init__(self, api_key: str, model_name: str = settings.model_name_flash):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """Reranks retrieved documents using a batch relevance scoring prompt in a single API call."""
        if not documents:
            return []
        if len(documents) == 1:
            return documents[:top_n]

        # Build list of snippets for prompt
        snippets_text = ""
        for idx, doc in enumerate(documents):
            snippets_text += f"Snippet {idx}:\n{doc.text}\n---\n"

        prompt = f"""You are evaluating search results relevance for the query: "{query}"

Below are several text snippets retrieved from the database. Please evaluate each snippet and assign a relevance score between 0 and 10 (where 10 is highly relevant and directly answers the query, and 0 is irrelevant).

Snippets list:
{snippets_text}

Instructions:
Return a JSON array of objects representing the scores. Each object must have:
- "index": The index integer of the snippet (e.g. 0, 1, 2)
- "score": The relevance score integer between 0 and 10

Return raw JSON only.
"""
        try:
            response = _call_gemini_with_retry(
                client=self.client,
                model=self.model_name,
                contents=prompt
            )
            scores_data = _parse_json_response(response.text)
            
            # Map scores back to docs
            scored_docs = []
            score_map = {item["index"]: item["score"] for item in scores_data if "index" in item and "score" in item}
            
            for idx, doc in enumerate(documents):
                score = score_map.get(idx, 0)
                scored_docs.append((score, doc))
                
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs[:top_n]]
            
        except Exception as e:
            print(f"Batch reranker failed: {e}. Falling back to default order.")
            return documents[:top_n]


class RetrievalRouter:
    def __init__(self, api_key: str, model_name: str = settings.model_name_flash):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def route(self, query: str, has_tree: bool) -> Tuple[str, str]:
        """Decides whether to route the query to 'vector', 'page_index', or 'both'.
        
        Logs the reasoning.
        """
        if not has_tree:
            return "vector", "No structural tree index is available for PageIndex retrieval; falling back to vector search."

        prompt = f"""Analyze the user's query and decide which retrieval strategy is best.

User Query: "{query}"

Retrieval Strategies:
1. "vector": Best for semantic, broad questions across multiple documents, or general factual retrieval.
2. "page_index": Best for structural queries referencing specific pages, sections, outlines, tables of contents, or navigating a single large manual (e.g., "What is in Chapter 4?", "According to page 12...").
3. "both": Best for complex queries that require both a broad search and structural layout navigation.

Instructions:
Select the best retrieval strategy.
Return a JSON object with two fields:
- "strategy": One of ["vector", "page_index", "both"]
- "reasoning": A brief sentence explaining why this strategy was chosen.

Format output as raw JSON only.
"""
        try:
            response = _call_gemini_with_retry(
                client=self.client,
                model=self.model_name,
                contents=prompt
            )
            data = _parse_json_response(response.text)
            strategy = data.get("strategy", "vector").lower()
            reasoning = data.get("reasoning", "Defaulting to vector search.")
            if strategy not in ["vector", "page_index", "both"]:
                strategy = "vector"
            return strategy, reasoning
        except Exception as e:
            return "vector", f"Error in retrieval router classification: {e}. Defaulting to vector."
