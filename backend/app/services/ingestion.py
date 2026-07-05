import os
import re
import yaml
import pypdf
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any]

class TreeNode(BaseModel):
    id: str
    title: str
    type: str  # "document", "chapter", "section", "page", "leaf"
    summary: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    children: List["TreeNode"] = []

# Resolve self-reference for TreeNode
TreeNode.model_rebuild()

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        separator = separators[0]
        next_separators = separators[1:]

        # Split text by the current separator
        parts = text.split(separator)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for part in parts:
            part_len = len(part)
            
            # If adding this part exceeds chunk_size
            if current_length + part_len + (len(separator) if current_chunk else 0) <= self.chunk_size:
                current_chunk.append(part)
                current_length += part_len + (len(separator) if len(current_chunk) > 1 else 0)
            else:
                # Handle large single parts
                if part_len > self.chunk_size:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    sub_chunks = self._split_text_recursive(part, next_separators)
                    chunks.extend(sub_chunks)
                else:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    
                    # Backtrack to implement overlap
                    overlap_parts = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        p_len = len(p) + (len(separator) if overlap_parts else 0)
                        if overlap_len + p_len <= self.chunk_overlap:
                            overlap_parts.insert(0, p)
                            overlap_len += p_len
                        else:
                            break
                    
                    current_chunk = overlap_parts + [part]
                    current_length = overlap_len + part_len + (len(separator) if overlap_parts else 0)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in docs:
            chunks = self.split_text(doc.text)
            for idx, chunk in enumerate(chunks):
                meta = doc.metadata.copy()
                meta["chunk_index"] = idx
                chunked_docs.append(Document(text=chunk, metadata=meta))
        return chunked_docs


def parse_pdf(file_path: str) -> List[Document]:
    """Parses a PDF file page by page, extracting text and metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")

    documents = []
    filename = os.path.basename(file_path)
    
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text() or ""
            documents.append(Document(
                text=text,
                metadata={
                    "source": filename,
                    "file_path": file_path,
                    "page_number": page_num + 1,
                    "file_type": "pdf"
                }
            ))
            
    return documents


def parse_markdown(file_path: str) -> List[Document]:
    """Parses a Markdown file and yields a single Document (or split by sections if needed)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Markdown file not found at {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [Document(
        text=text,
        metadata={
            "source": filename,
            "file_path": file_path,
            "file_type": "md"
        }
    )]


def parse_yaml(file_path: str) -> List[Document]:
    """Parses a YAML file, structuring the content into a readable string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found at {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Format YAML data as key-value string blocks
    text_content = yaml.dump(data, default_flow_style=False)
    
    return [Document(
        text=text_content,
        metadata={
            "source": filename,
            "file_path": file_path,
            "file_type": "yaml",
            "parsed_data": data
        }
    )]


def build_pdf_structural_tree(file_path: str) -> TreeNode:
    """Builds a structural tree for a PDF document page-by-page.
    
    If the document has bookmarks, we can use them. Otherwise, we group pages
    into higher-level nodes or treat pages as sections. For a general PDF, 
    we treat pages as leaf nodes under a root document.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")

    filename = os.path.basename(file_path)
    root = TreeNode(
        id=f"root_{filename}",
        title=filename,
        type="document",
        summary=f"Full PDF document structure for {filename}."
    )
    
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        total_pages = len(reader.pages)
        
        # Let's extract first 200 characters from each page as an index-level preview/summary
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text = page.extract_text() or ""
            
            # Simple heuristic for title: first non-empty line of the page
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            page_title = lines[0] if lines else f"Page {page_num + 1}"
            if len(page_title) > 60:
                page_title = page_title[:57] + "..."
            
            summary_snippet = text[:250].replace("\n", " ").strip() + "..."
            
            page_node = TreeNode(
                id=f"page_{page_num + 1}",
                title=f"Page {page_num + 1}: {page_title}",
                type="page",
                summary=summary_snippet,
                content=text,
                metadata={
                    "page_number": page_num + 1,
                    "source": filename
                }
            )
            root.children.append(page_node)
            
    return root


def build_markdown_structural_tree(file_path: str) -> TreeNode:
    """Parses Markdown file structure based on heading headers (# H1, ## H2, etc.)
    and returns a hierarchical TreeNode.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Markdown file not found at {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    root = TreeNode(
        id=f"root_{filename}",
        title=filename,
        type="document",
        summary=f"Structured Markdown content from {filename}."
    )

    # Simple stack-based parsing of Markdown headings
    stack = [(0, root)]  # (header level, node)
    node_counter = 0

    current_text = []

    def flush_text(target_node: TreeNode):
        if current_text:
            text_str = "".join(current_text).strip()
            if text_str:
                # Add text to the current node content
                if target_node.content:
                    target_node.content += "\n\n" + text_str
                else:
                    target_node.content = text_str
                
                # Set summary from beginning of content
                snippet = text_str[:250].replace("\n", " ").strip()
                if len(text_str) > 250:
                    snippet += "..."
                target_node.summary = snippet
            current_text.clear()

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            # We found a header. First flush any accumulated text to the active node
            active_level, active_node = stack[-1]
            flush_text(active_node)

            level = len(match.group(1))
            title = match.group(2).strip()
            node_counter += 1
            
            new_node = TreeNode(
                id=f"sec_{node_counter}",
                title=title,
                type="section",
                metadata={"source": filename, "heading_level": level}
            )

            # Find the correct parent in stack
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent_node = stack[-1][1]
            parent_node.children.append(new_node)
            stack.append((level, new_node))
        else:
            current_text.append(line)

    # Flush remaining text
    if stack:
        flush_text(stack[-1][1])

    return root
