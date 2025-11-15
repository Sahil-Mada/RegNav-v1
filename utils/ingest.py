"""
Document ingestion utilities.
Handles chunking, domain assignment, hierarchy classification, and embedding preparation.
"""

import uuid
from typing import List, Dict, Optional
from utils.router import DomainRouter
from utils.document_classifier import DocumentHierarchyClassifier


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens.
    Uses a simple character-based approach (roughly 4 chars = 1 token).
    Attempts to break at sentence boundaries.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens (~4 chars per token)
        overlap: Overlap between chunks in tokens
        
    Returns:
        List of text chunks
    """
    # Rough approximation: 4 characters per token
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + char_chunk_size, text_length)
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence endings near the end
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = chunk.rfind(punct)
                if last_punct > char_chunk_size - 200:  # If found reasonably close to end
                    chunk = chunk[:last_punct + 1]
                    end = start + len(chunk)
                    break
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - char_overlap
    
    return chunks


def assign_domain_to_chunk(chunk_text: str, router: DomainRouter) -> Dict[str, any]:
    """
    Assign a domain to a text chunk using the router.
    
    Args:
        chunk_text: Text chunk to classify
        router: DomainRouter instance
        
    Returns:
        Dictionary with domain assignment information
    """
    routing_result = router.route_query(chunk_text)
    
    return {
        "domain": routing_result["domain"],
        "subdomain": None,  # Can be enhanced to assign subdomain
        "confidence": routing_result["confidence"],
        "method": routing_result["method"]
    }


def process_document(
    text: str,
    file_name: str,
    file_id: Optional[str] = None,
    router: DomainRouter = None,
    hierarchy_classifier: DocumentHierarchyClassifier = None,
    hierarchy_level: Optional[str] = None,
    chunk_size: int = 800,
    overlap: int = 100
) -> Dict[str, any]:
    """
    Process a document: chunk text, assign domains, classify hierarchy, and prepare for embedding.
    
    Args:
        text: Extracted text from document
        file_name: Name of the source file
        file_id: Optional file ID (generated if not provided)
        router: DomainRouter instance (created if not provided)
        hierarchy_classifier: DocumentHierarchyClassifier instance (created if not provided)
        hierarchy_level: Pre-assigned hierarchy level (if None, will be classified)
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        Dictionary containing:
        - chunks: List of processed chunks with metadata
        - hierarchy_level: Assigned hierarchy level
        - hierarchy_classification: Full classification result
    """
    if router is None:
        router = DomainRouter()
    
    if hierarchy_classifier is None:
        hierarchy_classifier = DocumentHierarchyClassifier()
    
    if file_id is None:
        file_id = str(uuid.uuid4())
    
    # Classify hierarchy level if not provided
    if hierarchy_level is None:
        hierarchy_result = hierarchy_classifier.classify_document(text, file_name)
        hierarchy_level = hierarchy_result.get("level")
        hierarchy_classification = hierarchy_result
    else:
        hierarchy_classification = {
            "level": hierarchy_level,
            "confidence": 1.0,
            "method": "user_assigned"
        }
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    # Process each chunk
    processed_chunks = []
    for idx, chunk in enumerate(chunks):
        # Assign domain
        domain_assignment = assign_domain_to_chunk(chunk, router)
        
        # Create metadata (include hierarchy level)
        metadata = {
            "domain": domain_assignment["domain"],
            "subdomain": domain_assignment.get("subdomain"),
            "hierarchy_level": hierarchy_level,
            "file_id": file_id,
            "file_name": file_name,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "domain_confidence": domain_assignment["confidence"],
            "assignment_method": domain_assignment["method"]
        }
        
        # Create unique ID
        chunk_id = f"{file_id}_chunk_{idx}"
        
        processed_chunks.append({
            "text": chunk,
            "metadata": metadata,
            "id": chunk_id
        })
    
    return {
        "chunks": processed_chunks,
        "hierarchy_level": hierarchy_level,
        "hierarchy_classification": hierarchy_classification
    }

