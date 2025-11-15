"""
Vector store utilities for ChromaDB with domain filtering.
Handles embedding generation and hierarchical retrieval.
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = None


class HierarchicalVectorStore:
    """
    Manages embeddings and vector database with domain-based filtering.
    Supports OpenAI embeddings and local sentence-transformers models.
    """
    
    def __init__(self,
                 embedding_type: str = "local",
                 openai_api_key: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the hierarchical vector store.
        
        Args:
            embedding_type: "openai" or "local"
            openai_api_key: OpenAI API key (required if embedding_type is "openai")
            model_name: Model name for local embeddings (sentence-transformers)
            persist_directory: Directory to persist ChromaDB data
        """
        self.embedding_type = embedding_type
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize embedding model
        if embedding_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            if OpenAI:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                # Fallback to old API
                import openai
                openai.api_key = openai_api_key
                self.openai_client = None
            self.embedding_model = None
        else:
            # Load local sentence-transformers model
            self.embedding_model = SentenceTransformer(model_name)
            self.openai_client = None
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="regulatory_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        if self.embedding_type == "openai":
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            else:
                # Fallback to old API
                import openai
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response['data'][0]['embedding']
        else:
            return self.embedding_model.encode(text).tolist()
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.embedding_type == "openai":
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [item.embedding for item in response.data]
            else:
                # Fallback to old API
                import openai
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [item['embedding'] for item in response['data']]
        else:
            return self.embedding_model.encode(texts).tolist()
    
    def add_documents(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        """
        Add documents to the vector database.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries for each chunk
            ids: List of unique IDs for each chunk
        """
        embeddings = self.get_embeddings(texts)
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_by_domain(self, query: str, domain: str, top_k: int = 5) -> List[dict]:
        """
        Hierarchical retrieval: search only within a specific domain.
        
        Args:
            query: Search query text
            domain: Domain to filter by
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing 'text', 'metadata', and 'distance'
        """
        query_embedding = self.get_embedding(query)
        
        # Filter by domain using where clause
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"domain": domain}  # Filter by domain metadata
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results
    
    def search_all(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search across all domains (fallback if domain filtering fails).
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing 'text', 'metadata', and 'distance'
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def get_domain_stats(self) -> Dict[str, int]:
        """
        Get statistics on documents per domain.
        
        Returns:
            Dictionary mapping domain names to document counts
        """
        # Get all documents to count by domain
        # Note: This is a simple implementation - for large collections,
        # consider using ChromaDB's get() with where filters
        all_results = self.collection.get()
        
        domain_counts = {}
        if all_results['metadatas']:
            for metadata in all_results['metadatas']:
                domain = metadata.get('domain', 'Unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return domain_counts
    
    def get_hierarchy_stats(self) -> Dict[str, int]:
        """
        Get statistics on documents per hierarchy level.
        
        Returns:
            Dictionary mapping hierarchy level names to document counts
        """
        all_results = self.collection.get()
        
        hierarchy_counts = {}
        if all_results['metadatas']:
            for metadata in all_results['metadatas']:
                hierarchy_level = metadata.get('hierarchy_level', 'Unknown')
                hierarchy_counts[hierarchy_level] = hierarchy_counts.get(hierarchy_level, 0) + 1
        
        return hierarchy_counts
    
    def get_documents_by_hierarchy(self, hierarchy_level: str) -> List[Dict[str, any]]:
        """
        Get all documents (unique files) for a specific hierarchy level.
        
        Args:
            hierarchy_level: Hierarchy level to filter by
            
        Returns:
            List of unique document information
        """
        all_results = self.collection.get(
            where={"hierarchy_level": hierarchy_level}
        )
        
        # Get unique file_ids
        unique_files = {}
        if all_results['metadatas'] and all_results['documents']:
            for i, metadata in enumerate(all_results['metadatas']):
                file_id = metadata.get('file_id')
                file_name = metadata.get('file_name', 'Unknown')
                if file_id and file_id not in unique_files:
                    unique_files[file_id] = {
                        'file_id': file_id,
                        'file_name': file_name,
                        'hierarchy_level': hierarchy_level,
                        'chunk_count': 0
                    }
                if file_id in unique_files:
                    unique_files[file_id]['chunk_count'] += 1
        
        return list(unique_files.values())
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(name="regulatory_documents")
        self.collection = self.client.get_or_create_collection(
            name="regulatory_documents",
            metadata={"hnsw:space": "cosine"}
        )

