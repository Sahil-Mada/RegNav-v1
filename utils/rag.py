"""
RAG (Retrieval-Augmented Generation) utilities for hierarchical retrieval.
Handles the two-step retrieval process and generation logic.
"""

import os
from typing import List, Optional, Dict
from utils.vectorstore import HierarchicalVectorStore
from utils.router import DomainRouter
from utils.api_validator import APIValidator

try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    try:
        import openai
        OpenAI = None
        APIError = Exception
        RateLimitError = Exception
    except ImportError:
        OpenAI = None
        APIError = Exception
        RateLimitError = Exception


class HierarchicalRAG:
    """
    Implements hierarchical RAG with two-step retrieval:
    1. Route query to domain
    2. Retrieve only from that domain
    """
    
    def __init__(self,
                 vector_store: HierarchicalVectorStore,
                 router: DomainRouter = None,
                 llm_type: str = "openai",
                 openai_api_key: Optional[str] = None,
                 ollama_model: str = "llama2"):
        """
        Initialize the hierarchical RAG pipeline.
        
        Args:
            vector_store: HierarchicalVectorStore instance
            router: DomainRouter instance (created if not provided)
            llm_type: "openai" or "ollama"
            openai_api_key: OpenAI API key (required if llm_type is "openai")
            ollama_model: Model name for Ollama (required if llm_type is "ollama")
        """
        self.vector_store = vector_store
        self.router = router or DomainRouter()
        self.llm_type = llm_type
        self.ollama_model = ollama_model
        
        if llm_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI LLM")
            if OpenAI:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                # Fallback to old API
                import openai
                openai.api_key = openai_api_key
                self.openai_client = None
        else:
            self.openai_client = None
    
    def route_query(self, query: str) -> Dict[str, any]:
        """
        Step 1: Route query to appropriate domain.
        
        Args:
            query: User query
            
        Returns:
            Routing result with predicted domain
        """
        return self.router.route_query(query)
    
    def retrieve_context(self, query: str, domain: str, top_k: int = 5) -> List[dict]:
        """
        Step 2: Retrieve context from the specified domain.
        
        Args:
            query: User query
            domain: Domain to retrieve from
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        return self.vector_store.search_by_domain(query, domain, top_k=top_k)
    
    def format_context(self, retrieved_chunks: List[dict]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_name', 'Unknown')
            domain = metadata.get('domain', 'Unknown')
            chunk_idx = metadata.get('chunk_index', '?')
            context_parts.append(
                f"[Chunk {i} from {source} (Domain: {domain}, Chunk {chunk_idx})]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, domain: str) -> str:
        """
        Generate a response using the LLM with domain-filtered context.
        
        Args:
            query: User question
            context: Retrieved context string
            domain: Domain that was used for retrieval
            
        Returns:
            Generated response
        """
        prompt = f"""You are an FDA regulatory assistant. You must answer using ONLY the provided domain-filtered context. If the context does not contain the answer, reply: 'The knowledge base does not contain this information.'

Domain: {domain}

Context:
{context}

Question: {query}

Answer:"""
        
        if self.llm_type == "openai":
            try:
                if self.openai_client:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an FDA regulatory assistant that answers questions based on provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return response.choices[0].message.content.strip()
                else:
                    # Fallback to old API
                    import openai
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an FDA regulatory assistant that answers questions based on provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return response.choices[0].message.content.strip()
            except (APIError, RateLimitError, Exception) as e:
                error_msg = APIValidator.handle_openai_error(e, "Answer generation")
                # Return a helpful message instead of crashing
                return f"{error_msg}\n\nPlease check your API key and billing status, or switch to local models (Ollama) in the configuration."
        
        elif self.llm_type == "ollama":
            try:
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception as e:
                return f"Error calling Ollama: {str(e)}. Make sure Ollama is running and the model is available."
        
        else:
            return "Invalid LLM type specified."
    
    def query(self, user_question: str, top_k: int = 5) -> Dict[str, any]:
        """
        Complete hierarchical RAG pipeline: route, retrieve, and generate.
        
        Args:
            user_question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with 'answer', 'context', 'domain', 'routing_result', and 'retrieved_chunks'
        """
        # Step 1: Route query to domain
        routing_result = self.route_query(user_question)
        predicted_domain = routing_result["domain"]
        
        # Step 2: Retrieve from predicted domain
        retrieved_chunks = self.retrieve_context(user_question, predicted_domain, top_k=top_k)
        
        # If no results in predicted domain, try fallback to all domains
        if not retrieved_chunks:
            retrieved_chunks = self.vector_store.search_all(user_question, top_k=top_k)
            if retrieved_chunks:
                # Update domain to match retrieved chunks
                predicted_domain = retrieved_chunks[0].get('metadata', {}).get('domain', predicted_domain)
        
        # Format context
        context = self.format_context(retrieved_chunks)
        
        # Generate response
        if context:
            answer = self.generate_response(user_question, context, predicted_domain)
        else:
            answer = "No relevant context found in the knowledge base. Please add documents first."
        
        return {
            'answer': answer,
            'context': context,
            'domain': predicted_domain,
            'routing_result': routing_result,
            'retrieved_chunks': retrieved_chunks
        }

