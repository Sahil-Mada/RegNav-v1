"""
Document Summarization utilities.
Generates summaries for documents using LLM or extractive methods.
"""

from typing import Optional, Dict
import re

try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = None


class DocumentSummarizer:
    """
    Generates summaries for documents.
    Supports OpenAI and Ollama LLMs, with fallback to extractive summarization.
    """
    
    def __init__(self,
                 llm_type: str = "openai",
                 openai_api_key: Optional[str] = None,
                 ollama_model: str = "llama2"):
        """
        Initialize the summarizer.
        
        Args:
            llm_type: "openai", "ollama", or "extractive"
            openai_api_key: OpenAI API key (required if llm_type is "openai")
            ollama_model: Model name for Ollama (required if llm_type is "ollama")
        """
        self.llm_type = llm_type
        self.ollama_model = ollama_model
        
        if llm_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI summarization")
            if OpenAI:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                import openai
                openai.api_key = openai_api_key
                self.openai_client = None
        else:
            self.openai_client = None
    
    def summarize_with_llm(self, text: str, max_length: int = 200) -> str:
        """
        Generate summary using LLM.
        
        Args:
            text: Document text to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Generated summary
        """
        # Truncate text if too long (to avoid token limits)
        # Rough approximation: 1 token = 4 characters
        max_chars = max_length * 4 * 10  # Allow for longer input
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        prompt = f"""Summarize the following regulatory document in {max_length} words or less. Focus on key regulatory requirements, scope, and important details.

Document:
{text}

Summary:"""
        
        if self.llm_type == "openai":
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a regulatory document summarization assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
            else:
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a regulatory document summarization assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
        
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
                return f"Error calling Ollama: {str(e)}"
        
        else:
            return self.extractive_summary(text, max_length)
    
    def extractive_summary(self, text: str, max_sentences: int = 5) -> str:
        """
        Generate extractive summary by selecting key sentences.
        Simple implementation - can be enhanced with more sophisticated methods.
        
        Args:
            text: Document text
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Simple heuristic: take first few sentences and last sentence
        if len(sentences) <= max_sentences:
            return ". ".join(sentences) + "."
        
        # Take first sentences and last sentence
        summary_sentences = sentences[:max_sentences-1] + [sentences[-1]]
        return ". ".join(summary_sentences) + "."
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Generate summary (uses LLM if available, otherwise extractive).
        
        Args:
            text: Document text
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if self.llm_type in ["openai", "ollama"]:
            return self.summarize_with_llm(text, max_length)
        else:
            return self.extractive_summary(text, max_length // 20)  # Approximate sentences

