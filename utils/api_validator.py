"""
API Validation and Error Handling utilities.
Checks API availability and handles errors gracefully.
"""

from typing import Optional, Dict, Tuple
import time

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
except ImportError:
    try:
        import openai
        OpenAI = None
        APIError = Exception
        RateLimitError = Exception
        APIConnectionError = Exception
        APITimeoutError = Exception
    except ImportError:
        OpenAI = None
        APIError = Exception
        RateLimitError = Exception
        APIConnectionError = Exception
        APITimeoutError = Exception


class APIValidator:
    """
    Validates API keys and checks API availability.
    Provides error handling and fallback mechanisms.
    """
    
    @staticmethod
    def validate_openai_key(api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Validate OpenAI API key by making a simple test call.
        
        Args:
            api_key: OpenAI API key to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key or not api_key.strip():
            return False, "API key is empty"
        
        try:
            if OpenAI:
                client = OpenAI(api_key=api_key)
                # Make a minimal test call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                return True, None
            else:
                # Fallback to old API
                import openai
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                return True, None
        
        except RateLimitError as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                return False, "API quota exceeded. Please check your OpenAI billing and plan."
            return False, f"Rate limit error: {error_msg}"
        
        except APIError as e:
            error_msg = str(e)
            error_code = getattr(e, 'code', None)
            
            if error_code == 'insufficient_quota' or 'quota' in error_msg.lower():
                return False, "API quota exceeded. Please check your OpenAI billing and plan."
            elif error_code == 'invalid_api_key' or 'authentication' in error_msg.lower():
                return False, "Invalid API key. Please check your OpenAI API key."
            elif error_code == 'rate_limit_exceeded':
                return False, "Rate limit exceeded. Please wait before trying again."
            else:
                return False, f"API error: {error_msg}"
        
        except APIConnectionError as e:
            return False, f"Connection error: {str(e)}. Please check your internet connection."
        
        except APITimeoutError as e:
            return False, f"Request timeout: {str(e)}. Please try again."
        
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                return False, "API quota exceeded. Please check your OpenAI billing and plan."
            return False, f"Unexpected error: {error_msg}"
    
    @staticmethod
    def handle_openai_error(error: Exception, operation: str = "API call") -> str:
        """
        Handle OpenAI API errors and return user-friendly messages.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            
        Returns:
            User-friendly error message
        """
        error_str = str(error)
        error_type = type(error).__name__
        
        # Check for quota errors
        if "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
            return f"❌ {operation} failed: API quota exceeded. Please check your OpenAI billing and plan. Consider switching to local models in the configuration."
        
        # Check for rate limit errors
        if "rate_limit" in error_str.lower() or "429" in error_str:
            return f"⚠️ {operation} failed: Rate limit exceeded. Please wait a moment and try again, or switch to local models."
        
        # Check for authentication errors
        if "invalid_api_key" in error_str.lower() or "authentication" in error_str.lower():
            return f"❌ {operation} failed: Invalid API key. Please check your OpenAI API key in the configuration."
        
        # Check for connection errors
        if "connection" in error_str.lower() or "network" in error_str.lower():
            return f"⚠️ {operation} failed: Connection error. Please check your internet connection."
        
        # Generic error
        return f"❌ {operation} failed: {error_str}. Consider switching to local models if this persists."
    
    @staticmethod
    def is_quota_error(error: Exception) -> bool:
        """
        Check if error is a quota-related error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if quota error, False otherwise
        """
        error_str = str(error)
        return "quota" in error_str.lower() or "insufficient_quota" in error_str.lower()
    
    @staticmethod
    def is_rate_limit_error(error: Exception) -> bool:
        """
        Check if error is a rate limit error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if rate limit error, False otherwise
        """
        error_str = str(error)
        return "rate_limit" in error_str.lower() or "429" in error_str

