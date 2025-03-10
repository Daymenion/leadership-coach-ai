"""
Centralized OpenAI client utility.
Provides consistent OpenAI API access across the entire application without proxy issues.
Optimized for Turkish language processing.
"""

import os
import httpx
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OpenAIClient:
    """
    A robust OpenAI client that works without proxy issues.
    This is a centralized utility for all OpenAI API calls in the application.
    Optimized for Turkish language processing.
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60, max_retries: int = 3):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retries for failed API calls
        """
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. API calls will fail.")
            self.http_client = None
            return
            
        # API configuration
        self.base_url = "https://api.openai.com/v1"
        self.max_retries = max_retries
        self.retry_delay = 2  # seconds
        
        try:
            # Create HTTP client without proxy settings
            self.http_client = httpx.Client(timeout=timeout)
            logger.debug("Successfully initialized HTTP client for OpenAI API")
        except Exception as e:
            logger.error(f"Error initializing HTTP client: {str(e)}")
            self.http_client = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API calls."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _handle_api_error(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API error responses and provide detailed errors."""
        error_detail = "Unknown error"
        try:
            error_data = response.json()
            if 'error' in error_data:
                error_detail = error_data['error'].get('message', str(error_data))
            else:
                error_detail = str(error_data)
        except:
            error_detail = response.text or f"HTTP {response.status_code}"
            
        # Log detailed error information
        logger.error(f"OpenAI API error: {error_detail}")
        logger.error(f"Status code: {response.status_code}")
        
        return {"error": error_detail, "status_code": response.status_code}
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,  # Lower temperature for Turkish for more consistency
        max_tokens: int = 1000,
        stream: bool = False,
        retry_on_error: bool = True,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion via the OpenAI API.
        Optimized for Turkish language generation.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use for completion
            temperature: Temperature for generation (lower for Turkish)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            retry_on_error: Whether to retry on error
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            API response as a dictionary
        """
        if not self.http_client:
            logger.error("HTTP client not available")
            raise ValueError("OpenAI client not properly initialized. Check API key.")
            
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add response format if specified
        if response_format:
            payload["response_format"] = response_format
            
        # Retry logic for API calls
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.http_client.post(
                    url,
                    headers=self.get_headers(),
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_info = self._handle_api_error(response)
                    
                    # Decide whether to retry based on status code
                    if retry_on_error and (response.status_code in [429, 500, 502, 503, 504]):
                        retries += 1
                        if retries <= self.max_retries:
                            sleep_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                            logger.warning(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                            time.sleep(sleep_time)
                            continue
                    
                    # If we reach here, we're out of retries or shouldn't retry
                    raise ValueError(f"OpenAI API error: {error_info['error']}")
            except httpx.RequestError as e:
                logger.error(f"Network error when calling OpenAI API: {str(e)}")
                
                if retry_on_error and retries < self.max_retries:
                    retries += 1
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    raise ValueError(f"Network error when calling OpenAI API: {str(e)}")
        
        # If we reach here, all retries failed
        raise ValueError("All retries failed when calling OpenAI API")
    
    def embeddings(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        retry_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Create embeddings via the OpenAI API.
        
        Args:
            texts: Single text or list of texts to embed
            model: Model to use for embeddings
            dimensions: Optional output dimensions for the embeddings
            retry_on_error: Whether to retry on error
            
        Returns:
            API response with embeddings
        """
        if not self.http_client:
            logger.error("HTTP client not available")
            raise ValueError("OpenAI client not properly initialized. Check API key.")
            
        url = f"{self.base_url}/embeddings"
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        payload = {
            "model": model,
            "input": texts
        }
        
        # Add dimensions if specified
        if dimensions:
            payload["dimensions"] = dimensions
        
        # Retry logic for API calls
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.http_client.post(
                    url,
                    headers=self.get_headers(),
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_info = self._handle_api_error(response)
                    
                    # Decide whether to retry based on status code
                    if retry_on_error and (response.status_code in [429, 500, 502, 503, 504]):
                        retries += 1
                        if retries <= self.max_retries:
                            sleep_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                            logger.warning(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                            time.sleep(sleep_time)
                            continue
                    
                    # If we reach here, we're out of retries or shouldn't retry
                    raise ValueError(f"OpenAI API error: {error_info['error']}")
            except httpx.RequestError as e:
                logger.error(f"Network error when calling OpenAI API: {str(e)}")
                
                if retry_on_error and retries < self.max_retries:
                    retries += 1
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    raise ValueError(f"Network error when calling OpenAI API: {str(e)}")
        
        # If we reach here, all retries failed
        raise ValueError("All retries failed when calling OpenAI API")

# Create a singleton instance
_default_client = None

def get_client(api_key: Optional[str] = None) -> OpenAIClient:
    """
    Get a default OpenAI client instance (singleton pattern).
    
    Args:
        api_key: Optional API key (uses environment variable if not provided)
        
    Returns:
        OpenAIClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = OpenAIClient(api_key=api_key)
    return _default_client


# Simple test function
def test_client(turkish: bool = True):
    """
    Test the OpenAI client with a simple query.
    
    Args:
        turkish: Whether to test with Turkish (True) or English (False)
    """
    client = get_client()
    
    try:
        # Choose prompt based on language preference
        if turkish:
            system = "Sen bir liderlik koçusun."
            user = "Merhaba, nasılsın?"
        else:
            system = "You are a leadership coach."
            user = "Hello, how are you?"
            
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("API Response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        # Extract and print just the assistant's message
        assistant_message = response['choices'][0]['message']['content']
        print("\nAssistant's response:")
        print(assistant_message)
        
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_client(turkish=True) 