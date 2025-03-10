"""
Module for interacting with OpenAI's GPT-4o-mini model.
Uses the centralized OpenAI client to avoid proxy issues.
Optimized for Turkish language interactions.
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from src.utils.openai_client import get_client
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Class for interacting with OpenAI's GPT-4o-mini model.
    Optimized for Turkish language interactions.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the OpenAIService.
        
        Args:
            model_name: The model to use for responses (default: gpt-4o-mini)
        """
        self.model_name = model_name
        self.client = get_client()
    
    def generate_response(
        self, 
        query: str, 
        context_chunks: List[Dict],
        web_results: Optional[List[Dict]] = None,
        temperature: float = 0.5,  # Lower temperature for more consistent Turkish responses
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query using retrieval-augmented generation.
        Optimized for Turkish language interactions.
        
        Args:
            query: The user's query in Turkish
            context_chunks: Retrieved context chunks from the knowledge base
            web_results: Optional web search results
            temperature: Temperature for response generation (lower for Turkish)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Response from the OpenAI API
        """
        try:
            # Check if there are any context chunks at all
            has_kb_context = any(chunk.get('source_type', '') != 'web' for chunk in context_chunks)
            has_web_context = any(chunk.get('source_type', '') == 'web' for chunk in context_chunks) or (web_results and len(web_results) > 0)
            
            logger.info(f"Generating response with: KB context: {has_kb_context}, Web context: {has_web_context}")
            
            # Format context chunks into a readable format, separating KB and web sources
            kb_context = ""
            web_context = ""
            
            # Process each context chunk
            for idx, chunk in enumerate(context_chunks, 1):
                score = chunk.get('score', 0)
                chunk_text = chunk.get('text', '')
                source_type = chunk.get('source_type', 'kb')
                
                if source_type == 'web':
                    # This is a web result
                    title = chunk.get('metadata', {}).get('title', 'Web Content')
                    url = chunk.get('source', '')
                    source_provider = chunk.get('metadata', {}).get('source', 'unknown')
                    
                    web_context += f"\nWeb Context {idx} [Relevance: {score:.2f}] [Source: {source_provider}] {title}\n"
                    web_context += f"{chunk_text}\n"
                    web_context += f"URL: {url}\n\n"
                else:
                    # This is a knowledge base result
                    video_title = chunk.get('video_title', 'Unknown Video')
                    url = chunk.get('url_with_timestamp', '')
                    
                    kb_context += f"\nYouTube Context {idx} [Relevance: {score:.2f}] {video_title}\n"
                    kb_context += f"{chunk_text}\n"
                    kb_context += f"Source: {url}\n\n"
            
            # Combine formatted contexts
            formatted_context = ""
            if kb_context:
                formatted_context += "\n--- YouTube Content Results ---\n" + kb_context
            if web_context:
                formatted_context += "\n--- Web Search Results ---\n" + web_context
            
            # Add additional web results that weren't included in context_chunks
            if web_results and len(web_results) > 0:
                web_results_not_in_context = [
                    r for r in web_results 
                    if not any(c.get('source', '') == r.get('url', '') for c in context_chunks)
                ]
                
                if web_results_not_in_context:
                    formatted_web_results = "\n--- Additional Web Sources ---\n"
                    for idx, result in enumerate(web_results_not_in_context, 1):
                        title = result.get('title', 'No Title')
                        snippet = result.get('snippet', '') or result.get('content', '')
                        url = result.get('url', '')
                        source_provider = result.get('source', 'web')
                        
                        formatted_web_results += f"{idx}. [{source_provider}] {title}\n"
                        formatted_web_results += f"{snippet}\n"
                        formatted_web_results += f"URL: {url}\n\n"
                    
                    formatted_context += formatted_web_results
            
            # Create system prompt with instructions on how to use different sources
            """
            7. ALWAYS INCLUDE REFERENCES to your sources at the end of your response
            8. For YouTube sources, include the video title and URL
            9. For web sources, include the title and URL"""
            # and how to include references in the response
            system_prompt = """
            You are "Leadership Coach", a professional leadership coach specializing in leadership development, team management, and professional growth.
            
            Your task is to provide expert guidance on leadership topics using a combination of curated YouTube content and up-to-date web search results. Always make your responses educational, practical, and evidence-based.
            
            IMPORTANT RULES:
            1. Answers should be in Turkish language using proper Turkish characters (ı, ğ, ü, ş, ö, ç, İ)
            2. Responses must be focused on leadership education and professional development
            3. If information isn't available in the context, honestly acknowledge the limitation
            4. Base answers on the provided context but reformulate in your own words
            5. Reject unprofessional or off-topic questions politely
            6. Make responses clear, concise, and valuable for the user

            Remember to be professional, respectful, and helpful in your responses.
            """
            
            # Combine user query with context and web results
            user_prompt = f"""
            Soru: {query}
            
            Bilgi Kaynakları:
            {formatted_context}
            """
            
            # Call OpenAI API
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract assistant's response
            assistant_response = ""
            if response and "choices" in response and len(response["choices"]) > 0:
                assistant_response = response["choices"][0]["message"]["content"]
            
            # Prepare sources for return in a structured format
            sources_used = []
            
            # Add KB sources
            kb_sources = [
                {
                    "type": "video",
                    "title": chunk.get("video_title", ""),
                    "url": chunk.get("url_with_timestamp", ""),
                    "text_snippet": chunk.get("text", "")[:150] + "..." if len(chunk.get("text", "")) > 150 else chunk.get("text", ""),
                    "score": chunk.get("score", 0)
                }
                for chunk in context_chunks
                if chunk.get("source_type", "") != "web" and chunk.get("video_title")
            ]
            
            # Add web sources
            web_sources = []
            
            # First add sources from context chunks
            for chunk in context_chunks:
                if chunk.get("source_type") == "web":
                    web_sources.append({
                        "type": "web",
                        "title": chunk.get("metadata", {}).get("title", "Web Content"),
                        "url": chunk.get("source", ""),
                        "text_snippet": chunk.get("text", "")[:150] + "..." if len(chunk.get("text", "")) > 150 else chunk.get("text", ""),
                        "score": chunk.get("score", 0),
                        "source_provider": chunk.get("metadata", {}).get("source", "web")
                    })
            
            # Then add additional web sources that might not be in context_chunks
            if web_results:
                for result in web_results:
                    # Check if this URL is already in sources
                    if not any(s.get("url") == result.get("url") for s in web_sources):
                        web_sources.append({
                            "type": "web",
                            "title": result.get("title", "Web Result"),
                            "url": result.get("url", ""),
                            "text_snippet": (result.get("content") or result.get("snippet", ""))[:150] + "..." 
                                if len(result.get("content") or result.get("snippet", "")) > 150 
                                else (result.get("content") or result.get("snippet", "")),
                            "score": result.get("relevance_score", 0),
                            "source_provider": result.get("source", "web")
                        })
            
            # Combine all sources
            sources_used = kb_sources + web_sources
            
            # Sort sources by score (highest first)
            sources_used.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Return formatted response
            return {
                "response": assistant_response,
                "context_used": [
                    {
                        "text": chunk.get("text", ""),
                        "score": chunk.get("score", 0),
                        "video_title": chunk.get("video_title", ""),
                        "url": chunk.get("url_with_timestamp", "")
                    }
                    for chunk in context_chunks if chunk.get("source_type", "") != "web"
                ],
                "web_results_used": web_sources,
                "sources": sources_used,
                "model": self.model_name,
                "success": True
            }
        
        except Exception as e:
            # Log the error and return an error response
            logging.error(f"Error generating response: {str(e)}")
            
            return {
                "response": "Yanıt oluşturulurken bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
                "error": str(e),
                "success": False
            }

# Test functionality
if __name__ == "__main__":
    service = OpenAIService()
    test_query = "Liderlik yeteneklerimi nasıl geliştirebilirim?"
    test_context = [
        {
            "text": "Liderlik yetenekleri pratik ve geri bildirim ile gelişir. Mentor bulun ve düzenli olarak kişisel gelişim planınızı gözden geçirin.",
            "score": 0.85,
            "video_title": "Liderlik Gelişimi 101",
            "url_with_timestamp": "https://youtube.com/watch?v=12345&t=120"
        }
    ]
    
    response = service.generate_response(test_query, test_context)
    print(response["response"])