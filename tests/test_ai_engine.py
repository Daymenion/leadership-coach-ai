"""
Tests for the AI engine components.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.ai_engine.web_search import WebSearch
from src.ai_engine.openai_service import OpenAIService
from src.ai_engine.query_processor import QueryProcessor

class TestWebSearch:
    """Tests for the WebSearch class."""
    
    def test_should_use_web_search(self):
        """Test determining whether web search should be used."""
        web_search = WebSearch()
        
        # Test with no knowledge base results
        assert web_search.should_use_web_search("test query", []) is True
        
        # Test with low-quality knowledge base results
        kb_results = [
            {"score": 0.3},
            {"score": 0.4}
        ]
        assert web_search.should_use_web_search("test query", kb_results, min_score=0.6) is True
        
        # Test with high-quality knowledge base results
        kb_results = [
            {"score": 0.7},
            {"score": 0.8},
            {"score": 0.9}
        ]
        assert web_search.should_use_web_search("test query", kb_results, min_score=0.6) is False
        
        # Test with mixed-quality knowledge base results
        kb_results = [
            {"score": 0.7},
            {"score": 0.5}
        ]
        assert web_search.should_use_web_search("test query", kb_results, min_score=0.6, min_results=2) is True
        assert web_search.should_use_web_search("test query", kb_results, min_score=0.6, min_results=1) is False

class TestOpenAIService:
    """Tests for the OpenAIService class."""
    
    @patch('openai.OpenAI')
    def test_generate_response_format(self, mock_openai):
        """Test that response is formatted correctly."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create OpenAIService
        openai_service = OpenAIService()
        
        # Mock context chunks
        context_chunks = [
            {
                "video_id": "abc123",
                "video_title": "Test Video",
                "text": "Test text",
                "url_with_timestamp": "https://example.com"
            }
        ]
        
        # Generate response
        response = openai_service.generate_response("test query", context_chunks)
        
        # Check response format
        assert "response" in response
        assert "sources" in response
        assert "model" in response
        assert "usage" in response
        
        # Check response content
        assert response["response"] == "Test response"
        assert len(response["sources"]) == 1
        assert response["sources"][0]["type"] == "video"
        assert response["sources"][0]["title"] == "Test Video"
        assert response["model"] == "gpt-4o-mini"
        assert response["usage"]["prompt_tokens"] == 100
        assert response["usage"]["completion_tokens"] == 50
        assert response["usage"]["total_tokens"] == 150

class TestQueryProcessor:
    """Tests for the QueryProcessor class."""
    
    def test_process_query_metadata(self):
        """Test that query metadata is added to response."""
        # Mock dependencies
        vector_store = MagicMock()
        web_search = MagicMock()
        openai_service = MagicMock()
        
        # Mock search_knowledge_base
        kb_results = [{"text": "Test text"}]
        vector_store.search.return_value = kb_results
        
        # Mock should_use_web_search
        web_search.should_use_web_search.return_value = True
        
        # Mock search
        web_results = [{"title": "Test title", "content": "Test content", "url": "https://example.com"}]
        web_search.search.return_value = web_results
        
        # Mock generate_response
        openai_response = {
            "response": "Test response",
            "sources": [],
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        openai_service.generate_response.return_value = openai_response
        
        # Create QueryProcessor
        query_processor = QueryProcessor(
            vector_store=vector_store,
            web_search=web_search,
            openai_service=openai_service
        )
        
        # Process query
        response = query_processor.process_query("test query")
        
        # Check that metadata is added
        assert "query" in response
        assert "kb_results_count" in response
        assert "web_search_used" in response
        assert "web_results_count" in response
        
        # Check metadata values
        assert response["query"] == "test query"
        assert response["kb_results_count"] == 1
        assert response["web_search_used"] is True
        assert response["web_results_count"] == 1 