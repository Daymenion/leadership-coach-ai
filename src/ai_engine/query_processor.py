"""
Module for processing user queries and coordinating responses.
"""

import os
import logging
import json
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple

from src.knowledge_base.vector_store import VectorStore
from src.ai_engine.web_search import WebSearch
from src.ai_engine.openai_service import OpenAIService

# Configure logging
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Class for processing user queries and coordinating responses.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        web_search: Optional[WebSearch] = None,
        openai_service: Optional[OpenAIService] = None,
        processed_dir: str = "data/processed"
    ):
        """
        Initialize the QueryProcessor.
        
        Args:
            vector_store: VectorStore instance
            web_search: WebSearch instance
            openai_service: OpenAIService instance
            processed_dir: Directory containing processed data
        """
        # Initialize vector store with proper error handling
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            try:
                self.vector_store = VectorStore(storage_dir=processed_dir)
            except Exception as e:
                logger.error(f"Error initializing VectorStore: {str(e)}")
                # Create a minimal vector store that won't cause attribute errors
                self.vector_store = vector_store or VectorStore(storage_dir=processed_dir)
        
        # Initialize web search with error handling
        if web_search is not None:
            self.web_search = web_search
        else:
            try:
                self.web_search = WebSearch(max_results=3)
            except Exception as e:
                logger.error(f"Error initializing WebSearch: {str(e)}")
                logger.error(traceback.format_exc())
                # Create a dummy web search that always returns empty results
                self.web_search = type('DummyWebSearch', (), {
                    'search': lambda *args, **kwargs: [],
                    'should_use_web_search': lambda *args, **kwargs: True,
                    'search_available': True
                })()
                logger.warning("Using dummy web search due to initialization error")
        
        self.openai_service = openai_service or OpenAIService()
        
        # Load vector data and chunks if available
        self._load_vector_data()
    
    def _load_vector_data(self):
        """Helper method to load vector data with robust error handling"""
        self.index = None
        self.chunks = None
        
        try:
            # Check for vectors file and metadata file
            vectors_file = os.path.join(self.vector_store.storage_dir, "vectors.npy") 
            metadata_file = os.path.join(self.vector_store.storage_dir, "metadata.json")
            chunks_file = os.path.join(self.vector_store.storage_dir, "..", "chunks", "transcript_chunks_improved.json")
            
            # If chunks file doesn't exist, try the regular transcript_chunks file
            if not os.path.exists(chunks_file):
                chunks_file = os.path.join(self.vector_store.storage_dir, "..", "chunks", "transcript_chunks.json")
            
            # Load chunks if they exist
            if os.path.exists(chunks_file):
                try:
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        self.chunks = json.load(f)
                    logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
                except Exception as e:
                    logger.error(f"Error loading chunks from {chunks_file}: {str(e)}")
                    self.chunks = []
            else:
                logger.warning(f"Chunks file not found at {chunks_file}")
                self.chunks = []
                
            # Validate vectors
            if self.vector_store.vectors is not None and len(self.vector_store.metadata) > 0:
                logger.info(f"Vector store already loaded with {len(self.vector_store.metadata)} vectors")
                
        except Exception as e:
            logger.error(f"Error loading vector data: {str(e)}")
            self.index = None
            self.chunks = []
    
    def process_query(
        self, 
        query: str,
        kb_results_count: int = 5,
        min_kb_results: int = 2,
        min_kb_score: float = 0.6,
        always_use_web: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query
            kb_results_count: Number of knowledge base results to retrieve
            min_kb_results: Minimum number of relevant KB results needed
            min_kb_score: Minimum relevance score threshold for KB results
            always_use_web: Whether to always use web search
            
        Returns:
            Dictionary with response text and sources
        """
        # Search knowledge base
        start_time = time.time()
        kb_results = self.search_knowledge_base(query, k=kb_results_count)
        kb_search_time = time.time() - start_time
        logger.info(f"Knowledge base search took {kb_search_time:.2f} seconds")
        
        # Analyze knowledge base result quality
        kb_scores = [result.get("score", 0) for result in kb_results]
        avg_kb_score = sum(kb_scores) / len(kb_scores) if kb_scores else 0
        good_kb_results = [r for r in kb_results if r.get("score", 0) >= min_kb_score]
        
        logger.info(f"Knowledge base results: {len(kb_results)} total, {len(good_kb_results)} good results")
        logger.info(f"Average KB result score: {avg_kb_score:.4f}")
        
        # Determine if web search is needed
        if len(good_kb_results) < min_kb_results:
            use_web_search = True
        else:
            use_web_search = False
        web_results = []
        web_search_time = 0
        
        
        try:
            if hasattr(self.web_search, 'search_available') and self.web_search.search_available:
                # Perform web search if needed
                if use_web_search:
                    try:
                        logger.info(f"Performing web search for query: {query}")
                        start_time = time.time()
                        web_results = self.web_search.search(query)
                        web_search_time = time.time() - start_time
                        
                        if web_results:
                            logger.info(f"Web search successful! Found {len(web_results)} results in {web_search_time:.2f} seconds")
                            
                            # Log a sample of the results for debugging
                            if len(web_results) > 0:
                                sample_result = web_results[0]
                                logger.info(f"Sample web result - Title: '{sample_result.get('title', '')[:50]}...', " +
                                           f"Source: {sample_result.get('source', 'unknown')}, " +
                                           f"URL: {sample_result.get('url', '')[:50]}...")
                        else:
                            logger.warning("Web search returned no results")
                    except Exception as e:
                        logger.error(f"Error performing web search: {str(e)}")
                        logger.error(traceback.format_exc())
                        logger.warning("Continuing with knowledge base results only")
                        web_results = []
        except Exception as e:
            logger.error(f"Error determining web search usage: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing with knowledge base results only")
            web_results = []
        
        # Combine and prioritize results for context
        context_results = self._prepare_context(kb_results, web_results)
        
        # Generate response
        try:
            response_data = self.openai_service.generate_response(
                query=query,
                context_chunks=context_results,
                web_results=web_results if web_results else None
            )
            
            # Add diagnostics to the response
            response_data["diagnostics"] = {
                "kb_results_found": len(kb_results),
                "kb_good_results": len(good_kb_results),
                "avg_kb_score": avg_kb_score,
                "web_results_found": len(web_results),
                "web_search_used": use_web_search,
                "web_search_sources": [result.get('source', 'unknown') for result in web_results[:3]] if web_results else [],
                "kb_search_time": kb_search_time,
                "web_search_time": web_search_time
            }
            
            return response_data
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a fallback response
            return {
                "response": f"I'm sorry, I encountered an error while processing your request. Error details: {str(e)}",
                "sources": [],
                "context_used": kb_results,
                "web_results_used": web_results,
                "success": False
            }
    
    def _prepare_context(self, kb_results: List[Dict], web_results: List[Dict]) -> List[Dict]:
        """
        Prepare and combine knowledge base and web results for context.
        
        Args:
            kb_results: Knowledge base search results
            web_results: Web search results
            
        Returns:
            Combined and prioritized context chunks
        """
        # Start with knowledge base results
        context = list(kb_results)
        
        # If no web results, just return KB results
        if not web_results:
            return context
            
        # Create context chunks from web results
        web_contexts = []
        for i, result in enumerate(web_results):
            # Create a context chunk for each web result
            web_context = {
                "chunk_id": f"web_{i}",
                "text": f"{result.get('title', 'No Title')}\n\n{result.get('content', result.get('snippet', 'No Content'))}",
                "source": result.get('url', 'Unknown Source'),
                "source_type": "web",
                "score": result.get('relevance_score', 0.5),  # Use the relevance score we calculated
                "metadata": {
                    "title": result.get('title', 'No Title'),
                    "url": result.get('url', 'Unknown URL'),
                    "source": result.get('source', 'web_search')
                }
            }
            web_contexts.append(web_context)
            
        # Add web contexts to the results
        context.extend(web_contexts)
        
        # Sort by score descending
        context.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return context
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search the knowledge base for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.vector_store is None or self.vector_store.vectors is None or len(self.vector_store.metadata) == 0:
            logger.warning("Vector store not initialized or empty")
            return []
        
        try:
            # Search the vector store
            results = self.vector_store.search(query, top_k=k)
            logger.info(f"Found {len(results)} results in knowledge base for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return []


if __name__ == "__main__":
    # Example usage
    processor = QueryProcessor()
    
    # Process a query
    query = "What are the key traits of successful leaders?"
    response_data = processor.process_query(query)
    
    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{response_data['response']}")
    print(f"\nSources used: {len(response_data.get('sources', []))}")
    for i, source in enumerate(response_data.get('sources', [])):
        print(f"  [{i+1}] {source.get('type', 'UNKNOWN').upper()}: {source.get('title', 'Untitled')}")