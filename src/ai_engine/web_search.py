"""
Module for performing web searches to supplement the knowledge base.
Using multiple search engines with fallback mechanisms for reliability.
"""

import logging
import traceback
import time
import os
import re
import json
import urllib.parse
import httpx
import requests  # Add requests for direct content scraping
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Tuple
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from src.utils.openai_client import get_client

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class WebSearch:
    """
    Class for performing web searches to supplement the knowledge base.
    Uses multiple search engines with fallback mechanisms.
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 15, max_retries: int = 3):
        """
        Initialize the WebSearch.
        
        Args:
            max_results: Maximum number of search results to return
            timeout: Timeout for HTTP requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.max_results = max_results
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = 2  # seconds
        self.search_available = False
        self.client = get_client()
        
        # Get API keys from environment variables
        self.google_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
        self.google_cx = os.environ.get("GOOGLE_SEARCH_CX", "")  # Custom Search Engine ID
        self.bing_api_key = os.environ.get("BING_SEARCH_API_KEY", "")
        
        # Initialize HTTP client for searches with appropriate headers and timeouts
        try:
            self.http_client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LeadershipCoach/1.0"
                }
            )
            # Check if at least one search method is available
            if self.google_api_key and self.google_cx:
                logger.info("Google Custom Search API configured")
                self.search_available = True
            elif self.bing_api_key:
                logger.info("Bing Search API configured")
                self.search_available = True
            else:
                # Fallback to scraper mode
                logger.warning("No search API keys found, will attempt direct scraping (less reliable)")
                self.search_available = True
            
            logger.info("Web search client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing web search client: {str(e)}")
            logger.error(traceback.format_exc())
            self.http_client = None
            self.search_available = False
    
    def search(self, query: str, leadership_focused: bool = True) -> List[Dict]:
        """
        Perform a web search for the given query.
        
        Args:
            query: Search query
            leadership_focused: Whether to focus the search on leadership topics
            
        Returns:
            List of search result dictionaries
        """
        # Return empty results if search is not available
        if not self.search_available or self.http_client is None:
            logger.warning(f"Web search not available. Returning empty results for query: {query}")
            return []
        
        # Check if query is on-topic before searching
        enriched_query = self._enrich_query(query, True)
        if leadership_focused and enriched_query == -1:
            logger.info(f"Query determined to be off-topic: '{query}'. Returning empty results.")
            return []
        
        # Use the enriched query for searching
        search_query = enriched_query
        
        # Perform search using available methods with fallbacks
        try:
            results = []
            
            # Try Google Custom Search API first (if configured)
            if self.google_api_key and self.google_cx:
                logger.info(f"Using Google Custom Search API for query: {search_query}")
                results = self._google_cse_search(search_query)
            
            # If Google failed or not configured, try Bing
            if not results and self.bing_api_key:
                logger.info(f"Using Bing Search API for query: {search_query}")
                results = self._bing_search(search_query)
            
            # If API methods failed, try direct DuckDuckGo scraping as suggested
            if not results:
                logger.info(f"Using direct DuckDuckGo scraping for query: {search_query}")
                results = self._direct_ddg_scrape(search_query)
            
            # If all methods failed, return empty results
            if not results:
                logger.warning("All search methods failed. Returning empty results.")
                return []
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(results, query)
            
            logger.info(f"Found {len(filtered_results)} web search results after filtering")
            return filtered_results[:self.max_results]  # Limit to max_results
        
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _google_cse_search(self, query: str) -> List[Dict]:
        """
        Perform a search using Google's Custom Search JSON API.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            encoded_query = quote_plus(query)
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cx,
                "q": encoded_query,
                "num": min(10, self.max_results * 2)  # Request more than needed for filtering
            }
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = self.http_client.get(url, params=params)
                    response.raise_for_status()
                    break
                except httpx.HTTPError as e:
                    logger.warning(f"Google CSE API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            # Process response
            data = response.json()
            
            if "items" not in data:
                logger.warning(f"No search results found in Google CSE response")
                return []
            
            results = []
            for item in data["items"]:
                result = {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "content": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "source": "google"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Google CSE search error: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _bing_search(self, query: str) -> List[Dict]:
        """
        Perform a search using Bing's Search API.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            encoded_query = quote_plus(query)
            url = "https://api.bing.microsoft.com/v7.0/search"
            
            headers = {
                "Ocp-Apim-Subscription-Key": self.bing_api_key
            }
            
            params = {
                "q": encoded_query,
                "count": min(10, self.max_results * 2),  # Request more than needed for filtering
                "responseFilter": "webpages",
                "textDecorations": "false",
                "textFormat": "raw"
            }
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = httpx.get(url, headers=headers, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    break
                except httpx.HTTPError as e:
                    logger.warning(f"Bing API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            # Process response
            data = response.json()
            
            if "webPages" not in data or "value" not in data["webPages"]:
                logger.warning("No search results found in Bing response")
                return []
            
            results = []
            for item in data["webPages"]["value"]:
                result = {
                    "title": item.get("name", ""),
                    "snippet": item.get("snippet", ""),
                    "content": item.get("snippet", ""),
                    "url": item.get("url", ""),
                    "source": "bing"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Bing search error: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _get_real_url(self, ddg_url: str) -> str:
        """
        Extract the actual URL from DuckDuckGo's redirect URL.
        
        Args:
            ddg_url: DuckDuckGo redirect URL
            
        Returns:
            The actual destination URL
        """
        try:
            parsed_url = urlparse(ddg_url)
            query_params = parse_qs(parsed_url.query)
            if 'uddg' in query_params:
                return unquote(query_params['uddg'][0])
            return ddg_url
        except Exception as e:
            logger.error(f"Error extracting real URL from {ddg_url}: {str(e)}")
            return ddg_url
    
    def _direct_ddg_scrape(self, query: str) -> List[Dict]:
        """
        Direct implementation of DuckDuckGo search with content scraping as provided in sample code.
        
        Args:
            query: Search query
            
        Returns:
            List of search results with scraped content
        """
        logger.info(f"Running direct DuckDuckGo scraping for query: {query}")
        try:
            # Format the search URL
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            # Set up headers to avoid being detected as a bot
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            # Make the request using requests library for compatibility
            response = requests.get(search_url, headers=headers)
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo search failed with status code: {response.status_code}")
                return []
                
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize results list
            results = []
            
            # Find all result links
            links = []
            
            # Extract links from search results
            for a in soup.select('.result__a')[:self.max_results * 2]:  # Get more than needed for filtering
                href = a.get('href')
                if href:
                    real_url = self._get_real_url(href)
                    title = a.get_text(strip=True)
                    if real_url and title:
                        links.append((title, real_url))
            
            logger.info(f"Found {len(links)} links from DuckDuckGo search")
            
            # Scrape content from each link
            for i, (title, url) in enumerate(links[:self.max_results]):
                try:
                    logger.info(f"Scraping content from {url}")
                    
                    # Request the webpage content
                    page_resp = requests.get(url, headers=headers, timeout=10)
                    
                    if page_resp.status_code == 200:
                        page_soup = BeautifulSoup(page_resp.text, 'html.parser')
                        
                        # Remove script and style elements
                        for element in page_soup(['script', 'style', 'header', 'footer', 'nav', 'iframe', 'noscript']):
                            element.decompose()
                        
                        # Extract clean text
                        text = page_soup.get_text(separator=' ', strip=True)
                        
                        # Clean up whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        # Limit content to a reasonable size (2000 chars)
                        content = text[:2000] + ("..." if len(text) > 2000 else "")
                        
                        # Extract a snippet (first 200 chars of content)
                        snippet = content[:200] + ("..." if len(content) > 200 else "")
                        
                        # Find result snippet if available in search results
                        result_container = a.find_parent('.result')
                        if result_container:
                            snippet_element = result_container.select_one('.result__snippet')
                            if snippet_element:
                                snippet = snippet_element.get_text(strip=True)
                        
                        # Create result entry
                        result = {
                            "title": title,
                            "snippet": snippet,
                            "content": content,
                            "url": url,
                            "source": "ddg-direct"
                        }
                        
                        results.append(result)
                        logger.info(f"Successfully extracted content from {url}")
                    else:
                        logger.warning(f"Failed to scrape {url}, status code: {page_resp.status_code}")
                except Exception as e:
                    logger.warning(f"Error scraping content from {url}: {str(e)}")
                    continue
            
            logger.info(f"Successfully scraped content from {len(results)} websites")
            return results
            
        except Exception as e:
            logger.error(f"Direct DuckDuckGo scraping error: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _extract_content_from_webpage(self, url: str) -> Tuple[bool, str]:
        """
        Extracts main content from a webpage.
        
        Args:
            url: URL of the webpage
            
        Returns:
            Tuple containing (success, content)
        """
        try:
            # Make request with custom headers to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.9999.99 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }
            
            # Request the webpage
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"Error extracting content from {url}: HTTP status {response.status_code}")
                return False, ""
                
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, header, footer, and nav elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Get text and clean it up
            text = soup.get_text(separator=' ')
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Extract a reasonable amount of content (about 1000 characters)
            content = text[:1000] + ("..." if len(text) > 1000 else "")
            
            return True, content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return False, ""
    
    
    def _enrich_query(self, query: str, leadership_focused: bool = True) -> str:
        """
        Enrich the search query with additional context to improve results.
        First determines if the query is on-topic for leadership coaching,
        then enhances it with relevant context.
        
        Args:
            query: Original search query
            leadership_focused: Whether to focus on leadership topics
            
        Returns:
            Enhanced search query
        """
        try:
            # Skip enrichment if leadership focus is turned off
            if not leadership_focused:
                return query
                
            # Create prompt to determine if query is on-topic and enrich it
            system_prompt = """You are analyzing a query to determine if it's related to leadership, management, 
            professional development, team building, organizational behavior, or business coaching. 
            If it IS related, enhance the query by adding relevant keywords for better search results.
            If it's NOT related, simply respond with 'OFF-TOPIC' on a single line.
            
            For on-topic queries, your enhanced query should:
            1. Maintain the original intent
            2. Add relevant leadership/management terminology
            3. Be concise (under 100 characters if possible)
            4. Focus on factual information rather than opinions
            5. Format the response as 'ENHANCED: [your enhanced query]'"""
            
            # Call OpenAI API
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this query: {query}"}
                ],
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=100
            )
            
            # Extract assistant's response
            assistant_response = ""
            if response and "choices" in response and len(response["choices"]) > 0:
                assistant_response = response["choices"][0]["message"]["content"]
                
            # Check if it's off-topic
            if assistant_response.upper().startswith("OFF-TOPIC"):
                logger.info(f"Query '{query}' determined to be off-topic, using original query")
                return -1 # Signal off-topic
                
            # Extract enhanced query
            if assistant_response.upper().startswith("ENHANCED:"):
                enhanced_query = assistant_response[9:].strip()  # Remove 'ENHANCED: ' prefix
                logger.info(f"Enhanced query from '{query}' to '{enhanced_query}'")
                return enhanced_query
            
            # If format is unexpected, return original query
            logger.warning(f"Unexpected enrichment response format: '{assistant_response}', using original query")
            return query
                
        except Exception as e:
            logger.warning(f"Error enriching query: {str(e)}")
            logger.debug(traceback.format_exc())
            return query  # Return original query in case of any error
        
    def _filter_and_rank_results(self, results: List[Dict], original_query: str) -> List[Dict]:
        """
        Filter and rank search results based on relevance to the query.
        
        Args:
            results: List of search results
            original_query: The original user query
            
        Returns:
            Filtered and ranked results
        """
        if not results:
            return []
            
        filtered_results = []
        query_terms = set(original_query.lower().split())
        
        for result in results:
            # Skip if missing critical fields
            if not result.get('title'):
                continue
                
            # Extract or use snippet/content
            content = result.get('content') or result.get('snippet', '')
            
            # Calculate relevance score
            title = result.get('title', '').lower()
            content_lower = content.lower() if content else ''
            
            # Count term matches in title and content
            title_matches = sum(1 for term in query_terms if term in title)
            content_matches = sum(1 for term in query_terms if term in content_lower)
            
            # Calculate overall relevance score (title matches weighted more)
            relevance_score = (title_matches * 2 + content_matches) / (len(query_terms) * 3)
            
            # Add relevance score to result
            result['relevance_score'] = relevance_score
            
            # Standardize the content/snippet field
            if not result.get('snippet') and result.get('content'):
                result['snippet'] = result['content'][:200] + ("..." if len(result['content']) > 200 else "")
            elif not result.get('content') and result.get('snippet'):
                result['content'] = result['snippet']
            
            # Filter out results with very low relevance
            if relevance_score > 0.2:
                filtered_results.append(result)
        
        # Sort by relevance score
        filtered_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return filtered_results

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    web_search = WebSearch(max_results=3)
    results = web_search.search("effective leadership communication strategies")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content'][:150]}...")