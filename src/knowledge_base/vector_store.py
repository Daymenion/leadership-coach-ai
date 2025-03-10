"""
Module for creating and managing vector stores for the Leadership Coach knowledge base.
Optimized for Turkish language search and retrieval.
"""

import os
import logging
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from tqdm import tqdm
from src.utils.openai_client import get_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for efficient semantic search in transcript chunks.
    Optimized for Turkish language search.
    """
    
    def __init__(self, storage_dir: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to store vector files
            embedding_model: OpenAI embedding model to use
        """
        self.storage_dir = storage_dir
        self.embedding_model = embedding_model
        self.embedding_dimensions = 1536  # Default for OpenAI embeddings
        self.vectors_file = os.path.join(storage_dir, "vectors.npy")
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.client = get_client()
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize vectors and metadata if files exist
        self.vectors = self._load_vectors() if os.path.exists(self.vectors_file) else None
        self.metadata = self._load_metadata() if os.path.exists(self.metadata_file) else []
        
        logger.info(f"Vector store initialized at {storage_dir}")
        if self.vectors is not None:
            logger.info(f"Loaded {len(self.metadata)} existing vectors")
    
    def _load_vectors(self) -> np.ndarray:
        """Load vectors from file."""
        try:
            return np.load(self.vectors_file)
        except Exception as e:
            logger.error(f"Error loading vectors: {str(e)}")
            return np.array([])
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from file."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return []
    
    def _save_vectors(self, vectors: np.ndarray) -> bool:
        """Save vectors to file."""
        try:
            np.save(self.vectors_file, vectors)
            return True
        except Exception as e:
            logger.error(f"Error saving vectors: {str(e)}")
            return False
    
    def _save_metadata(self, metadata: List[Dict[str, Any]]) -> bool:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Create embedding for a text using OpenAI's API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Normalize text for better embedding
            text = text.replace("\n", " ").strip()
            
            # Call OpenAI's embedding API
            response = self.client.embeddings(
                texts=text,
                model=self.embedding_model
            )
            
            if response and "data" in response and len(response["data"]) > 0:
                embedding = response["data"][0]["embedding"]
                return np.array(embedding, dtype=np.float32)
            else:
                logger.warning(f"Unexpected embedding response format: {response}")
                return None
        
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries (must contain 'text' field)
            batch_size: Number of embeddings to create in each batch
            
        Returns:
            Success status
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return False
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Create empty arrays if no existing vectors
        if self.vectors is None:
            self.vectors = np.empty((0, self.embedding_dimensions), dtype=np.float32)
            self.metadata = []
        
        successful = 0
        total_batches = (len(documents) - 1) // batch_size + 1
        
        # Process documents in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(documents))
            batch = documents[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} documents)")
            
            new_vectors = []
            new_metadata = []
            
            for doc in tqdm(batch, desc=f"Batch {batch_idx + 1}"):
                try:
                    # Skip if text is missing or empty
                    if "text" not in doc or not doc["text"]:
                        logger.warning(f"Skipping document without text: {doc.get('video_id', 'unknown')}")
                        continue
                    
                    # Create embedding
                    embedding = self.create_embedding(doc["text"])
                    
                    if embedding is not None:
                        # Prepare metadata
                        metadata = {
                            "text": doc["text"],
                            "video_id": doc.get("video_id", ""),
                            "video_title": doc.get("video_title", ""),
                            "url_with_timestamp": doc.get("url_with_timestamp", ""),
                            "start_time": doc.get("start_time", 0),
                            "end_time": doc.get("end_time", 0),
                            "chunk_index": doc.get("chunk_index", 0)
                        }
                        
                        new_vectors.append(embedding)
                        new_metadata.append(metadata)
                        successful += 1
                    else:
                        logger.warning(f"Failed to create embedding for document: {doc.get('video_id', 'unknown')}")
                
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
            
            # Add new vectors and metadata to existing ones
            if new_vectors:
                new_vectors_array = np.array(new_vectors, dtype=np.float32)
                self.vectors = np.vstack([self.vectors, new_vectors_array])
                self.metadata.extend(new_metadata)
                
                # Save after each batch
                logger.info(f"Saving batch {batch_idx + 1} to vector store")
                self._save_vectors(self.vectors)
                self._save_metadata(self.metadata)
            
            # Sleep between batches to avoid rate limits
            if batch_idx < total_batches - 1:
                time.sleep(0.5)
        
        logger.info(f"Successfully added {successful} of {len(documents)} documents to vector store")
        return successful > 0
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for documents similar to the query.
        
        Args:
            query: Search query in Turkish
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        # Check if vector store has data
        if self.vectors is None or len(self.vectors) == 0 or not self.metadata:
            logger.warning("Vector store is empty. Cannot perform search.")
            return []
        
        try:
            logger.info(f"Searching for: '{query}'")
            
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            if query_embedding is None:
                logger.error("Failed to create embedding for search query")
                return []
            
            # Calculate cosine similarity
            similarities = self._calculate_similarities(query_embedding, self.vectors)
            
            # Get indices of top results
            if top_k > len(similarities):
                top_k = len(similarities)
                
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Create result documents
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                metadata = self.metadata[idx].copy()
                metadata["score"] = score
                results.append(metadata)
            
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def _calculate_similarities(self, query_embedding: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and vectors.
        
        Args:
            query_embedding: Query embedding vector
            vectors: Document embedding vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate dot product for each document vector
        similarities = np.zeros(vectors.shape[0])
        
        for i in range(vectors.shape[0]):
            # Get document vector
            doc_vector = vectors[i]
            
            # Normalize document vector
            doc_norm = np.linalg.norm(doc_vector)
            if doc_norm > 0:
                doc_vector = doc_vector / doc_norm
                
            # Calculate cosine similarity (dot product of normalized vectors)
            similarities[i] = np.dot(query_embedding, doc_vector)
            
        return similarities
    
    def get_document_by_id(self, video_id: str, chunk_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get documents by video ID and optional chunk index.
        
        Args:
            video_id: YouTube video ID
            chunk_index: Optional chunk index
            
        Returns:
            List of matching documents
        """
        if not self.metadata:
            return []
        
        results = []
        for i, meta in enumerate(self.metadata):
            if meta.get("video_id") == video_id:
                if chunk_index is None or meta.get("chunk_index") == chunk_index:
                    document = meta.copy()
                    results.append(document)
        
        return results

    def clear(self) -> bool:
        """
        Clear the vector store (delete all vectors and metadata).
        
        Returns:
            Success status
        """
        try:
            self.vectors = np.empty((0, self.embedding_dimensions), dtype=np.float32)
            self.metadata = []
            
            # Delete files if they exist
            if os.path.exists(self.vectors_file):
                os.remove(self.vectors_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            logger.info("Vector store cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Create vector store
    store = VectorStore("data/vector_store")
    
    # Test search
    test_query = "Liderlik becerileri nasıl geliştirilir?"
    results = store.search(test_query, top_k=3)
    
    print(f"Query: {test_query}")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.4f}):")
        print(f"Title: {result['video_title']}")
        print(f"Text: {result['text'][:150]}...")
        print(f"URL: {result['url_with_timestamp']}") 