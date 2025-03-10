"""
Module for processing transcript chunks with LLM to improve grammar and vocabulary.
This module focuses on improving Turkish transcript chunks before embedding.
Uses the centralized OpenAI client to avoid proxy issues.
"""

import os
import json
import logging
import time
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.utils.openai_client import get_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChunkProcessor:
    """Class for processing transcript chunks with LLM to improve grammar."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the ChunkProcessor.
        
        Args:
            model: Model to use for grammar correction
        """
        self.model = model
        self.client = get_client()
        logger.info(f"ChunkProcessor initialized with model: {model}")
    
    def clean_response_text(self, text: str) -> str:
        """
        Clean and extract the text from LLM response.
        
        Args:
            text: The raw text from LLM response
            
        Returns:
            Cleaned text without artifacts
        """
        # Remove known pattern markers
        pattern_markers = [
            "Düzeltilmiş Metin:", 
            "İyileştirilmiş Metin:", 
            "Sonuç:", 
            "Türkçe Düzeltme:", 
            "Düzeltme:",
            "```",
            "İşte düzeltilmiş metin:"
        ]
        
        cleaned_text = text
        for marker in pattern_markers:
            if marker in cleaned_text:
                parts = cleaned_text.split(marker, 1)
                if len(parts) > 1:
                    cleaned_text = parts[1].strip()
        
        # Remove any remaining quotation marks and whitespace at beginning/end
        cleaned_text = cleaned_text.strip('" \n\t')
        
        return cleaned_text
    
    def improve_grammar(self, chunk: Dict[str, Any], retry_count: int = 2) -> Dict[str, Any]:
        """
        Improve the grammar and vocabulary of a Turkish transcript chunk.
        
        Args:
            chunk: A transcript chunk (dictionary with text and metadata)
            retry_count: Number of retries in case of failure
            
        Returns:
            Updated chunk with improved text
        """
        # If chunk has no text, return it as is
        if not chunk or "text" not in chunk or not chunk["text"]:
            logger.warning("Empty chunk or missing text field")
            return chunk
        
        # Store original text
        original_text = chunk["text"]
        
        # Skip short chunks (less than 20 characters)
        if len(original_text) < 20:
            logger.info(f"Skipping short chunk ({len(original_text)} chars)")
            return chunk
        
        # Prepare messages for API
        system_prompt = """
        Sen bir Türkçe dil uzmanısın. Görevin, video transkriptlerindeki Türkçe metinlerin dilbilgisi ve kelime kullanımını düzeltmektir.
        
        Lütfen aşağıdaki metni düzelt:
        1. Yazım hatalarını düzelt
        2. Dilbilgisi hatalarını düzelt 
        3. Kelime tekrarlarını azalt
        4. Akıcılığı artır
        5. Cümle yapılarını iyileştir
        6. Anlaşılabilirliği artır
        
        ÖNEMLİ: Sadece metni düzelt, içeriği veya anlamı değiştirme. Ekstra açıklama ekleme. Sadece düzeltilmiş metni döndür.
        """
        
        user_content = f"Metin: {original_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Track attempts
        attempt = 0
        max_attempts = retry_count + 1  # Initial attempt + retries
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # Call the OpenAI API through our centralized client
                start_time = time.time()
                
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,  # Lower temperature for more focused corrections
                    max_tokens=len(original_text) + 200  # Allow some extra tokens
                )
                
                elapsed_time = time.time() - start_time
                
                # Get the improved text from the response
                if "choices" in response and len(response["choices"]) > 0:
                    improved_text = response["choices"][0]["message"]["content"]
                    cleaned_text = self.clean_response_text(improved_text)
                    
                    # Update chunk with improved text
                    chunk["original_text"] = original_text
                    chunk["text"] = cleaned_text
                    
                    # Log success
                    logger.info(f"Grammar improvement completed in {elapsed_time:.2f}s")
                    
                    if cleaned_text != original_text:
                        logger.debug(f"Original: {original_text[:100]}...")
                        logger.debug(f"Improved: {cleaned_text[:100]}...")
                    else:
                        logger.debug("No changes made to the text")
                    
                    return chunk
                else:
                    logger.warning(f"Unexpected API response format: {response}")
                    if attempt < max_attempts:
                        logger.info(f"Retrying ({attempt}/{max_attempts})...")
                        time.sleep(2)  # Wait before retry
                    
            except Exception as e:
                logger.error(f"Error in grammar improvement (attempt {attempt}/{max_attempts}): {str(e)}")
                if attempt < max_attempts:
                    logger.info(f"Retrying in 3 seconds...")
                    time.sleep(3)  # Longer wait on error
        
        # If all attempts failed, return original chunk
        logger.warning(f"All grammar improvement attempts failed for chunk. Returning original.")
        return chunk
    
    def process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 10, delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process multiple chunks in batches.
        
        Args:
            chunks: List of chunks to process
            batch_size: Number of chunks to process in each batch
            delay: Delay between batches in seconds
            
        Returns:
            List of processed chunks with improved text
        """
        if not chunks:
            logger.warning("No chunks to process")
            return []
        
        total_chunks = len(chunks)
        logger.info(f"Starting to process {total_chunks} chunks in batches of {batch_size}")
        
        processed_chunks = []
        successful = 0
        failed = 0
        
        # Process chunks in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({len(batch)} chunks)")
            
            for chunk in batch:
                try:
                    processed_chunk = self.improve_grammar(chunk)
                    processed_chunks.append(processed_chunk)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to process chunk: {str(e)}")
                    processed_chunks.append(chunk)  # Keep original chunk
                    failed += 1
            
            if i + batch_size < total_chunks:
                logger.debug(f"Sleeping for {delay}s before next batch")
                time.sleep(delay)
        
        logger.info(f"Processing completed. Success: {successful}, Failed: {failed}, Total: {total_chunks}")
        return processed_chunks

def test_chunk_improvement(chunk_text: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Test the chunk improvement on a specific text.
    
    Args:
        chunk_text: Text to improve
        model: Model to use for grammar correction
        
    Returns:
        Improved text
    """
    # Create a sample chunk
    chunk = {"text": chunk_text}
    
    # Initialize the processor and improve grammar
    processor = ChunkProcessor(model=model)
    improved_chunk = processor.improve_grammar(chunk)
    
    # Check if improvement was successful
    if "original_text" in improved_chunk:
        logger.info("Grammar improvement successful")
        logger.info(f"Original: {improved_chunk['original_text']}")
        logger.info(f"Improved: {improved_chunk['text']}")
        return improved_chunk["text"]
    else:
        logger.warning("No improvement made")
        return chunk_text

if __name__ == "__main__":
    # Test with a sample Turkish text
    sample_text = (
        "bu videoda size liderlik becerilerini nasıl geliştireceğinizi anlatcağım. "
        "iyi bir lider olmak içinn öncelikle kendinize iyi bakmalısınız. sonra takım arkadaşlarınızı dinlemelisiniz "
        "ve onların fikirrlerine saygı göstermelisiniz"
    )
    
    improved_text = test_chunk_improvement(sample_text)
    print("\nImproved Text:")
    print(improved_text) 