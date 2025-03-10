"""
Test script for grammar correction on Turkish transcript chunks.
Uses the centralized OpenAI client.
"""

import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv
from src.knowledge_base.chunk_processor import ChunkProcessor, test_chunk_improvement
from src.utils.openai_client import test_client

# Define log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'grammar_correction_test.log')

# Configure logging
logger = logging.getLogger("grammar_correction_test")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log startup message
logger.info(f"Starting grammar correction test, log file: {LOG_FILE}")

# Load environment variables
load_dotenv()

def test_with_text(text):
    """Test grammar correction on text."""
    logger.info("Testing grammar correction on a specific text sample")
    result = test_chunk_improvement(text)
    return result

def test_with_file(file_path):
    """Test grammar correction on chunks from a file."""
    logger.info(f"Testing grammar correction on chunks from file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    processor = ChunkProcessor()
    processed_chunks = processor.process_chunks(chunks, batch_size=5, delay=1.0)
    
    logger.info(f"Successfully processed {len(processed_chunks)} chunks")
    return processed_chunks

def test_with_samples():
    """Test grammar correction on sample chunks."""
    logger.info("Testing grammar correction on sample chunks")
    
    # Create sample chunks with Turkish text
    sample_chunks = [
        {
            "text": "Bu videoda liderlik üzerine konuşuyoruz. İyi bir lider olabilmek için, takımınızı dinlemelisiniz."
        },
        {
            "text": "Liderlik becerilerini geliştirmek istiyorsan, her gün pratiğini yapmalısın. Başarılı olmak için sabırlı olmalısın."
        }
    ]
    
    processor = ChunkProcessor()
    improved_chunks = processor.process_chunks(sample_chunks)
    
    for i, chunk in enumerate(improved_chunks, 1):
        logger.info(f"\nSample {i}:")
        logger.info(f"Original: {chunk.get('original_text', chunk.get('text', 'No text found'))}")
        logger.info(f"Improved: {chunk.get('text', 'No improvement found')}")
    
    return improved_chunks

def test_openai_connectivity():
    """Test OpenAI API connectivity."""
    logger.info("Testing OpenAI API connectivity")
    
    if test_client(turkish=True):
        logger.info("OpenAI API connection successful!")
        return True
    else:
        logger.error("OpenAI API connection failed!")
        return False

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Test grammar correction functionality")
    parser.add_argument("--text", type=str, help="Text to test grammar correction on")
    parser.add_argument("--file", type=str, help="JSON file with chunks to process")
    parser.add_argument("--samples", action="store_true", help="Test with sample chunks")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    # Test OpenAI connectivity first
    if not test_openai_connectivity():
        logger.error("Aborting grammar correction tests due to API connectivity issues")
        return
    
    if args.text:
        test_with_text(args.text)
    elif args.file:
        test_with_file(args.file)
    elif args.samples:
        test_with_samples()
    else:
        logger.info("No test option specified. Please use --text, --file, or --samples")
        parser.print_help()
    
    # Final message
    logger.info("Grammar correction test completed.")
    print(f"Log file created at: {LOG_FILE}")

if __name__ == "__main__":
    # Make sure logs are properly flushed even if the program crashes
    try:
        main()
    finally:
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logging.shutdown() 