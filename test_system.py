"""
Comprehensive test script for LeadershipCoach system.
This script verifies that all components are functioning correctly with Turkish language support.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any
from dotenv import load_dotenv

# Define log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'system_test.log')

# Configure logging with explicit file handler
logger = logging.getLogger("system_test")
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
logger.info(f"Starting system test, log file: {LOG_FILE}")

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test OpenAI API connectivity and Turkish language support."""
    logger.info("Testing OpenAI API connectivity...")
    
    try:
        from src.utils.openai_client import test_client
        
        # Test with Turkish
        logger.info("Testing with Turkish prompt")
        success = test_client(turkish=True)
        
        if success:
            logger.info("OpenAI API connection successful with Turkish")
        else:
            logger.error("OpenAI API connection failed with Turkish")
            return False
        
        # Test with English
        logger.info("Testing with English prompt")
        success = test_client(turkish=False)
        
        if success:
            logger.info("OpenAI API connection successful with English")
        else:
            logger.error("OpenAI API connection failed with English")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing OpenAI API: {str(e)}")
        return False

def test_whisper_transcription():
    """Test Whisper transcription with a sample video."""
    logger.info("Testing Whisper transcription...")
    
    try:
        from src.knowledge_base.youtube_extractor import test_audio_transcription
        
        # Test with a sample video
        test_video_id = "YTVJw5m0NWI"  # Alternative sample video with Turkish content
        
        success = test_audio_transcription(test_video_id, language="tr")
        
        if success:
            logger.info("Whisper transcription test successful")
        else:
            logger.error("Whisper transcription test failed")
            
        return success
    
    except Exception as e:
        logger.error(f"Error testing Whisper transcription: {str(e)}")
        return False

def test_chunk_processing():
    """Test chunk processing with a sample Turkish text."""
    logger.info("Testing chunk processing...")
    
    try:
        from src.knowledge_base.chunk_processor import test_chunk_improvement
        
        # Sample Turkish text with minor grammatical errors
        sample_text = (
            "bu videoda size liderlik becerilerini nasıl geliştireceğinizi anlatcağım. "
            "iyi bir lider olmak içinn öncelikle kendinize iyi bakmalısınız. sonra takım arkadaşlarınızı dinlemelisiniz "
            "ve onların fikirrlerine saygı göstermelisiniz"
        )
        
        improved_text = test_chunk_improvement(sample_text, model="gpt-3.5-turbo")
        
        if improved_text and improved_text != sample_text:
            logger.info("Chunk processing test successful")
            logger.info(f"Original: {sample_text}")
            logger.info(f"Improved: {improved_text}")
            return True
        else:
            logger.warning("Chunk processing test results are unclear")
            logger.warning("Either no improvement was made or there was an error")
            return False
    
    except Exception as e:
        logger.error(f"Error testing chunk processing: {str(e)}")
        return False

def test_vector_store():
    """Test vector store with sample data."""
    logger.info("Testing vector store...")
    
    try:
        from src.knowledge_base.vector_store import VectorStore
        import tempfile
        
        # Create temporary directory for vector store
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary vector store at {temp_dir}")
        
        # Create vector store
        store = VectorStore(temp_dir)
        
        # Sample documents
        documents = [
            {
                "text": "Liderlik, bir ekibi ortak hedeflere yönlendirme yeteneğidir.",
                "video_id": "test1",
                "video_title": "Liderlik Temelleri",
                "url_with_timestamp": "https://youtube.com/watch?v=test1&t=0s",
                "start_time": 0,
                "end_time": 10,
                "chunk_index": 0
            },
            {
                "text": "İyi bir lider, ekip üyelerini dinler ve onların fikirlerine değer verir.",
                "video_id": "test1",
                "video_title": "Liderlik Temelleri",
                "url_with_timestamp": "https://youtube.com/watch?v=test1&t=10s",
                "start_time": 10,
                "end_time": 20,
                "chunk_index": 1
            },
            {
                "text": "Etkili iletişim, başarılı liderliğin temel taşlarından biridir.",
                "video_id": "test2",
                "video_title": "Liderlikte İletişim",
                "url_with_timestamp": "https://youtube.com/watch?v=test2&t=0s",
                "start_time": 0,
                "end_time": 10,
                "chunk_index": 0
            }
        ]
        
        # Add documents
        success = store.add_documents(documents, batch_size=3)
        
        if not success:
            logger.error("Failed to add documents to vector store")
            return False
        
        # Test search
        query = "liderlik iletişim"
        results = store.search(query, top_k=2)
        
        if results and len(results) > 0:
            logger.info(f"Vector store search returned {len(results)} results")
            logger.info(f"Top result: {results[0]['text']} (Score: {results[0]['score']:.4f})")
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary vector store: {temp_dir}")
            
            return True
        else:
            logger.error("Vector store search returned no results")
            return False
    
    except Exception as e:
        logger.error(f"Error testing vector store: {str(e)}")
        return False

def test_response_generation():
    """Test response generation with sample data."""
    logger.info("Testing response generation...")
    
    try:
        from src.ai_engine.openai_service import OpenAIService
        
        # Create service
        service = OpenAIService()
        
        # Sample query
        query = "Liderlik yeteneklerimi nasıl geliştirebilirim?"
        
        # Sample context
        context = [
            {
                "text": "Liderlik yetenekleri pratik ve geri bildirim ile gelişir. Mentor bulun ve düzenli olarak kişisel gelişim planınızı gözden geçirin.",
                "score": 0.85,
                "video_title": "Liderlik Gelişimi 101",
                "url_with_timestamp": "https://youtube.com/watch?v=test1&t=120s"
            }
        ]
        
        # Generate response
        response = service.generate_response(query, context)
        
        if response and "response" in response and response["response"]:
            logger.info("Response generation test successful")
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response['response'][:100]}...")
            return True
        else:
            logger.error("Response generation failed or returned empty response")
            return False
    
    except Exception as e:
        logger.error(f"Error testing response generation: {str(e)}")
        return False

def run_all_tests(skip_slow: bool = False):
    """
    Run all tests and report results.
    
    Args:
        skip_slow: Whether to skip slow tests like transcription
    """
    logger.info("Starting comprehensive system tests")
    
    tests = {
        "OpenAI API Connectivity": test_openai_api,
        "Chunk Processing": test_chunk_processing,
        "Vector Store": test_vector_store,
        "Response Generation": test_response_generation
    }
    
    if not skip_slow:
        tests["Whisper Transcription"] = test_whisper_transcription
    
    results = {}
    passed = 0
    failed = 0
    
    for name, test_func in tests.items():
        logger.info(f"\n========== Testing {name} ==========")
        try:
            success = test_func()
            results[name] = success
            
            if success:
                passed += 1
                logger.info(f"{name}: PASSED")
            else:
                failed += 1
                logger.error(f"{name}: FAILED")
        except Exception as e:
            results[name] = False
            failed += 1
            logger.error(f"{name}: ERROR - {str(e)}")
    
    # Print summary
    logger.info("\n========== Test Summary ==========")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
    
    return passed == len(tests)

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Test LeadershipCoach system components")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests like transcription")
    parser.add_argument("--test", type=str, choices=["api", "transcription", "chunk", "vector", "response", "all"], 
                         default="all", help="Specific test to run")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    if args.test == "api":
        test_openai_api()
    elif args.test == "transcription":
        test_whisper_transcription()
    elif args.test == "chunk":
        test_chunk_processing()
    elif args.test == "vector":
        test_vector_store()
    elif args.test == "response":
        test_response_generation()
    else:
        run_all_tests(skip_slow=args.skip_slow)
    
    # Final message
    logger.info("System test completed. Check the log file for details.")
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