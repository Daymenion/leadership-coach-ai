"""
Test script for OpenAI API connectivity using our centralized client.
This script checks if the API connection works without proxy issues.
"""

import os
import json
import logging
import argparse
from dotenv import load_dotenv
from src.utils.openai_client import get_client, test_client

# Define log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'api_connectivity_test.log')

# Configure logging
logger = logging.getLogger("api_connectivity_test")
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
logger.info(f"Starting API connectivity test, log file: {LOG_FILE}")

# Load environment variables
load_dotenv()

def test_chat_completion(prompt, model="gpt-3.5-turbo"):
    """
    Test the chat completion functionality.
    
    Args:
        prompt: Text prompt for the chat
        model: Model to use
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Testing chat completion with model: {model}")
    logger.info(f"Prompt: {prompt}\n")
    
    try:
        client = get_client()
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "Sen bir liderlik koçusun."},
                {"role": "user", "content": prompt}
            ],
            model=model
        )
        
        logger.info("\nResponse:")
        
        # Extract and display the text response
        if "choices" in response and len(response["choices"]) > 0:
            response_text = response["choices"][0]["message"]["content"]
            logger.info(response_text)
            return True
        else:
            logger.error("No valid response received")
            return False
            
    except Exception as e:
        logger.error(f"Error testing chat completion: {str(e)}")
        return False

def test_turkish_prompt():
    """Test with a Turkish prompt."""
    return test_chat_completion("Merhaba, liderlik becerilerini geliştirmek için ne önerirsin?")

def test_english_prompt():
    """Test with an English prompt."""
    return test_chat_completion("Hello, what strategies would you recommend for improving leadership skills?")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test OpenAI API connectivity")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                        help="Model to use for testing")
    parser.add_argument("--prompt", type=str, 
                        help="Custom prompt to test")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    logger.info("Starting OpenAI API connectivity test...")
    logger.info(f"Using model: {args.model}")
    
    success = True
    
    # Use test_client for basic connectivity test
    if test_client(turkish=True):
        logger.info("\nTesting with Turkish prompt:")
        if not test_turkish_prompt():
            success = False
        
        logger.info("\nTesting with English prompt:")
        if not test_english_prompt():
            success = False
            
        # Optional custom prompt
        if args.prompt:
            logger.info(f"\nTesting with custom prompt: {args.prompt}")
            if not test_chat_completion(args.prompt, args.model):
                success = False
    else:
        logger.error("Initial API connection test failed")
        success = False
    
    if success:
        logger.info("\n✅ All tests passed! The OpenAI client is working correctly.")
    else:
        logger.error("\n❌ Some tests failed. Check the logs for details.")
    
    # Final message
    logger.info("API connectivity test completed.")
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