"""
Script to initialize the knowledge base for Leadership Coach AI using audio-based transcription.
Includes grammar correction for Turkish transcript chunks using the centralized OpenAI client.
Optimized for Turkish language processing and embedding.
"""

import os
import sys
import time
import traceback
import logging
import subprocess
import json
import shutil
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.knowledge_base.youtube_extractor import YouTubeExtractor, test_audio_transcription
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.chunk_processor import ChunkProcessor
from src.utils.helpers import ensure_directories_exist
from src.utils.openai_client import get_client, test_client

# Define log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'init_knowledge_base.log')

# Configure logging with explicit file handler
logger = logging.getLogger("knowledge_base_init")
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

# Log startup message to verify logging is working
logger.info(f"Initializing knowledge base, log file: {LOG_FILE}")

def install_dependencies():
    """
    Install required dependencies for audio-based transcription and Turkish text processing.
    """
    logger.info("Checking and installing dependencies...")
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        logger.info("yt-dlp is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing yt-dlp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"], check=True)
    
    # Check if FFmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("FFmpeg is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("FFmpeg is not installed. Please install FFmpeg manually as it's required for audio processing.")
        logger.info("Download FFmpeg from: https://ffmpeg.org/download.html")
    
    # Check and install other Python dependencies
    try:
        # These are the core dependencies for the knowledge base initialization
        dependencies = [
            "openai-whisper",
            "httpx",
            "python-dotenv",
            "numpy",
            "torch",
            "langchain",
            "langchain-openai",
            "chromadb",
            "tqdm",
            "pydub"
        ]
        
        logger.info("Installing Python dependencies...")
        for dep in dependencies:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        
        logger.info("All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        raise

def improve_chunk_grammar(chunks_file_path: str, output_file_path: Optional[str] = None) -> str:
    """
    Improve grammar of Turkish transcript chunks.
    
    Args:
        chunks_file_path: Path to the JSON file containing transcript chunks
        output_file_path: Path to save the improved chunks (defaults to original name with _improved suffix)
        
    Returns:
        Path to the file with improved chunks
    """
    logger.info(f"Improving grammar of transcript chunks in {chunks_file_path}")
    
    try:
        # Set default output path if not provided
        if not output_file_path:
            base, ext = os.path.splitext(chunks_file_path)
            output_file_path = f"{base}_improved{ext}"
        
        # Load chunks from file
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks for grammar improvement")
        
        # Create chunk processor and improve chunks
        processor = ChunkProcessor(model="gpt-3.5-turbo")
        improved_chunks = processor.process_chunks(chunks, batch_size=5, delay=1.0)
        
        # Save improved chunks to the output file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(improved_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(improved_chunks)} improved chunks to {output_file_path}")
        
        # Count chunks with grammar improvements
        improved_count = sum(1 for chunk in improved_chunks if chunk.get('original_text') != chunk.get('text'))
        logger.info(f"Grammar improved in {improved_count} of {len(improved_chunks)} chunks")
        
        return output_file_path
    
    except Exception as e:
        logger.error(f"Error improving chunk grammar: {str(e)}")
        logger.error(traceback.format_exc())
        return chunks_file_path  # Return the original file path in case of error

def yt_dlp_version_check():
    """Check and report yt-dlp version, and update if needed."""
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, check=True)
        current_version = result.stdout.strip()
        logger.info(f"Current yt-dlp version: {current_version}")
        
        # Try updating yt-dlp to the latest version
        logger.info("Updating yt-dlp to the latest version...")
        update_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True,
            text=True
        )
        
        if update_result.returncode == 0:
            logger.info("yt-dlp updated successfully")
            
            # Check new version
            result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, check=True)
            new_version = result.stdout.strip()
            logger.info(f"Updated yt-dlp version: {new_version}")
            
            return True
        else:
            logger.warning("Failed to update yt-dlp. Using current version.")
            return False
    
    except Exception as e:
        logger.error(f"Error checking/updating yt-dlp: {str(e)}")
        return False

def main():
    """
    Main function to initialize the knowledge base.
    
    1. Installs dependencies
    2. Creates necessary directories
    3. Tests API connectivity
    4. Tests audio-based transcription
    5. Processes YouTube playlist or videos
    6. Improves grammar of Turkish transcripts
    7. Creates a vector store from the processed data
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Initialize Knowledge Base for Leadership Coach AI')
    parser.add_argument('--playlist', type=str, help='YouTube playlist URL (overrides env variable)')
    parser.add_argument('--videos', nargs='+', help='List of YouTube video IDs to process')
    parser.add_argument('--skip-transcription', action='store_true', help='Skip audio transcription steps')
    parser.add_argument('--skip-grammar', action='store_true', help='Skip grammar improvement')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Install dependencies
        install_dependencies()
        
        # Update yt-dlp to the latest version
        yt_dlp_version_check()
        
        # Ensure required directories exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        directories = [
            os.path.join(data_dir, 'audio'),
            os.path.join(data_dir, 'transcripts'),
            os.path.join(data_dir, 'chunks'),
            os.path.join(data_dir, 'vector_store')
        ]
        ensure_directories_exist(directories)
        
        # Check OpenAI API connectivity
        logger.info("Testing connection to OpenAI API...")
        if test_client(turkish=True):
            logger.info("Connection to OpenAI API successful")
        else:
            logger.error("Connection to OpenAI API failed. Check your API key and connectivity.")
            return
        
        # Determine source of videos (playlist URL or individual video IDs)
        playlist_url = args.playlist or os.getenv('YOUTUBE_PLAYLIST_URL')
        video_ids = args.videos
        
        if not args.skip_transcription:
            # Test audio-based transcription
            test_video_id = "YTVJw5m0NWI"  # Use a fallback sample video
            if video_ids and len(video_ids) > 0:
                test_video_id = video_ids[0]
            elif playlist_url:
                # Try to extract first video from playlist for testing
                try:
                    logger.info(f"Extracting first video from playlist for testing...")
                    extractor = YouTubeExtractor(language='tr')
                    playlist_videos = extractor.extract_playlist_videos(playlist_url)
                    if playlist_videos:
                        test_video_id = playlist_videos[0]
                        logger.info(f"Using video {test_video_id} from playlist for testing")
                except Exception as e:
                    logger.warning(f"Failed to extract videos from playlist: {str(e)}")
                    logger.warning(f"Using fallback video {test_video_id} for testing")
            
            logger.info(f"Testing audio-based transcription with video {test_video_id}...")
            test_result = test_audio_transcription(test_video_id)
            
            if not test_result:
                logger.error("Audio-based transcription test failed.")
                logger.error("This could be due to YouTube restrictions or network issues.")
                logger.error("Please try these troubleshooting steps:")
                logger.error("1. Check your internet connection")
                logger.error("2. Make sure the video is publicly accessible")
                logger.error("3. Try using a different video")
                logger.error("4. Update yt-dlp with: pip install -U yt-dlp")
                
                # Ask user if they want to continue despite the error
                user_input = input("Continue despite transcription test failure? (y/n): ")
                if user_input.lower() != 'y':
                    logger.error("Aborted knowledge base initialization due to transcription test failure")
                    return
                logger.warning("Continuing despite transcription test failure")
            else:
                logger.info("Audio-based transcription test passed successfully")
        
        # Define output directory and chunks file path before conditional block
        output_dir = os.path.join(data_dir, 'chunks')
        chunks_file = os.path.join(output_dir, 'transcript_chunks.json')
        
        # Process videos to create knowledge base
        if not args.skip_transcription:
            # Initialize YouTube extractor
            extractor = YouTubeExtractor(language='tr')
            
            # Process the playlist or individual videos (download audio, transcribe, and chunk)
            os.makedirs(output_dir, exist_ok=True)
            
            if playlist_url:
                logger.info(f"Processing YouTube playlist: {playlist_url}")
                # Process playlist to extract transcripts and create chunks
                extractor.process_playlist(
                    playlist_url=playlist_url,
                    save_file=chunks_file,
                    output_dir=output_dir,
                    max_videos=args.max_videos
                )
            elif video_ids:
                logger.info(f"Processing {len(video_ids)} individual YouTube videos")
                all_chunks = []
                
                for i, video_id in enumerate(video_ids, 1):
                    logger.info(f"Processing video {i}/{len(video_ids)}: {video_id}")
                    
                    # Process single video
                    video_info, transcript, chunks = extractor.process_video(video_id, output_dir=output_dir)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"Added {len(chunks)} chunks from video {video_id}")
                    else:
                        logger.warning(f"No chunks created for video {video_id}")
                
                # Save all chunks to file
                if all_chunks:
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(all_chunks)} total chunks to {chunks_file}")
                else:
                    logger.error("No chunks were created from any videos")
                    return
            else:
                logger.error("No playlist URL or video IDs provided. Please specify one or the other.")
                logger.error("Use --playlist URL or --videos ID1 ID2 ID3 ... or set YOUTUBE_PLAYLIST_URL environment variable")
                return
        
        # Improve grammar of the Turkish transcripts
        if not args.skip_grammar and os.path.exists(chunks_file):
            logger.info("Starting grammar improvement of Turkish transcripts...")
            improved_chunks_file = improve_chunk_grammar(chunks_file)
        else:
            improved_chunks_file = chunks_file
            if args.skip_grammar:
                logger.info("Skipping grammar improvement as requested")
            
        # Create vector store from improved chunks
        logger.info("Creating vector store from processed data...")
        vector_store_dir = os.path.join(data_dir, 'vector_store')
        
        # Remove existing vector store if it exists
        if os.path.exists(vector_store_dir):
            logger.info(f"Removing existing vector store at {vector_store_dir}")
            shutil.rmtree(vector_store_dir)
        
        # Create vector store from improved chunks
        vector_store = VectorStore(vector_store_dir)
        
        # Check if improved chunks file exists
        if not os.path.exists(improved_chunks_file):
            logger.error(f"Chunks file not found: {improved_chunks_file}")
            logger.error("Cannot create vector store without transcript chunks.")
            logger.error("Please run without --skip-transcription first to generate transcript chunks.")
            return
        
        # Load improved chunks and create embeddings
        with open(improved_chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Add documents to vector store
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        vector_store.add_documents(chunks)
        
        logger.info(f"Successfully created vector store with {len(chunks)} embeddings")
        logger.info(f"Knowledge base initialization completed successfully")
        
        # Test search with a sample query
        test_query = "Liderlik becerileri nasıl geliştirilir?"
        logger.info(f"Testing search with query: '{test_query}'")
        
        results = vector_store.search(test_query, top_k=2)
        if results:
            logger.info(f"Search test successful, found {len(results)} results")
            logger.info(f"Top result: {results[0]['text'][:100]}...")
        else:
            logger.warning("Search test returned no results")
    
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {str(e)}")
        logger.error(traceback.format_exc())
        
    # Final message to confirm logging
    logger.info("Knowledge base initialization process completed. Check the log file for details.")
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