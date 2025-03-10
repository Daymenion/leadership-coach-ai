"""
Module for extracting and processing transcripts from YouTube videos.
Uses audio-based transcription with Whisper, optimized for Turkish language.
"""

import os
import json
import logging
import subprocess
import time
import whisper
import tempfile
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class YouTubeExtractor:
    """
    Class for extracting and processing transcripts from YouTube videos using audio-based transcription.
    Optimized for Turkish language processing.
    """
    
    def __init__(self, language: str = "tr", whisper_model: str = "base"):
        """
        Initialize the YouTube Extractor.
        
        Args:
            language: Target language for transcription (default: Turkish)
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.language = language
        self.whisper_model_name = whisper_model
        self.whisper_model = None  # Lazy load
        
        # Check for yt-dlp and ffmpeg
        self._check_dependencies()
        
        # Create directories
        self.audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        
        logger.info(f"YouTubeExtractor initialized with language: {language}, model: {whisper_model}")
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            # Check yt-dlp
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
            
            # Check ffmpeg
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            
            logger.info("Dependencies (yt-dlp, ffmpeg) are available")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Dependency check failed: {str(e)}")
            logger.warning("Make sure yt-dlp and ffmpeg are installed and available in PATH")
    
    def _load_whisper_model(self):
        """Load the Whisper model on demand."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper {self.whisper_model_name} model...")
            try:
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                logger.info(f"Whisper {self.whisper_model_name} model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
                raise
    
    def extract_video_id(self, url: str) -> str:
        """
        Extract YouTube video ID from URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            YouTube video ID
        """
        if not url:
            raise ValueError("URL cannot be empty")
        
        # Handle different URL formats
        if "youtu.be" in url:
            # Short URL format: https://youtu.be/VIDEO_ID
            path = urlparse(url).path
            video_id = path.strip("/")
        elif "youtube.com" in url:
            # Standard URL format: https://www.youtube.com/watch?v=VIDEO_ID
            query = urlparse(url).query
            params = parse_qs(query)
            video_id = params.get("v", [""])[0]
        else:
            # Assume the URL is already a video ID
            video_id = url
        
        # Validate video ID format (typically 11 characters)
        if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            logger.warning(f"Invalid YouTube video ID format: {video_id}")
        
        return video_id
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get information about a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video information
        """
        logger.info(f"Getting information for video {video_id}")
        
        try:
            # Use yt-dlp to get video information as JSON
            cmd = [
                "yt-dlp",
                "-J",  # Output as JSON
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_info = {
                "video_id": video_id,
                "title": info.get("title", ""),
                "channel": info.get("channel", ""),
                "duration": info.get("duration", 0),
                "upload_date": info.get("upload_date", ""),
                "view_count": info.get("view_count", 0),
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
            
            logger.info(f"Successfully retrieved info for video: {video_info['title']}")
            return video_info
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting video info for {video_id}: {e.stderr}")
            return {"video_id": video_id, "error": str(e.stderr)}
        
        except Exception as e:
            logger.error(f"Unexpected error getting video info for {video_id}: {str(e)}")
            return {"video_id": video_id, "error": str(e)}
    
    def download_audio(self, video_id: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Download audio from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            output_dir: Directory to save audio file (default: self.audio_dir)
            
        Returns:
            Path to downloaded audio file or None if failed
        """
        output_dir = output_dir or self.audio_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        output_path = os.path.join(output_dir, f"{video_id}.mp3")
        
        logger.info(f"Downloading audio for video {video_id}")
        
        try:
            # Use yt-dlp to download audio
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "mp3",  # Convert to mp3
                "--audio-quality", "0",  # Best quality
                "-o", output_path,  # Output path
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(output_path):
                logger.info(f"Successfully downloaded audio to {output_path}")
                return output_path
            else:
                logger.error(f"Audio file not found at {output_path} after download")
                return None
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading audio for {video_id}: {e.stderr}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error downloading audio for {video_id}: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcript segments with text, start, and duration
        """
        logger.info(f"Transcribing audio from {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return []
        
        try:
            # Load Whisper model if not already loaded
            self._load_whisper_model()
            
            # Transcribe audio
            logger.info(f"Starting transcription with Whisper ({self.whisper_model_name} model)")
            result = self.whisper_model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                verbose=False
            )
            
            segments = result.get("segments", [])
            
            # Format segments
            formatted_segments = []
            for seg in segments:
                formatted_segments.append({
                    "text": seg.get("text", "").strip(),
                    "start": seg.get("start", 0),
                    "duration": seg.get("end", 0) - seg.get("start", 0)
                })
            
            logger.info(f"Transcription completed with {len(formatted_segments)} segments")
            
            # Log sample of transcript
            if formatted_segments:
                sample_text = formatted_segments[0]["text"]
                if len(sample_text) > 50:
                    sample_text = sample_text[:50] + "..."
                logger.info(f"First segment: {sample_text}")
            
            return formatted_segments
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return []
    
    def extract_transcript(self, video_id: str, save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract transcript from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            save_dir: Directory to save transcript and audio
            
        Returns:
            List of transcript segments
        """
        logger.info(f"Extracting transcript for video {video_id}")
        
        # Download audio
        audio_path = self.download_audio(video_id, output_dir=save_dir)
        
        if not audio_path:
            logger.error(f"Failed to download audio for video {video_id}")
            return []
        
        # Transcribe audio
        transcript = self.transcribe_audio(audio_path)
        
        # Save transcript
        if save_dir and transcript:
            os.makedirs(save_dir, exist_ok=True)
            transcript_path = os.path.join(save_dir, f"{video_id}_transcript.json")
            
            try:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved transcript to {transcript_path}")
            except Exception as e:
                logger.error(f"Error saving transcript: {str(e)}")
        
        return transcript
    
    def extract_playlist_videos(self, playlist_url: str) -> List[str]:
        """
        Extract video IDs from a YouTube playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            
        Returns:
            List of video IDs
        """
        logger.info(f"Extracting videos from playlist {playlist_url}")
        
        try:
            # Use yt-dlp to get playlist video IDs
            cmd = [
                "yt-dlp",
                "--flat-playlist",  # Don't download videos
                "--print", "id",  # Only print video IDs
                playlist_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_ids = result.stdout.strip().split('\n')
            
            # Filter out empty IDs
            video_ids = [vid for vid in video_ids if vid]
            
            logger.info(f"Found {len(video_ids)} videos in playlist")
            return video_ids
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting playlist videos: {e.stderr}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error extracting playlist videos: {str(e)}")
            return []
    
    def create_chunks_from_transcript(self, transcript: List[Dict], video_info: Dict[str, Any], chunk_size: int = 5) -> List[Dict[str, Any]]:
        """
        Create chunks from a transcript for better embedding.
        
        Args:
            transcript: List of transcript segments
            video_info: Dictionary with video information
            chunk_size: Number of segments per chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not transcript:
            logger.warning(f"Empty transcript for video {video_info.get('video_id', 'unknown')}")
            return []
        
        logger.info(f"Creating chunks from transcript with {len(transcript)} segments")
        
        chunks = []
        for i in range(0, len(transcript), chunk_size):
            # Get segment batch
            segment_batch = transcript[i:i+chunk_size]
            
            # Skip empty segments
            if not segment_batch:
                continue
            
            # Combine text from segments
            combined_text = " ".join([segment["text"] for segment in segment_batch if segment.get("text")])
            
            # Skip empty chunks
            if not combined_text.strip():
                continue
            
            # Calculate start and end times
            start_time = segment_batch[0]["start"]
            end_time = segment_batch[-1]["start"] + segment_batch[-1]["duration"]
            
            # Create URL with timestamp
            url_with_timestamp = f"https://www.youtube.com/watch?v={video_info['video_id']}&t={int(start_time)}s"
            
            # Create chunk
            chunk = {
                "video_id": video_info["video_id"],
                "video_title": video_info.get("title", ""),
                "video_url": f"https://www.youtube.com/watch?v={video_info['video_id']}",
                "url_with_timestamp": url_with_timestamp,
                "start_time": start_time,
                "end_time": end_time,
                "text": combined_text,
                "chunk_index": i // chunk_size
            }
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from transcript")
        return chunks
    
    def process_video(self, video_id: str, output_dir: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a YouTube video: extract information, download audio, transcribe, and create chunks.
        
        Args:
            video_id: YouTube video ID
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (video_info, transcript, chunks)
        """
        logger.info(f"Processing video {video_id}")
        
        try:
            # Get video information
            video_info = self.get_video_info(video_id)
            
            if "error" in video_info:
                logger.error(f"Error getting video info: {video_info['error']}")
                return video_info, [], []
            
            # Extract transcript
            transcript = self.extract_transcript(video_id, save_dir=output_dir)
            
            if not transcript:
                logger.warning(f"No transcript available for video {video_id}")
                return video_info, [], []
            
            # Create chunks
            chunks = self.create_chunks_from_transcript(transcript, video_info)
            
            return video_info, transcript, chunks
        
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            return {"video_id": video_id, "error": str(e)}, [], []
    
    def process_playlist(self, playlist_url: str, save_file: str, output_dir: Optional[str] = None, max_videos: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Process a YouTube playlist: extract videos, get information, download audio, transcribe, and create chunks.
        
        Args:
            playlist_url: YouTube playlist URL
            save_file: Path to save chunks file
            output_dir: Directory to save output files
            max_videos: Maximum number of videos to process
            
        Returns:
            Tuple of (video_data, transcripts)
        """
        logger.info(f"Processing playlist {playlist_url}")
        
        # Extract video IDs from playlist
        video_ids = self.extract_playlist_videos(playlist_url)
        
        if not video_ids:
            logger.error("No videos found in playlist")
            return [], {}
        
        # Limit number of videos if specified
        if max_videos and max_videos > 0:
            video_ids = video_ids[:max_videos]
            logger.info(f"Processing first {max_videos} videos from playlist")
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process each video
        all_video_data = []
        all_transcripts = {}
        all_chunks = []
        
        for i, video_id in enumerate(video_ids, 1):
            logger.info(f"Processing video {i}/{len(video_ids)}: {video_id}")
            
            try:
                # Process video
                video_info, transcript, chunks = self.process_video(video_id, output_dir)
                
                # Store results
                if "error" not in video_info:
                    all_video_data.append(video_info)
                
                if transcript:
                    all_transcripts[video_id] = transcript
                
                if chunks:
                    all_chunks.extend(chunks)
                
                # Sleep to avoid rate limiting
                if i < len(video_ids):
                    time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {str(e)}")
                continue
        
        # Save all chunks to file
        if save_file and all_chunks:
            try:
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(all_chunks)} chunks to {save_file}")
            except Exception as e:
                logger.error(f"Error saving chunks to {save_file}: {str(e)}")
        
        logger.info(f"Processed {len(all_video_data)} videos with {len(all_chunks)} total chunks")
        return all_video_data, all_transcripts


def test_audio_transcription(video_id: str, language: str = "tr") -> bool:
    """
    Test audio-based transcription on a single video.
    
    Args:
        video_id: YouTube video ID
        language: Target language for transcription
        
    Returns:
        Success status
    """
    logger.info(f"Testing audio-based transcription for video {video_id}")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Initialize extractor
        extractor = YouTubeExtractor(language=language, whisper_model="tiny")
        
        # Download audio
        audio_path = extractor.download_audio(video_id, output_dir=temp_dir)
        
        if not audio_path:
            logger.error("Failed to download audio")
            return False
        
        logger.info(f"Successfully downloaded audio to {audio_path}")
        
        # Transcribe first 30 seconds only for quick test
        try:
            # Create a 30-second clip for faster testing
            short_audio_path = os.path.join(temp_dir, f"{video_id}_short.mp3")
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-t", "30",  # 30 seconds
                "-c:a", "copy",
                short_audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            test_audio = short_audio_path if os.path.exists(short_audio_path) else audio_path
        except Exception as e:
            logger.warning(f"Error creating short audio clip: {str(e)}")
            test_audio = audio_path
        
        # Transcribe
        transcript = extractor.transcribe_audio(test_audio)
        
        if not transcript:
            logger.error("Transcription failed")
            return False
        
        logger.info(f"Transcription successful with {len(transcript)} segments")
        
        # Log first segment
        if transcript:
            logger.info(f"First segment: {transcript[0]['text']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing audio transcription: {str(e)}")
        return False
    
    finally:
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error removing temporary directory: {str(e)}")


if __name__ == "__main__":
    # Test with a single video
    test_video_id = "JQhkbN75Ohs"  # Replace with your test video ID
    
    success = test_audio_transcription(test_video_id, language="tr")
    print(f"Test {'successful' if success else 'failed'}") 