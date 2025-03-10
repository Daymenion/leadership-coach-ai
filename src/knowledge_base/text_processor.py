"""
Module for processing text and transcripts.
Optimized for Turkish language processing with specialization in leadership content.
"""

import os
import json
import re
import unicodedata
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processor for normalizing and cleaning text, especially in Turkish.
    """
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", language: str = "tr"):
        """
        Initialize the text processor.
        
        Args:
            raw_dir: Directory containing raw transcripts
            processed_dir: Directory to store processed text
            language: Target language (default: Turkish)
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.language = language
        
        # Common Turkish stopwords
        self.tr_stopwords = {
            "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey", 
            "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", 
            "en", "gibi", "hem", "hep", "hepsi", "her", "hiç", "için", "ile", "ise", 
            "kez", "ki", "kim", "mı", "mu", "mü", "nasıl", "ne", "neden", "nerde", 
            "nereye", "niçin", "niye", "o", "sanki", "şey", "siz", "şu", "tüm", "ve", 
            "veya", "ya", "yani"
        }
        
        # Leadership domain-specific terms that shouldn't be removed
        self.leadership_terms = {
            "lider", "liderlik", "yönetim", "takım", "strateji", "vizyon", "misyon",
            "motivasyon", "delegasyon", "yetki", "sorumluluk", "iletişim", "geri bildirim",
            "koç", "mentor", "gelişim", "performans", "hedef", "amaç", "başarı", "değer",
            "kültür", "değişim", "inovasyon", "çözüm", "karar", "etik", "güven"
        }
        
        logger.info(f"TextProcessor initialized with language: {language}")
    
    def load_transcripts(self, transcript_file: str = "all_transcripts.json") -> Dict[str, List[Dict]]:
        """
        Load transcripts from a JSON file.
        
        Args:
            transcript_file: Name of the transcript file
            
        Returns:
            Dictionary mapping video IDs to transcript lists
        """
        transcript_path = os.path.join(self.raw_dir, transcript_file)
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcripts = json.load(f)
            logger.info(f"Loaded transcripts for {len(transcripts)} videos from {transcript_path}")
            return transcripts
        except FileNotFoundError:
            logger.warning(f"Transcript file {transcript_path} not found")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error parsing transcript file {transcript_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading transcripts: {str(e)}")
            return {}
    
    def load_video_info(self, playlist_info_file: str = None) -> Dict[str, Any]:
        """
        Load video metadata from a JSON file.
        
        Args:
            playlist_info_file: Name of the playlist info file (if None, find first matching file)
            
        Returns:
            Dictionary containing playlist and video information
        """
        try:
            if playlist_info_file is None:
                # Find first playlist info file
                for file in os.listdir(self.raw_dir):
                    if file.startswith("playlist_") and file.endswith("_info.json"):
                        playlist_info_file = file
                        break
                
                if playlist_info_file is None:
                    logger.warning("No playlist info file found")
                    return {"title": "Unknown Playlist", "videos": []}
            
            playlist_path = os.path.join(self.raw_dir, playlist_info_file)
            with open(playlist_path, "r", encoding="utf-8") as f:
                playlist_info = json.load(f)
            
            logger.info(f"Loaded info for playlist: {playlist_info.get('title', 'Unknown')} with {len(playlist_info.get('videos', []))} videos")
            return playlist_info
        except Exception as e:
            logger.error(f"Error loading video info: {str(e)}")
            return {"title": "Unknown Playlist", "videos": []}
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace, fixing unicode issues, etc.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix Turkish character issues
        text = text.replace('í', 'ı').replace('ð', 'ğ').replace('þ', 'ş')
        
        # Remove non-breaking spaces and other invisible characters
        text = re.sub(r'[\u00A0\u2000-\u200F\u2028-\u202F\u205F-\u206F]', ' ', text)
        
        # Normalize Turkish specific patterns
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)  # Replace comma with period in numbers
        
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_turkish: bool = True) -> str:
        """
        Remove special characters except those needed for natural language.
        
        Args:
            text: Input text
            keep_turkish: Whether to keep Turkish-specific characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        if keep_turkish:
            # Keep Turkish characters (ğ, ü, ş, i, ö, ç, ı) and standard Latin letters
            pattern = r'[^a-zA-ZğüşiöçıĞÜŞİÖÇI0-9.,!?;:\'"\s\-]'
        else:
            # Keep only standard Latin letters
            pattern = r'[^a-zA-Z0-9.,!?;:\'"\s\-]'
        
        return re.sub(pattern, '', text)
    
    def clean_transcript_text(self, text: str) -> str:
        """
        Clean transcript text by removing artifacts and normalizing.
        
        Args:
            text: Transcript text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize text first
        text = self.normalize_text(text)
        
        # Remove timestamps like [00:01:23]
        text = re.sub(r'\[\d{1,2}:\d{1,2}(:\d{1,2})?\]', '', text)
        
        # Remove speaker labels like "[Speaker]:" or "Speaker:"
        text = re.sub(r'\[[^\]]+\]:', '', text)
        text = re.sub(r'^\s*[\w\s]+:', '', text, flags=re.MULTILINE)
        
        # Remove various transcript artifacts
        patterns = [
            r'\(applause\)', r'\(laughter\)', r'\(music\)',
            r'\(\s*inaudible\s*\)', r'\(\s*unintelligible\s*\)',
            r'\[[^\]]*\]', r'\([^)]*\)'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Fix common transcript issues
        text = text.replace('...', '. ')
        text = re.sub(r'\.{2,}', '. ', text)  # Replace multiple periods
        text = re.sub(r'\s+\.', '.', text)  # Fix spaces before periods
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str, keep_leadership_terms: bool = True) -> str:
        """
        Remove common stopwords from text.
        
        Args:
            text: Input text
            keep_leadership_terms: Whether to keep leadership domain-specific terms
            
        Returns:
            Text with stopwords removed
        """
        if not text:
            return ""
        
        # Split text into words
        words = text.lower().split()
        
        # Filter stopwords
        if keep_leadership_terms:
            filtered_words = [word for word in words if word not in self.tr_stopwords or word in self.leadership_terms]
        else:
            filtered_words = [word for word in words if word not in self.tr_stopwords]
        
        return ' '.join(filtered_words)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling Turkish punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Clean and normalize text first
        text = self.clean_transcript_text(text)
        
        # Handle common abbreviations in Turkish to avoid incorrect splits
        abbreviations = ['Dr.', 'Prof.', 'Av.', 'Yrd.', 'Doç.', 'vb.', 'vs.', 'bkz.', 'çev.']
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.', '<point>'))
        
        # Split on sentence terminators
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviation periods
        sentences = [s.replace('<point>', '.') for s in sentences]
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            chunk_size: Maximum size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        # Clean text first
        text = self.clean_transcript_text(text)
        
        # Get sentences
        sentences = self.split_into_sentences(text)
        
        # Create chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk_size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Add current chunk to results
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_transcript(self, transcript: List[Dict[str, Any]], chunk_size: int = 5) -> List[Dict[str, Any]]:
        """
        Process transcript segments into chunks.
        
        Args:
            transcript: List of transcript segments
            chunk_size: Number of segments per chunk
            
        Returns:
            List of processed chunks
        """
        if not transcript:
            return []
        
        processed_chunks = []
        
        for i in range(0, len(transcript), chunk_size):
            # Get segment batch
            segment_batch = transcript[i:i+chunk_size]
            
            # Extract and clean text from segments
            texts = []
            for segment in segment_batch:
                text = segment.get("text", "")
                if text:
                    cleaned_text = self.clean_transcript_text(text)
                    texts.append(cleaned_text)
            
            # Combine cleaned texts
            combined_text = " ".join(texts)
            
            # Skip empty chunks
            if not combined_text.strip():
                continue
            
            # Calculate start and end times
            start_time = segment_batch[0].get("start", 0)
            end_time = segment_batch[-1].get("start", 0) + segment_batch[-1].get("duration", 0)
            
            # Create chunk
            chunk = {
                "text": combined_text,
                "start_time": start_time,
                "end_time": end_time,
                "segment_count": len(segment_batch),
                "chunk_index": i // chunk_size
            }
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def process_all_transcripts(self, chunk_size: int = 5) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
        """
        Process all transcripts and create chunks.
        
        Args:
            chunk_size: Number of transcript entries to combine into one chunk
            
        Returns:
            Tuple of (chunks_by_video, video_info)
        """
        try:
            # Load transcripts and video info
            transcripts = self.load_transcripts()
            if not transcripts:
                logger.warning("No transcripts found to process")
                return {}, {}
                
            video_info = self.load_video_info()
            
            # Create a video ID to title mapping
            video_id_to_info = {video['video_id']: video for video in video_info.get('videos', [])}
            
            chunks_by_video = {}
            all_chunks = []
            
            for video_id, transcript in tqdm(transcripts.items(), desc="Processing transcripts"):
                # Get video metadata
                video_info_dict = video_id_to_info.get(video_id, {})
                video_title = video_info_dict.get('title', f"Unknown Video ({video_id})")
                video_url = video_info_dict.get('url', f'https://www.youtube.com/watch?v={video_id}')
                
                # Create chunks for this video
                chunks = self.process_transcript(transcript, chunk_size)
                
                # Skip if no chunks were created
                if not chunks:
                    logger.warning(f"No chunks created for video {video_id} ({video_title})")
                    continue
                
                # Add video metadata to each chunk
                for chunk in chunks:
                    chunk['video_id'] = video_id
                    chunk['video_title'] = video_title
                    chunk['video_url'] = video_url
                    
                    # Add timestamp URL
                    start_seconds = int(chunk['start_time'])
                    chunk['url_with_timestamp'] = f"{video_url}&t={start_seconds}s"
                    
                    # Add to all chunks list
                    all_chunks.append(chunk)
                
                chunks_by_video[video_id] = chunks
                logger.info(f"Processed {len(chunks)} chunks for video: {video_title}")
            
            # Skip saving if no chunks were created
            if not all_chunks:
                logger.warning("No chunks created from any transcripts")
                return {}, video_info
                
            # Save all processed chunks
            os.makedirs(self.processed_dir, exist_ok=True)
            with open(os.path.join(self.processed_dir, "all_chunks.json"), "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                
            # Save chunks by video
            with open(os.path.join(self.processed_dir, "chunks_by_video.json"), "w", encoding="utf-8") as f:
                json.dump(chunks_by_video, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processed a total of {len(all_chunks)} chunks from {len(chunks_by_video)} videos")
            return chunks_by_video, video_info
        except Exception as e:
            logger.error(f"Error processing transcripts: {str(e)}")
            return {}, {}


if __name__ == "__main__":
    # Example usage
    processor = TextProcessor()
    chunks_by_video, video_info = processor.process_all_transcripts(chunk_size=5)
    print(f"Processed chunks for {len(chunks_by_video)} videos") 