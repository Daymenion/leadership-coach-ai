"""
Module for converting text to speech.
"""

import os
import tempfile
import base64
from typing import Dict, Optional, Tuple
from gtts import gTTS
from pydub import AudioSegment
import re

class TextToSpeech:
    """
    Class for converting text to speech.
    """
    
    def __init__(self, audio_dir: str = "data/audio"):
        """
        Initialize the TextToSpeech.
        
        Args:
            audio_dir: Directory to store audio files
        """
        self.audio_dir = audio_dir
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def text_to_speech(
        self, 
        text: str, 
        lang: str = "tr",
        slow: bool = False,
        save_path: Optional[str] = None
    ) -> Tuple[str, bytes]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert
            lang: Language code
            slow: Whether to speak slowly
            save_path: Path to save the audio file (optional)
            
        Returns:
            Tuple of (file_path, audio_bytes)
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(temp_path)
        
        # Read audio bytes
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        
        # Save to permanent location if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            file_path = save_path
        else:
            # Generate a unique filename in the audio directory
            filename = f"speech_{base64.urlsafe_b64encode(os.urandom(6)).decode()}.mp3"
            file_path = os.path.join(self.audio_dir, filename)
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return file_path, audio_bytes
    
    def get_audio_base64(self, audio_bytes: bytes) -> str:
        """
        Convert audio bytes to base64 for HTML embedding.
        
        Args:
            audio_bytes: Audio file bytes
            
        Returns:
            Base64-encoded audio string
        """
        return base64.b64encode(audio_bytes).decode("utf-8")
    
    def clean_text_for_tts(self, text: str) -> str:
        """
        Clean text for TTS by removing URLs, source references, and other elements
        that shouldn't be read aloud.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text suitable for TTS
        """
        # Remove URL patterns
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove citation markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove source references like "Source: Video Title"
        text = re.sub(r'Source: .*?($|\n)', '', text)
        text = re.sub(r'Kaynak: .*?($|\n)', '', text)
        
        # Remove phrases like "Click here", "View Source", etc.
        phrases_to_remove = [
            "Click here", "View Source", "View more", "Read more",
            "Kaynağı görüntüle", "Buraya tıklayın", "Daha fazla oku"
        ]
        for phrase in phrases_to_remove:
            text = text.replace(phrase, '')
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def split_text_for_tts(self, text: str, max_chars: int = 500) -> list:
        """
        Split long text into smaller chunks for TTS processing.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences to avoid cutting in the middle of a sentence
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add period back if it was removed during split
            if sentence and not sentence.endswith("."):
                sentence += "."
                
            # If adding this sentence exceeds max_chars, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def process_long_text(
        self, 
        text: str, 
        lang: str = "tr",
        slow: bool = False,
        max_chars: int = 500
    ) -> Tuple[str, bytes]:
        """
        Process long text by splitting it into chunks and combining the audio.
        First cleans the text to remove elements not suitable for TTS.
        
        Args:
            text: Text to convert
            lang: Language code
            slow: Whether to speak slowly
            max_chars: Maximum characters per chunk
            
        Returns:
            Tuple of (file_path, audio_bytes)
        """
        # Clean text for TTS by removing URLs, citations, etc.
        cleaned_text = self.clean_text_for_tts(text)
        
        # Split text into manageable chunks
        chunks = self.split_text_for_tts(cleaned_text, max_chars)
        
        if not chunks:
            return None, b""
        
        # Process each chunk
        temp_files = []
        combined = AudioSegment.empty()
        
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            tts = gTTS(text=chunk, lang=lang, slow=slow)
            tts.save(temp_path)
            
            # Add to combined audio
            segment = AudioSegment.from_mp3(temp_path)
            combined += segment
        
        # Save combined audio
        filename = f"speech_{base64.urlsafe_b64encode(os.urandom(6)).decode()}.mp3"
        file_path = os.path.join(self.audio_dir, filename)
        combined.export(file_path, format="mp3")
        
        # Read combined audio bytes
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return file_path, audio_bytes


if __name__ == "__main__":
    # Example usage
    tts = TextToSpeech()
    
    # Short text
    text = "Welcome to the Leadership Coach AI. How can I help you today?"
    file_path, audio_bytes = tts.text_to_speech(text)
    print(f"Generated audio file: {file_path}")
    
    # Long text
    long_text = """Leadership is the art of motivating a group of people to act toward achieving a common goal. 
    Effective leadership is based on ideas, but won't happen unless those ideas can be communicated to others in a way that engages them.
    Great leaders move us by tapping into our emotions. They inspire and motivate us to be our best selves."""
    
    file_path, audio_bytes = tts.process_long_text(long_text)
    print(f"Generated long audio file: {file_path}") 