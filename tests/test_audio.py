"""
Tests for the audio components.
"""

import os
import pytest
import tempfile
import base64
from src.audio.text_to_speech import TextToSpeech

class TestTextToSpeech:
    """Tests for the TextToSpeech class."""
    
    def test_split_text_for_tts(self):
        """Test splitting text for TTS."""
        tts = TextToSpeech()
        
        # Test with short text
        short_text = "This is a short text."
        chunks = tts.split_text_for_tts(short_text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Test with long text
        long_text = "This is a longer text. It has multiple sentences. " + \
                    "We want to make sure it gets split correctly. " + \
                    "Each chunk should be under the max_chars limit. " + \
                    "But we should try to split at sentence boundaries."
        
        chunks = tts.split_text_for_tts(long_text, max_chars=50)
        assert len(chunks) > 1
        
        # Check that each chunk is under the limit
        for chunk in chunks:
            assert len(chunk) <= 50
    
    def test_get_audio_base64(self):
        """Test converting audio bytes to base64."""
        tts = TextToSpeech()
        
        # Test with empty bytes
        empty_bytes = b""
        empty_base64 = tts.get_audio_base64(empty_bytes)
        assert empty_base64 == ""
        
        # Test with some bytes
        test_bytes = b"test audio data"
        test_base64 = tts.get_audio_base64(test_bytes)
        
        # Check that it's a valid base64 string
        assert isinstance(test_base64, str)
        
        # Check that it decodes back to the original bytes
        decoded_bytes = base64.b64decode(test_base64)
        assert decoded_bytes == test_bytes
    
    @pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI environment")
    def test_text_to_speech(self):
        """Test converting text to speech."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts = TextToSpeech(audio_dir=temp_dir)
            
            # Test with short text
            text = "Hello, world!"
            file_path, audio_bytes = tts.text_to_speech(text)
            
            # Check that file was created
            assert os.path.exists(file_path)
            
            # Check that audio bytes are not empty
            assert len(audio_bytes) > 0
            
            # Check that file contains the audio bytes
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            assert file_bytes == audio_bytes 