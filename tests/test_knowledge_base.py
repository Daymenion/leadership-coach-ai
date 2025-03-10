"""
Tests for the knowledge base components.
"""

import os
import pytest
import tempfile
import json
from src.knowledge_base.youtube_extractor import YouTubeExtractor
from src.knowledge_base.text_processor import TextProcessor
from src.knowledge_base.vector_store import VectorStore

class TestYouTubeExtractor:
    """Tests for the YouTubeExtractor class."""
    
    def test_extract_transcript(self):
        """Test extracting a transcript from a YouTube video."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = YouTubeExtractor(data_dir=temp_dir)
            
            # Use a known video ID from the playlist
            video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
            
            # Extract transcript without saving
            transcript = extractor.extract_transcript(video_id, save=False)
            
            # Check that transcript is a list of dictionaries
            assert transcript is not None
            assert isinstance(transcript, list)
            if transcript:  # Some videos might not have transcripts
                assert isinstance(transcript[0], dict)
                assert "text" in transcript[0]
                assert "start" in transcript[0]
                assert "duration" in transcript[0]

class TestTextProcessor:
    """Tests for the TextProcessor class."""
    
    def test_clean_text(self):
        """Test cleaning text."""
        processor = TextProcessor()
        
        # Test removing extra whitespace
        assert processor.clean_text("  Hello  world  ") == "Hello world"
        
        # Test removing timestamps
        assert processor.clean_text("Hello [00:01] world") == "Hello world"
        
        # Test removing non-ASCII characters
        assert processor.clean_text("Hello 世界") == "Hello "
    
    def test_create_chunks(self):
        """Test creating chunks from a transcript."""
        processor = TextProcessor()
        
        # Create a mock transcript
        transcript = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0},
            {"text": "this", "start": 2.0, "duration": 1.0},
            {"text": "is", "start": 3.0, "duration": 1.0},
            {"text": "a", "start": 4.0, "duration": 1.0},
            {"text": "test", "start": 5.0, "duration": 1.0},
        ]
        
        # Create chunks with chunk_size=2
        chunks = processor.create_chunks(transcript, chunk_size=2)
        
        # Check that chunks are created correctly
        assert len(chunks) == 3
        assert chunks[0]["text"] == "Hello world"
        assert chunks[0]["start_time"] == 0.0
        assert chunks[0]["end_time"] == 2.0
        assert chunks[1]["text"] == "this is"
        assert chunks[2]["text"] == "a test"

class TestVectorStore:
    """Tests for the VectorStore class."""
    
    def test_search(self):
        """Test searching for chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock chunks file
            chunks = [
                {"text": "Leadership is about vision and inspiration", "chunk_index": 0, "video_title": "Test Video 1"},
                {"text": "Management is about execution and organization", "chunk_index": 1, "video_title": "Test Video 2"},
                {"text": "Communication is key to effective leadership", "chunk_index": 2, "video_title": "Test Video 3"},
            ]
            
            # Save chunks to file
            os.makedirs(temp_dir, exist_ok=True)
            with open(os.path.join(temp_dir, "all_chunks.json"), "w") as f:
                json.dump(chunks, f)
            
            # Create vector store
            vector_store = VectorStore(processed_dir=temp_dir)
            
            # Create embeddings and index
            embeddings = vector_store.create_embeddings(chunks)
            index = vector_store.build_index(embeddings)
            
            # Search for chunks
            results = vector_store.search(
                query="leadership vision",
                k=2,
                embeddings=embeddings,
                index=index,
                chunks=chunks
            )
            
            # Check that results are returned
            assert len(results) > 0
            assert isinstance(results[0], dict)
            assert "text" in results[0]
            assert "score" in results[0]
            assert "search_rank" in results[0]
            
            # Check that the most relevant result is returned first
            assert "leadership" in results[0]["text"].lower() 