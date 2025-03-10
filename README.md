# Leadership Coach AI Knowledge Base

A comprehensive system for creating a knowledge base from YouTube videos, with a focus on leadership content in Turkish. The system uses audio-based transcription, grammar improvement, and vector embedding for efficient retrieval.

## Features

- **Audio-based Transcription**: Downloads and transcribes YouTube videos using Whisper
- **Grammar Correction**: Improves the quality of Turkish transcripts using OpenAI's language models
- **Vector Embedding**: Creates searchable embeddings from transcript chunks
- **Turkish Language Optimization**: All components are optimized for Turkish language processing
- **Robust Error Handling**: Comprehensive error handling and logging throughout the system
- **Detailed Logging**: All operations are logged to both console and log files for monitoring and debugging
- **User-Friendly Interface**: Clean and intuitive chat UI
- **Referenced Responses**: Provides sources for all information
- **Voice Output**: Converts text responses to speech
- **Web Search Capability**: Supplements knowledge with web searches when needed

## Requirements

- Python 3.9+
- FFmpeg (for audio processing)
- OpenAI API key (for grammar correction and embeddings)
- YouTube videos or playlist

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Daymenion/LeadershipCoach.git
   cd LeadershipCoach
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - **Windows**: Download from [FFmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Initializing the Knowledge Base

The main script `init_knowledge_base.py` initializes the knowledge base from YouTube videos. You can run it with various options:

```bash
# Process a YouTube playlist
python init_knowledge_base.py --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"

# Process specific videos
python init_knowledge_base.py --videos VIDEO_ID1 VIDEO_ID2 VIDEO_ID3

# Limit the number of videos to process
python init_knowledge_base.py --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" --max-videos 5

# Skip grammar improvement (faster but less quality)
python init_knowledge_base.py --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" --skip-grammar

# Skip transcription (use existing transcript files)
python init_knowledge_base.py --skip-transcription

# Change the logging level
python init_knowledge_base.py --videos VIDEO_ID --log-level DEBUG
```

### Running the Application

1. Start the Streamlit App:
   ```bash
   streamlit run app.py
   ```

   This will start the application and open it in your default web browser.

2. Build the Knowledge Base (when first running the application):
   - In the sidebar, click on "Advanced Settings" to expand it
   - Click the "Rebuild Knowledge Base" button
   - Wait for the process to complete (this may take several minutes)

3. Using the Application:
   - Type leadership-related questions in the chat input
   - View the AI's responses with references to the source material
   - Listen to the audio version of the response (if voice output is enabled)
   - Adjust settings in the sidebar:
     - Number of knowledge base results
     - Enable/disable web search
     - Enable/disable voice output
     - Adjust response temperature and max tokens

### Testing Individual Components

Several test scripts are available to test specific components of the system:

```bash
# Test API connectivity
python tests/test_api_connectivity.py

# Test grammar correction
python tests/test_grammar_correction.py --samples

# Test system components
python tests/test_system.py

# Test audio transcription
python tests/test_transcript_extraction.py --check-system
python tests/test_transcript_extraction.py --video VIDEO_ID

```

## YouTube Audio Transcription System

### Overview

The system downloads audio from YouTube videos and uses Whisper to transcribe the content. This approach ensures we can get transcripts from any YouTube video, even if they don't have official transcripts available.

### Implementation Details

The system follows these steps for each video:
1. Extract video information (title, author, duration) using yt-dlp
2. Download the audio in the best available quality using yt-dlp
3. Convert the audio to MP3 format using FFmpeg (via yt-dlp)
4. Transcribe the audio using Whisper's base model
5. Process the transcript into logical chunks for storage and retrieval
6. Save both raw transcripts and processed chunks

## Logging

The system provides comprehensive logging with the following features:

- **Log Files**: All logs are saved to the `logs` directory
- **Log Levels**: You can set the log level with the `--log-level` parameter (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Detailed Formatting**: Logs include timestamps, component names, and log levels
- **Console Output**: All logs are displayed in the console during execution
- **File Persistence**: Logs are saved to files for later analysis:
  - Main initialization: `logs/init_knowledge_base.log`
  - System tests: `system_test.log`

To view log files:
```bash
# View the last 50 lines of the initialization log
tail -n 50 logs/init_knowledge_base.log

# Search for error messages in the log
grep "ERROR" logs/init_knowledge_base.log
```

## System Architecture

The system consists of the following components:

1. **YouTube Extractor (`src/knowledge_base/youtube_extractor.py`)**: 
   - Downloads audio from YouTube videos
   - Transcribes audio using Whisper
   - Creates transcript chunks

2. **Chunk Processor (`src/knowledge_base/chunk_processor.py`)**: 
   - Improves grammar and formatting of transcript chunks
   - Optimized for Turkish language

3. **Vector Store (`src/knowledge_base/vector_store.py`)**: 
   - Creates embeddings from transcript chunks
   - Provides semantic search functionality

4. **OpenAI Client (`src/utils/openai_client.py`)**: 
   - Centralizes all OpenAI API calls
   - Provides robust error handling and retries

5. **AI Engine (`src/ai_engine/openai_service.py`)**: 
   - Generates responses based on user queries and retrieved context
   - Optimized for Turkish language

## Troubleshooting

### YouTube Download Issues

If you encounter issues downloading from YouTube:

1. Update yt-dlp to the latest version:
   ```bash
   pip install -U yt-dlp
   ```

2. Check if the video is available in your region
3. Try different videos or a different playlist
4. Check your internet connection
5. Verify the video or playlist exists and is publicly accessible

### Transcription Issues

If transcription fails:

1. Ensure FFmpeg is installed correctly
2. Check that the audio was downloaded properly
3. Try with a different video
4. Check if Whisper is installed correctly

### OpenAI API Issues

If OpenAI API calls fail:

1. Check your API key in the .env file
2. Verify you have sufficient API credits
3. Check your internet connection
4. Run `python tests/test_api_connectivity.py` to test connectivity

### Knowledge Base Issues

If you encounter issues with the knowledge base:

1. Check that you have internet access to download YouTube transcripts
2. Ensure you have sufficient disk space for storing the knowledge base
3. Try rebuilding the knowledge base from the sidebar

### Audio Issues

If voice output isn't working:

1. Check that you have enabled voice output in the sidebar
2. Ensure your browser allows audio playback
3. Verify that the required audio libraries are installed

## File Structure

```
/
├── app.py                     # Main Streamlit application
├── init_knowledge_base.py     # Knowledge base initialization script
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment variables
├── logs/                      # Log files directory
├── tests/                     # Test scripts
│   ├── test_system.py         # Comprehensive system test
│   ├── test_api_connectivity.py # Test OpenAI API connectivity
│   ├── test_grammar_correction.py # Test grammar correction
│   ├── test_transcript_extraction.py # Test audio transcription
│   ├── test_audio.py          # Audio processing tests
│   ├── test_ai_engine.py      # AI Engine tests
│   ├── test_knowledge_base.py # Knowledge base tests
├── src/
│   ├── ai_engine/             # AI response generation
│   ├── knowledge_base/        # Knowledge base components
│   │   ├── chunk_processor.py # Grammar improvement
│   │   ├── text_processor.py  # Text processing utilities
│   │   ├── vector_store.py    # Vector store and search
│   │   └── youtube_extractor.py # YouTube extraction and transcription
│   ├── utils/                 # Utility functions
│   │   └── openai_client.py   # Centralized OpenAI client
│   └── audio/                 # Audio processing utilities
└── data/                      # Data storage (created during initialization)
    ├── audio/                 # Downloaded audio files
    ├── transcripts/           # Raw transcripts
    ├── chunks/                # Processed transcript chunks
    ├── vector_store/          # Vector embeddings
    └── logs/                  # Conversation logs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for Whisper and GPT models
- [langchain](https://github.com/langchain-ai/langchain) for vector store integration
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloading 
- [Streamlit](https://streamlit.io/) for the web interface 