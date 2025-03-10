"""
Leadership Coach AI - Main Application
A leadership coaching application that uses a knowledge base built from YouTube videos
along with web search capabilities to provide insights and guidance on leadership topics.
Optimized for Turkish language processing with audio-based transcription.
"""

import os
import sys
import streamlit as st
import base64
import json
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional

# Import custom modules
from src.knowledge_base.vector_store import VectorStore
from src.ai_engine.query_processor import QueryProcessor
from src.ai_engine.openai_service import OpenAIService
from src.ai_engine.web_search import WebSearch
from src.audio.text_to_speech import TextToSpeech
from src.utils.helpers import format_sources_for_display, log_conversation, ensure_directories_exist

# Configure logging with both file and console handlers
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Create logger
logger = logging.getLogger("leadership_coach_app")
logger.setLevel(logging.INFO)

# Reset handlers to avoid duplicate logging
if logger.handlers:
    logger.handlers = []

# Create file handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Define data directories with the new structure
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
CHUNKS_DIR = os.path.join(DATA_DIR, 'chunks')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')

# Ensure required directories exist
ensure_directories_exist([
    DATA_DIR,
    VECTOR_STORE_DIR, 
    CHUNKS_DIR,
    LOGS_DIR,
    AUDIO_DIR
])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "query_processor" not in st.session_state:
    st.session_state.query_processor = None

if "tts" not in st.session_state:
    try:
        st.session_state.tts = TextToSpeech(audio_dir=AUDIO_DIR)
    except Exception as e:
        logger.error(f"Error initializing TextToSpeech: {str(e)}")
        st.session_state.tts = None

if "knowledge_base_stats" not in st.session_state:
    st.session_state.knowledge_base_stats = None

if "openai_service" not in st.session_state:
    st.session_state.openai_service = None

if "web_search_available" not in st.session_state:
    st.session_state.web_search_available = False

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = VectorStore(storage_dir=VECTOR_STORE_DIR)
        logger.info(f"Vector store initialized with {len(st.session_state.vector_store.metadata)} vectors")
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())
        st.session_state.vector_store = None

# Page configuration
st.set_page_config(
    page_title="Leadership Coach AI",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .sources {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .sources h4 {
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sources ul {
        margin-bottom: 0;
    }
    .sources li {
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E0F2FE;
        border-left: 5px solid #0EA5E9;
    }
    .assistant-message {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
    }
    .message-content {
        margin-bottom: 0.5rem;
    }
    .audio-player {
        margin-top: 0.5rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .knowledge-base-status {
        background-color: #FEF3C7;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-ready {
        background-color: #D1FAE5;
    }
    .status-error {
        background-color: #FEE2E2;
    }
    .kb-stats {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .kb-stat-item {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        border-bottom: 1px dashed #E5E7EB;
    }
    .welcome-message {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #10B981;
    }
    .error-message {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #EF4444;
    }
    .info-message {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #0EA5E9;
    }
    .warning-message {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

def get_knowledge_base_stats() -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with knowledge base statistics
    """
    stats = {
        "status": "not_found",
        "videos_count": 0,
        "chunks_count": 0,
        "vectors_count": 0,
        "last_updated": "Never"
    }
    
    try:
        # Check if improved chunks file exists (new structure)
        chunks_file_path = os.path.join(CHUNKS_DIR, "transcript_chunks_improved.json")
        if not os.path.exists(chunks_file_path):
            # Try the regular chunks file
            chunks_file_path = os.path.join(CHUNKS_DIR, "transcript_chunks.json")
            if not os.path.exists(chunks_file_path):
                logger.warning("No chunks file found in the new data structure")
                return stats
        
        # Get stats from chunks file
        with open(chunks_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            stats["chunks_count"] = len(chunks)
            
            # Count unique videos
            video_ids = set()
            for chunk in chunks:
                if "video_id" in chunk:
                    video_ids.add(chunk["video_id"])
            stats["videos_count"] = len(video_ids)
        
        # Check if vector store exists (new structure)
        vectors_file = os.path.join(VECTOR_STORE_DIR, "vectors.npy")
        metadata_file = os.path.join(VECTOR_STORE_DIR, "metadata.json")
        
        if os.path.exists(vectors_file) and os.path.exists(metadata_file):
            # Get last modified time
            timestamp = max(os.path.getmtime(vectors_file), os.path.getmtime(metadata_file))
            stats["last_updated"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            # Get vector count
            try:
                # Load metadata to get count without loading full vectors
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                stats["vectors_count"] = len(metadata)
                stats["status"] = "ready" if len(metadata) > 0 else "empty"
            except Exception as e:
                logger.error(f"Error loading vector metadata: {str(e)}")
                stats["status"] = "error"
        
        return stats
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {str(e)}")
        logger.error(traceback.format_exc())
        stats["status"] = "error"
        return stats

def extract_sources_from_response(response_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract and format sources from a response data object.
    
    Args:
        response_data: The response data dictionary
        
    Returns:
        List of formatted source dictionaries for display
    """
    sources = []
    
    # Handle None or invalid response_data
    if not response_data or not isinstance(response_data, dict):
        return sources
    
    try:
        # Handle existing "sources" key for backward compatibility
        if "sources" in response_data and isinstance(response_data.get("sources"), list):
            return response_data.get("sources", [])
        
        # Add context sources
        context_items = response_data.get("context_used", []) or []
        if isinstance(context_items, list):
            for ctx in context_items:
                if not ctx or not isinstance(ctx, dict):
                    continue
                    
                sources.append({
                    "type": "video",
                    "title": ctx.get("video_title", "Untitled Video"),
                    "url": ctx.get("url", "#"),
                    "text_snippet": ctx.get("text", "")[:150] + "..." if ctx.get("text") else ""
                })
        
        # Add web results
        web_items = response_data.get("web_results_used", []) or []
        if isinstance(web_items, list):
            for web in web_items:
                if not web or not isinstance(web, dict):
                    continue
                    
                sources.append({
                    "type": "web",
                    "title": web.get("title", "Web Result"),
                    "url": web.get("url", "#"),
                    "text_snippet": web.get("content", "")[:150] + "..." if web.get("content") else ""
                })
    except Exception as e:
        logger.error(f"Error extracting sources from response: {str(e)}")
        logger.error(traceback.format_exc())
    
    return sources

def initialize_openai_service():
    """Initialize OpenAI service with appropriate model."""
    try:
        model_name = "gpt-4o-mini"  # Default to the most efficient model
        
        # Try to initialize with model
        service = OpenAIService(model_name=model_name)
        logger.info(f"OpenAI service initialized with model: {model_name}")
        return service
    except Exception as e:
        logger.error(f"Error initializing OpenAI service: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def initialize_query_processor() -> Optional[QueryProcessor]:
    """
    Initialize the query processor with vector store and other components.
    
    Returns:
        QueryProcessor instance or None if initialization fails
    """
    try:
        # Check if vector store is already in session state
        vector_store = st.session_state.vector_store
        if vector_store is None:
            try:
                # Try to initialize vector store
                vector_store = VectorStore(storage_dir=VECTOR_STORE_DIR)
                st.session_state.vector_store = vector_store
                logger.info(f"Vector store initialized with {len(vector_store.metadata or [])} vectors")
            except Exception as e:
                logger.error(f"Error initializing vector store: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        
        # Initialize OpenAI service if not already in session state
        if st.session_state.openai_service is None:
            st.session_state.openai_service = initialize_openai_service()
        
        if st.session_state.openai_service is None:
            logger.error("Failed to initialize OpenAI service")
            return None
        
        # Create web search instance with error handling
        web_search = None
        try:
            web_search = WebSearch(max_results=3)
            st.session_state.web_search_available = web_search.search_available
            logger.info(f"Web search initialized. Available: {web_search.search_available}")
        except Exception as e:
            logger.error(f"Error initializing web search: {str(e)}")
            logger.error(traceback.format_exc())
            st.session_state.web_search_available = False
        
        # Initialize query processor with explicit parameters
        query_processor = QueryProcessor(
            vector_store=vector_store,
            openai_service=st.session_state.openai_service,
            web_search=web_search,
            processed_dir=CHUNKS_DIR
        )
        
        logger.info("Query processor initialized successfully")
        return query_processor
    except Exception as e:
        logger.error(f"Error initializing query processor: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_init_knowledge_base():
    """
    Run the knowledge base initialization process using a subprocess call.
    This allows initializing the knowledge base from the UI without restarting the app.
    """
    try:
        import subprocess
        
        # Run the initialization script
        cmd = [sys.executable, "init_knowledge_base.py", "--skip-transcription"]
        
        st.info("Starting knowledge base initialization. This may take several minutes...")
        
        # Run the process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Show progress
        placeholder = st.empty()
        while process.poll() is None:
            placeholder.info("Knowledge base initialization in progress... Please wait.")
            time.sleep(2)
        
        # Process completed
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            placeholder.success("Knowledge base initialized successfully!")
            # Update stats and reset components
            st.session_state.knowledge_base_stats = get_knowledge_base_stats()
            st.session_state.vector_store = None  # Force reinitialization
            st.session_state.query_processor = initialize_query_processor()
            return True
        else:
            placeholder.error(f"Error initializing knowledge base: {stderr}")
            logger.error(f"Knowledge base initialization failed: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running knowledge base initialization: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error initializing knowledge base: {str(e)}")
        return False

def handle_runtime_error(error: Exception, message: str = "An error occurred") -> None:
    """
    Handle runtime errors gracefully with proper UI feedback and logging.
    
    Args:
        error: The exception that was raised
        message: A user-friendly error message prefix
    """
    error_str = str(error)
    logger.error(f"{message}: {error_str}")
    logger.error(traceback.format_exc())
    
    # Add a markdown error message for the user
    st.markdown(
        f"""
        <div class="error-message">
            <h4>❌ {message}</h4>
            <p>{error_str}</p>
            <p>Please try again or check the logs for more details.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">Leadership Coach AI</div>', unsafe_allow_html=True)
    
    # Knowledge base status
    if st.session_state.knowledge_base_stats is None:
        st.session_state.knowledge_base_stats = get_knowledge_base_stats()
    
    kb_stats = st.session_state.knowledge_base_stats
    kb_status_class = ""
    
    if kb_stats["status"] == "ready":
        kb_status_class = "status-ready"
        kb_status_text = "✅ Ready"
    elif kb_stats["status"] == "empty":
        kb_status_class = "status-error"
        kb_status_text = "⚠️ Empty"
    elif kb_stats["status"] == "error":
        kb_status_class = "status-error"
        kb_status_text = "❌ Error"
    else:
        kb_status_text = "⚠️ Not Found"
    
    st.markdown(f"""
    <div class="knowledge-base-status {kb_status_class}">
        <strong>Knowledge Base Status:</strong> {kb_status_text}
        <div class="kb-stats">
            <div class="kb-stat-item">
                <span>Videos:</span> <span>{kb_stats["videos_count"]}</span>
            </div>
            <div class="kb-stat-item">
                <span>Text Chunks:</span> <span>{kb_stats["chunks_count"]}</span>
            </div>
            <div class="kb-stat-item">
                <span>Vectors:</span> <span>{kb_stats["vectors_count"]}</span>
            </div>
            <div class="kb-stat-item">
                <span>Last Updated:</span> <span>{kb_stats["last_updated"]}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### About")
    st.markdown(
        "This AI coach provides specialized guidance on leadership practices, "
        "professional development, and business acumen using content from leadership interviews."
    )
    
    st.markdown("### Features")
    st.markdown(
        "- 🧠 Specialized knowledge base\n"
        "- 🌐 Web search capability\n"
        "- 📚 Referenced responses\n"
        "- 🔊 Voice output\n"
        "- 🌍 Multi-language transcript support\n"
    )
    
    st.markdown("### Settings")
    
    # Knowledge base settings
    st.markdown("#### Knowledge Base")
    kb_results_count = st.slider("Number of knowledge base results", 1, 10, 5)
    
    # Web search settings
    st.markdown("#### Web Search")
    web_search_available = st.session_state.web_search_available
    use_web_search = st.checkbox("Enable web search", value=web_search_available is not None, disabled=web_search_available is None)
    
    if web_search_available is None:
        st.warning("⚠️ Web search is unavailable due to a initialization error. Using knowledge base only.")
    
    # Voice settings
    st.markdown("#### Voice Output")
    enable_voice = st.checkbox("Enable voice output", value=st.session_state.tts is not None, disabled=st.session_state.tts is None)
    
    if st.session_state.tts is None:
        st.warning("⚠️ Voice output is unavailable due to an initialization error.")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temperature = st.slider("Response temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max response tokens", 100, 2000, 1000)
        
        # Knowledge base management
        st.markdown("#### Knowledge Base Management")
        
        # Button to rebuild knowledge base
        if st.button("Rebuild Knowledge Base"):
            with st.spinner("Rebuilding knowledge base..."):
                success = run_init_knowledge_base()
                if not success:
                    st.error("Failed to rebuild knowledge base. Check the logs for details.")

# Main content
st.markdown('<h1 class="main-header">Leadership Coach AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-powered leadership development companion</p>', unsafe_allow_html=True)

# Initialize or get query processor
kb_initialized = (
    (st.session_state.vector_store is not None and 
     st.session_state.vector_store.vectors is not None and 
     len(st.session_state.vector_store.metadata or []) > 0) and
    (os.path.exists(os.path.join(CHUNKS_DIR, "transcript_chunks_improved.json")) or
     os.path.exists(os.path.join(CHUNKS_DIR, "transcript_chunks.json")))
)

if st.session_state.query_processor is None and kb_initialized:
    try:
        st.session_state.query_processor = initialize_query_processor()
    except Exception as e:
        logger.error(f"Error initializing query processor: {str(e)}")
        logger.error(traceback.format_exc())
        st.session_state.query_processor = None

# Show appropriate UI based on initialization status
if st.session_state.query_processor is None:
    # Knowledge base status check
    if kb_initialized:
        try:
            # Try one more time to initialize the query processor
            st.session_state.query_processor = initialize_query_processor()
            
            # Display welcome message if initialization succeeded
            if st.session_state.query_processor and not st.session_state.messages:
                st.markdown(
                    """
                    <div class="welcome-message">
                        <h3>👋 Welcome to Leadership Coach AI!</h3>
                        <p>I'm your specialized leadership coach, providing guidance based on insights from leadership interviews and web resources.</p>
                        <p>Ask me any question about leadership, management strategies, professional development, or business acumen!</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="error-message">
                        <h3>❌ Error Initializing AI Engine</h3>
                        <p>There was an error loading the AI engine. Please try again or check the logs for details.</p>
                        <p>You can try rebuilding the knowledge base from the sidebar.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        except Exception as e:
            logger.error(f"Error in final attempt to initialize query processor: {str(e)}")
            logger.error(traceback.format_exc())
            st.markdown(
                f"""
                <div class="error-message">
                    <h3>❌ Error Initializing AI Engine</h3>
                    <p>There was an error loading the AI engine: {str(e)}</p>
                    <p>Please try rebuilding the knowledge base from the sidebar.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="info-message">
                <h3>⚙️ Knowledge Base Needed</h3>
                <p>The knowledge base needs to be built before you can start asking questions.</p>
                <p>Please click the "Rebuild Knowledge Base" button in the Advanced Settings section of the sidebar.</p>
                <p><strong>Note:</strong> This process will download and process YouTube transcripts, which may take several minutes.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.stop()
else:
    # Display welcome message if no messages yet
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-message">
                <h3>👋 Welcome to Leadership Coach AI!</h3>
                <p>I'm your specialized leadership coach, providing guidance based on insights from leadership interviews and web resources.</p>
                <p>Ask me any question about leadership, management strategies, professional development, or business acumen!</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Display feature status message
    feature_warnings = []
    if not st.session_state.web_search_available:
        feature_warnings.append("Web search is unavailable due to a compatibility issue. Using knowledge base only.")
    else:
        # If web search is available, don't add a warning
        pass
    
    if st.session_state.tts is None:
        feature_warnings.append("Voice output is unavailable. Text responses will be provided.")
    
    if feature_warnings:
        st.markdown(
            f"""
            <div class="warning-message">
                <h4>⚠️ Some Features Unavailable</h4>
                <ul>
                    {''.join(f'<li>{warning}</li>' for warning in feature_warnings)}
                </ul>
                <p>The app will continue to function with available features.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Display message about transcript support
st.markdown(
    """
    <div class="info-message">
        <h4>📝 Enhanced Transcript Support</h4>
        <p>Our system now supports multiple languages for transcripts, automatically translating non-English content. If videos lack transcripts, we'll still extract essential metadata to provide the best possible responses.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><div class="message-content">{content}</div></div>', unsafe_allow_html=True)
    else:
        # Split content and sources if available
        if isinstance(content, dict):
            response_text = content.get("response", "")
            
            # Extract sources using our helper function
            sources = extract_sources_from_response(content)
            sources_html = format_sources_for_display(sources)
            
            # Display response with sources
            st.markdown(
                f'<div class="chat-message assistant-message">'
                f'<div class="message-content">{response_text}</div>'
                f'{sources_html}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Display audio player if available
            if "audio_base64" in content and enable_voice and st.session_state.tts is not None:
                audio_base64 = content["audio_base64"]
                st.markdown(
                    f'<div class="audio-player">'
                    f'<audio controls autoplay="true">'
                    f'<source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">'
                    f'</audio></div>',
                    unsafe_allow_html=True
                )
        else:
            # Simple text response
            st.markdown(f'<div class="chat-message assistant-message"><div class="message-content">{content}</div></div>', unsafe_allow_html=True)

# Chat input
if kb_initialized and st.session_state.query_processor:
    try:
        if query := st.chat_input("Ask me about leadership..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            st.markdown(f'<div class="chat-message user-message"><div class="message-content">{query}</div></div>', unsafe_allow_html=True)
            
            # Process query
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    
                    # Process query with robust error handling
                    try:
                        # Check if web search is available
                        always_use_web = use_web_search and st.session_state.web_search_available
                        
                        # Process the query
                        response_data = st.session_state.query_processor.process_query(
                            query=query,
                            kb_results_count=kb_results_count,
                            always_use_web=always_use_web,
                            min_kb_score=0.5
                        )
                    except Exception as e:
                        logger.error(f"Error processing query: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        # Attempt to reinitialize query processor
                        st.warning("Attempting to recover from error...")
                        st.session_state.query_processor = initialize_query_processor()
                        
                        if st.session_state.query_processor:
                            # Try again with reinitialized processor
                            response_data = st.session_state.query_processor.process_query(
                                query=query,
                                kb_results_count=kb_results_count,
                                always_use_web=always_use_web,
                                min_kb_score=0.5
                            )
                        else:
                            raise Exception("Failed to recover from error. Please rebuild the knowledge base.")
                    
                    query_time = time.time() - start_time
                    logger.info(f"Query processed in {query_time:.2f} seconds")
                    
                    # Generate audio if enabled
                    if enable_voice and st.session_state.tts is not None:
                        with st.spinner("Generating audio..."):
                            try:
                                response_text = response_data.get("response", "")
                                if response_text:
                                    _, audio_bytes = st.session_state.tts.process_long_text(response_text)
                                    audio_base64 = st.session_state.tts.get_audio_base64(audio_bytes)
                                    response_data["audio_base64"] = audio_base64
                            except Exception as e:
                                logger.error(f"Error generating audio: {str(e)}")
                                logger.error(traceback.format_exc())
                                st.warning("There was an error generating the audio. The response will be displayed without audio.")
                    
                    # Log conversation
                    log_conversation(query, response_data, log_dir=LOGS_DIR)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": response_data})
                    
                    # Display assistant message
                    response_text = response_data.get("response", "")
                    
                    # Extract sources using our helper function
                    sources = extract_sources_from_response(response_data)
                    sources_html = format_sources_for_display(sources)
                    
                    st.markdown(
                        f'<div class="chat-message assistant-message">'
                        f'<div class="message-content">{response_text}</div>'
                        f'{sources_html}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display audio player if available
                    if enable_voice and st.session_state.tts is not None and "audio_base64" in response_data:
                        audio_base64 = response_data["audio_base64"]
                        st.markdown(
                            f'<div class="audio-player">'
                            f'<audio controls autoplay="true">'
                            f'<source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">'
                            f'</audio></div>',
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    error_response = {
                        "response": f"I'm sorry, I encountered an error while processing your request: {str(e)}. Please try again or rebuild the knowledge base if the issue persists.",
                        "sources": []
                    }
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                    handle_runtime_error(e, "Error processing query")
    except Exception as e:
        # Global error handler for any unexpected errors
        handle_runtime_error(e, "Unexpected error in chat interface")

# Footer
st.markdown("---")
st.markdown(
    "Leadership Coach AI uses content from leadership interviews to provide specialized guidance. "
    "All responses include references to the source material. The system supports multiple languages and handles videos with or without transcripts."
) 