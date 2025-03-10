"""
Helper functions for the Leadership Coach AI.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

def ensure_directories_exist(directories: List[str]) -> None:
    """
    Ensure that the specified directories exist.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_sources_for_display(sources: List[Dict]) -> str:
    """
    Format sources for display in the UI.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted sources as HTML
    """
    if not sources or not isinstance(sources, list):
        return ""
    
    html = "<div class='sources'><h4>Sources:</h4><ul>"
    
    for i, source in enumerate(sources):
        if not source or not isinstance(source, dict):
            continue
            
        source_type = source.get("type", "unknown")
        title = source.get("title", "Untitled")
        url = source.get("url", "#")
        snippet = source.get("text_snippet", "")
        
        # Sanitize values to prevent HTML issues
        title = title.replace("<", "&lt;").replace(">", "&gt;") if title else "Untitled"
        url = url or "#"
        snippet = snippet.replace("<", "&lt;").replace(">", "&gt;") if snippet else ""
        
        # Create the source item HTML
        icon = "🎥" if source_type == "video" else "🌐"
        html += f"<li><strong>{icon} {title}</strong>"
        
        if snippet:
            html += f"<br><em>{snippet}</em>"
            
        html += f"<br><a href='{url}' target='_blank'>View Source</a></li>"
    
    html += "</ul></div>"
    return html

def log_conversation(
    query: str, 
    response: Dict[str, Any], 
    log_dir: str = "data/logs"
) -> None:
    """
    Log a conversation to a file.
    
    Args:
        query: User query
        response: Response data
        log_dir: Directory to store logs
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response.get("response", ""),
        "sources": response.get("sources", []),
        "model": response.get("model", "unknown"),
        "usage": response.get("usage", {}),
        "kb_results_count": response.get("kb_results_count", 0),
        "web_search_used": response.get("web_search_used", False),
        "web_results_count": response.get("web_results_count", 0)
    }
    
    # Generate log filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"conversation_log_{date_str}.jsonl")
    
    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def extract_youtube_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from a URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        YouTube video ID or None if not found
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/user\/\S+\/\S+\/|youtube\.com\/user\/\S+\/|youtube\.com\/\S+\/\S+\/|youtube\.com\/\S+\/|youtube\.com\/attribution_link\?a=\S+&u=\/watch\?v=|youtube\.com\/attribution_link\?a=\S+&u=)([^"&?\/\s]{11})',
        r'(?:youtube\.com\/watch\?time_continue=\d+&v=)([^"&?\/\s]{11})',
        r'(?:youtube\.com\/watch\?v=)([^"&?\/\s]{11})',
        r'(?:youtu\.be\/)([^"&?\/\s]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def extract_youtube_playlist_id(url: str) -> Optional[str]:
    """
    Extract YouTube playlist ID from a URL.
    
    Args:
        url: YouTube playlist URL
        
    Returns:
        YouTube playlist ID or None if not found
    """
    patterns = [
        r'(?:youtube\.com\/playlist\?list=)([^"&?\/\s]+)',
        r'(?:youtube\.com\/watch\?v=[^"&?\/\s]+&list=)([^"&?\/\s]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None 