"""
Audio-based YouTube transcription test script.
This script tests downloading audio and using Whisper for transcription.
"""

import os
import argparse
import json
import logging
from src.knowledge_base.youtube_extractor import test_audio_transcription

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test audio-based transcription")
    parser.add_argument("--video", help="YouTube video ID to test")
    parser.add_argument("--check-system", action="store_true", help="Check system for dependencies")
    args = parser.parse_args()
    
    # Check system if requested
    if args.check_system:
        check_system_dependencies()
        return
    
    # Process video or use default
    video_id = args.video if args.video else "JQhkbN75Ohs"
    print(f"Testing audio-based transcription for video: {video_id}")
    
    # Run test function
    test_audio_transcription(video_id)

def check_system_dependencies():
    """Check if system has all required dependencies."""
    print("Checking system dependencies for audio-based transcription...")
    
    # Check for FFmpeg
    print("\nChecking for FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and working")
        else:
            print("❌ FFmpeg installation issue detected")
    except FileNotFoundError:
        print("❌ FFmpeg not found - please install FFmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  macOS: brew install ffmpeg")
        print("  Linux: apt install ffmpeg or equivalent")
    
    # Check for Whisper
    print("\nChecking for Whisper...")
    try:
        import whisper
        print("✅ Whisper is installed and available")
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"✅ CUDA is available - Device: {device_name}")
                print(f"  Transcription will be much faster with GPU acceleration")
            else:
                print("⚠️ CUDA is not available - Whisper will run on CPU (slower)")
                print("  Transcription will work but may be slow")
        except Exception as e:
            print(f"⚠️ CUDA check failed: {str(e)}")
            print("  Transcription will likely run on CPU (slower)")
    except ImportError:
        print("❌ Whisper is not installed")
        print("  Install with: pip install openai-whisper")
    except Exception as e:
        print(f"❌ Whisper is installed but has issues: {str(e)}")
        print("  You may need to reinstall: pip install --force-reinstall openai-whisper")
    
    # Check for yt-dlp
    print("\nChecking for yt-dlp...")
    try:
        import yt_dlp
        print("✅ yt-dlp is installed and available")
    except ImportError:
        print("❌ yt-dlp is not installed")
        print("  Install with: pip install yt-dlp")

if __name__ == "__main__":
    main() 