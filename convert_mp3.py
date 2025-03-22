#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import glob

def check_dependencies():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False

def convert_mp3_to_wav(input_mp3, output_wav=None):
    """Convert an MP3 file to WAV format using ffmpeg"""
    if not output_wav:
        # Generate output filename if not provided
        output_wav = os.path.splitext(input_mp3)[0] + ".wav"
    
    # Run ffmpeg to convert the file
    try:
        print(f"Converting: {input_mp3} -> {output_wav}")
        result = subprocess.run(
            ["ffmpeg", "-i", input_mp3, "-acodec", "pcm_s16le", "-ar", "44100", output_wav],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode == 0:
            print(f"Success! Created: {output_wav}")
            return True
        else:
            print(f"Error converting file: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"Exception during conversion: {e}")
        return False

def convert_songs_folder():
    """Convert all MP3 files in the songs folder to WAV format"""
    songs_folder = os.path.join(os.getcwd(), "songs")
    if not os.path.exists(songs_folder):
        print("Songs folder not found. Creating it...")
        os.makedirs(songs_folder)
        return
    
    # Get all MP3 files in the songs folder
    mp3_files = glob.glob(os.path.join(songs_folder, "*.mp3"))
    
    if not mp3_files:
        print("No MP3 files found in the songs folder")
        return
    
    print(f"Found {len(mp3_files)} MP3 files to convert")
    
    # Convert each MP3 file
    for mp3_file in mp3_files:
        convert_mp3_to_wav(mp3_file)

def main():
    """Main function"""
    # Check if ffmpeg is installed
    if not check_dependencies():
        print("Error: ffmpeg is not installed or not in your PATH")
        print("Please install ffmpeg to use this tool")
        print("  - On macOS: brew install ffmpeg")
        print("  - On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  - On Windows: Download from https://ffmpeg.org/download.html")
        return
    
    # Show menu
    print("=" * 60)
    print("             MP3 to WAV Converter for Hand DJ")
    print("=" * 60)
    print("\nThis tool converts MP3 files to WAV format for better compatibility")
    print("with the Hand DJ application.")
    
    print("\nWhat would you like to do?")
    print("1. Convert all MP3 files in the songs folder")
    print("2. Convert a specific MP3 file")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        # Convert all MP3 files in the songs folder
        convert_songs_folder()
    
    elif choice == '2':
        # Convert a specific MP3 file
        input_file = input("\nEnter the path to the MP3 file: ")
        if not os.path.exists(input_file):
            print("Error: File not found")
            return
        
        convert_mp3_to_wav(input_file)
    
    elif choice == '3':
        # Exit
        print("\nExiting...")
        return
    
    else:
        print("\nInvalid choice. Please select a number from 1-3.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 