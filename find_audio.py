#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path

def find_audio_files(search_dir=None, extensions=None):
    """
    Find audio files in the specified directory or common music locations.
    
    Args:
        search_dir (str): Directory to search in (optional)
        extensions (list): List of audio file extensions to look for
        
    Returns:
        list: List of found audio files
    """
    if extensions is None:
        extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aiff', '.aif']
    
    # Directories to search if none provided
    if search_dir is None:
        music_dirs = []
        
        # Add user's home directory
        home = str(Path.home())
        music_dirs.append(home)
        
        # Add common music directories
        music_dirs.append(os.path.join(home, 'Music'))
        music_dirs.append(os.path.join(home, 'Downloads'))
        
        # Common macOS paths
        if sys.platform == 'darwin':
            music_dirs.append('/Library/Audio/Apple Loops/Apple/Apple Loops for GarageBand')
            music_dirs.append(os.path.join(home, 'Library/Audio/Apple Loops/User Loops'))
        
        # Common Windows paths
        elif sys.platform == 'win32':
            music_dirs.append(os.path.join(os.environ.get('USERPROFILE', ''), 'Music'))
            music_dirs.append(os.path.join(os.environ.get('PUBLIC', ''), 'Music'))
        
        # Common Linux paths
        elif sys.platform.startswith('linux'):
            music_dirs.append(os.path.join(home, 'Music'))
            music_dirs.append('/usr/share/sounds')
    else:
        music_dirs = [search_dir]
    
    # Find all audio files
    all_files = []
    
    for directory in music_dirs:
        if not os.path.exists(directory):
            continue
        
        print(f"Searching in {directory}...")
        
        # Get all files with audio extensions in the directory (recursively)
        for ext in extensions:
            try:
                matches = glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True)
                all_files.extend(matches)
            except Exception as e:
                print(f"Error searching in {directory}: {e}")
    
    # Filter out duplicates
    return sorted(list(set(all_files)))

def main():
    """
    Main function to find and display audio files
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        search_dir = None
    
    print("Searching for audio files...")
    files = find_audio_files(search_dir)
    
    if not files:
        print("No audio files found. Try specifying a directory with audio files.")
        return
    
    print(f"\nFound {len(files)} audio files:\n")
    
    # Display the results
    for i, file in enumerate(files[:50]):  # Limit display to 50 files
        print(f"{i+1}. {file}")
    
    if len(files) > 50:
        print(f"...and {len(files) - 50} more files")
    
    # Ask if user wants to use one of these files
    print("\nTo use one of these files with HandDJ, run:")
    print("python hand_dj.py <file_path>")
    
    # Offer to test one of the files
    try:
        choice = input("\nEnter a number to test a file with HandDJ (or press Enter to exit): ")
        if choice and choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(files):
                print(f"\nRunning: python hand_dj.py \"{files[index]}\"")
                os.system(f'python hand_dj.py "{files[index]}"')
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main() 