#!/usr/bin/env python3
"""
Air DJ Controller Launcher
Simple launcher script for the DJ controller with webcam interface
"""

import sys
import os
import argparse
import random

def get_available_songs():
    """Get list of available songs from songs folder with required stem files"""
    songs_folder = "songs"
    available_songs = []
    
    if not os.path.exists(songs_folder):
        print(f"❌ Songs folder '{songs_folder}' not found!")
        return []
    
    # Get all directories (stem folders) in songs folder
    for item in sorted(os.listdir(songs_folder)):
        item_path = os.path.join(songs_folder, item)
        
        # Only check directories, skip files and special items
        if not os.path.isdir(item_path):
            continue
            
        # Check if the directory contains required vocal and instrumental files
        if has_required_stems(item_path):
            available_songs.append(item)
    
    return available_songs

def has_required_stems(song_path):
    """Check if a song directory has the required Vocals and Instrumental files"""
    try:
        files = os.listdir(song_path)
        has_vocals = any(file.lower().startswith('vocals') and file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac')) for file in files)
        has_instrumental = any(file.lower().startswith('instrumental') and file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac')) for file in files)
        
        return has_vocals and has_instrumental
    except (OSError, PermissionError):
        return False

def interactive_song_selection():
    """Interactive song selection menu"""
    available_songs = get_available_songs()
    
    if not available_songs:
        print("❌ No songs found in songs/ folder!")
        return None, None
    
    print(f"\n🎵 Choose Your Songs ({len(available_songs)} available)")
    print("=" * 50)
    
    # Display available songs
    for i, song in enumerate(available_songs):
        print(f"  {i+1:2d}. {song}")
    print()
    
    def select_song(deck_name):
        while True:
            try:
                choice = input(f"Select song for {deck_name} (1-{len(available_songs)}, or Enter for random): ").strip()
                
                if not choice:  # Random selection
                    selected = random.choice(available_songs)
                    print(f"🎲 Random: {selected}")
                    return selected
                
                idx = int(choice) - 1
                if 0 <= idx < len(available_songs):
                    return available_songs[idx]
                else:
                    print(f"❌ Please enter 1-{len(available_songs)}")
            
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input")
            except EOFError:
                return None
    
    deck1_song = select_song("DECK 1")
    if deck1_song is None:
        return None, None
        
    deck2_song = select_song("DECK 2")
    if deck2_song is None:
        return None, None
    
    return deck1_song, deck2_song

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Air DJ Controller - Hand Tracking DJ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python air_dj.py              # BPM sync enabled (default)
  python air_dj.py --unsync     # Disable BPM sync
  python air_dj.py --default    # Skip song selection
        """
    )
    
    # BPM and song selection arguments
    parser.add_argument('--unsync', action='store_true', help='Disable BPM synchronization between decks')
    parser.add_argument('--default', action='store_true', help='Use default hardcoded songs (skip interactive)')
    
    args = parser.parse_args()
    
    # Handle special commands first  
    if args.default:
        use_interactive = False
    else:
        use_interactive = True  # Default behavior
    
    # Determine BPM sync setting
    enable_bpm_sync = not args.unsync
    
    print("🎧 AIR DJ - Hand Tracking DJ Controller")
    
    # Show sync status
    if not enable_bpm_sync:
        print("🔄 BPM SYNC: DISABLED")
    
    print("✋ Use PINCH gestures to control • Press 'q' to quit\n")
    
    try:
        # Import and start the controller
        from dj_controller import DJController
        
        print("🎧 Starting Air DJ...")
        
        # Handle song selection
        selected_songs = None
        if use_interactive:
            selected_songs = interactive_song_selection()
        
        controller = DJController(enable_bpm_sync=enable_bpm_sync, selected_songs=selected_songs)
        controller.run()
        
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nPlease make sure you're running this in the virtual environment:")
        print("source venv_py311/bin/activate")
        print("python air_dj.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError starting DJ Controller: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
