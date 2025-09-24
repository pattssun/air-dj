#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import glob
import argparse

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    clear_screen()
    print("=" * 60)
    print("               HAND DJ - Air Mixing with Hand Tracking")
    print("=" * 60)
    print("\nControl music using just your hands in the air!")
    print("Left hand pinch: Speed | Right hand pinch: Pitch | Hand distance: Volume\n")

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import numpy
        from pyo import Server
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install -r requirements.txt")
        return False

def find_song_with_extension(song_name):
    """Find a song file with either MP3 or WAV extension"""
    # Check for WAV file first (preferred)
    wav_path = os.path.join("songs", f"{song_name}.wav")
    if os.path.exists(wav_path):
        return wav_path
        
    # Then check for MP3
    mp3_path = os.path.join("songs", f"{song_name}.mp3")
    if os.path.exists(mp3_path):
        return mp3_path
        
    return None

def get_songs_from_folder():
    """Get a list of all audio files in the songs folder"""
    # Create songs directory if it doesn't exist
    if not os.path.exists("songs"):
        os.makedirs("songs")
        print("Created 'songs' directory. Please add some audio files to it.")
        return []
    
    # Get all WAV and MP3 files
    song_files = []
    song_files.extend(glob.glob(os.path.join("songs", "*.wav")))
    song_files.extend(glob.glob(os.path.join("songs", "*.mp3")))
    
    # Remove duplicates (songs that have both MP3 and WAV versions)
    unique_songs = {}
    for song_path in song_files:
        base_name = os.path.splitext(os.path.basename(song_path))[0]
        ext = os.path.splitext(song_path)[1].lower()
        
        # Prefer WAV over MP3
        if base_name not in unique_songs or ext == '.wav':
            unique_songs[base_name] = song_path
    
    return list(unique_songs.values())

def select_song(default_song="timeless"):
    """Allow user to select a song from the songs folder"""
    songs = get_songs_from_folder()
    
    if not songs:
        print("No audio files found in the 'songs' folder.")
        return None
    
    # Check if default song exists
    default_song_path = find_song_with_extension(default_song)
    if default_song_path:
        print(f"Using default song: {default_song_path}")
        return default_song_path
    
    # List all available songs
    print("\nAvailable songs:")
    for i, song_path in enumerate(songs):
        print(f"{i+1}. {os.path.basename(song_path)}")
    
    # Get user selection
    try:
        selection = input("\nSelect a song number (or press Enter for the first song): ")
        if selection.strip() == "":
            return songs[0]
        
        selection_index = int(selection) - 1
        if 0 <= selection_index < len(songs):
            return songs[selection_index]
        else:
            print("Invalid selection. Using the first song.")
            return songs[0]
    except:
        print("Invalid input. Using the first song.")
        return songs[0] if songs else None

def convert_mp3_to_wav():
    """Run the MP3 to WAV converter"""
    convert_script = os.path.join(os.getcwd(), "audio_converter.py")
    if not os.path.exists(convert_script):
        print("Error: audio_converter.py script not found!")
        return
    
    try:
        subprocess.run([sys.executable, convert_script], check=True)
    except subprocess.SubprocessError as e:
        print(f"Error running converter: {e}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hand DJ - Control music with hand gestures")
    parser.add_argument('--simple', action='store_true', help='Use simple color tracking instead of MediaPipe hand tracking')
    parser.add_argument('--song', type=str, help='Specify a song to play from the songs folder')
    parser.add_argument('--convert', action='store_true', help='Run MP3 to WAV converter before starting')
    parser.add_argument('--hq', action='store_true', help='Enable high quality audio enhancements (default: enabled)')
    parser.add_argument('--debug', action='store_true', help='Show detailed error information')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Run converter if requested
    if args.convert:
        convert_mp3_to_wav()
    
    # Get song path
    song_path = None
    if args.song:
        song_path = find_song_with_extension(args.song)
        if not song_path:
            print(f"Song '{args.song}' not found in songs folder.")
            song_path = select_song()
    else:
        song_path = select_song()
    
    if not song_path:
        print("No song selected. Using default sine wave synthesis.")
    else:
        print(f"Selected song: {song_path}")
    
    print("\n=== Enhanced Audio & Controls ===")
    print("- Wider range of pitch and speed control")
    print("- Improved audio quality with reverb")
    print("- More sensitive gesture tracking")
    print("- Added harmonic enhancement\n")
    
    # Launch the appropriate version
    if args.simple:
        print("Starting simple color tracking mode...")
        from color_tracking_dj import SimpleHandDJ
        app = SimpleHandDJ(song_path)
        app.setup_audio_player(song_path)
        app.run()
    else:
        print("Starting MediaPipe hand tracking mode...")
        try:
            from hand_tracking_dj import HandDJ
            app = HandDJ(song_path)
            app.run()
        except Exception as e:
            if args.debug:
                import traceback
                print(f"\nError starting MediaPipe version: {e}")
                print("\nDetailed error information:")
                traceback.print_exc()
            
            print("\nWould you like to:")
            print("1. Try again with MediaPipe")
            print("2. Fall back to simple version")
            print("3. Quit")
            
            choice = input("Enter your choice (1/2/3): ").strip()
            
            if choice == "1":
                print("\nRetrying with MediaPipe...")
                from hand_tracking_dj import HandDJ
                app = HandDJ(song_path)
                app.run()
            elif choice == "2":
                print("\nFalling back to simple version...")
                from color_tracking_dj import SimpleHandDJ
                app = SimpleHandDJ(song_path)
                app.setup_audio_player(song_path)
                app.run()
            else:
                print("\nExiting...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nError: {e}") 