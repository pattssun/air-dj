#!/usr/bin/env python3
"""
Test script for Hand DJ application
This script does a basic verification of the application components
"""

import os
import sys
import importlib
import traceback

def check_module(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_audio_libraries():
    """Check if audio libraries work properly"""
    try:
        import pyo
        server = pyo.Server(duplex=0).boot()
        server.start()
        
        # Try to create a simple sine wave
        sine = pyo.Sine(freq=440, mul=0.1).out()
        print("Audio test successful: Generated test tone")
        
        # Clean up
        sine.stop()
        server.stop()
        return True
    except Exception as e:
        print(f"Audio test failed: {e}")
        return False

def check_video_capture():
    """Check if video capture works"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to open webcam")
            return False
        
        # Capture a single frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Failed to capture frame from webcam")
            return False
        
        # Print frame info
        height, width = frame.shape[:2]
        print(f"Video test successful: Captured frame ({width}x{height})")
        
        # Clean up
        cap.release()
        return True
    except Exception as e:
        print(f"Video test failed: {e}")
        return False

def check_song_folder():
    """Check if songs folder exists and contains audio files"""
    if not os.path.exists("songs"):
        print("Songs folder does not exist, creating it...")
        os.makedirs("songs")
        print("Created songs folder. Please add some audio files to it.")
        return False
    
    # Check for audio files
    audio_files = []
    for ext in ['.mp3', '.wav', '.ogg', '.flac', '.aiff', '.aif']:
        audio_files.extend([f for f in os.listdir("songs") if f.endswith(ext)])
    
    if not audio_files:
        print("No audio files found in the songs folder")
        return False
    
    print(f"Found {len(audio_files)} audio files in the songs folder")
    return True

def test_imports():
    """Test importing the main application modules"""
    results = []
    
    modules = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pyo", "pyo"),
        ("MediaPipe", "mediapipe"),
        ("Convert Script", "convert_mp3"),
        ("Simple Hand DJ", "simple_hand_dj"),
        ("Hand DJ", "hand_dj"),
        ("Run Script", "run")
    ]
    
    print("\nTesting module imports:")
    print("=" * 40)
    
    for name, module in modules:
        print(f"Testing {name}...", end=" ")
        if check_module(module):
            print("✓ OK")
            results.append(True)
        else:
            print("✗ FAILED")
            results.append(False)
    
    return all(results)

def main():
    """Main function"""
    print("=" * 60)
    print("Hand DJ Application Test Suite")
    print("=" * 60)
    
    print("\nRunning basic tests to verify application components...")
    
    # Test module imports
    import_success = test_imports()
    
    # Test audio libraries
    print("\nTesting audio system...")
    audio_success = check_audio_libraries()
    
    # Test video capture
    print("\nTesting video capture...")
    video_success = check_video_capture()
    
    # Check songs folder
    print("\nChecking songs folder...")
    songs_success = check_song_folder()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Module imports: {'✓ OK' if import_success else '✗ FAILED'}")
    print(f"Audio system: {'✓ OK' if audio_success else '✗ FAILED'}")
    print(f"Video capture: {'✓ OK' if video_success else '✗ FAILED'}")
    print(f"Songs folder: {'✓ OK' if songs_success else '✗ FAILED'}")
    
    # Overall result
    print("\nOverall Result:")
    if import_success and audio_success and video_success:
        print("✓ The application should work correctly")
        if not songs_success:
            print("  (but you need to add audio files to the songs folder)")
    else:
        print("✗ There might be issues with the application")
        print("  Please fix the failed components before running the app")
    
    print("\nSuggested next steps:")
    if not import_success:
        print("- Try reinstalling dependencies: pip install -r requirements.txt")
    if not audio_success:
        print("- Check audio system configuration")
    if not video_success:
        print("- Check webcam connection and permissions")
    if not songs_success:
        print("- Add audio files to the songs folder")
    
    if import_success and audio_success and video_success:
        print("- Run the application: python run.py")
        if not songs_success:
            print("  (after adding audio files to the songs folder)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during testing: {e}")
        traceback.print_exc()
    
    input("\nPress Enter to exit...") 