# Hand DJ Usage Guide

## Getting Started with Songs

The HandDJ application now supports loading songs from a dedicated `songs` folder. Follow these steps to use your own music:

1. Place your MP3 files in the `songs` folder in the project directory.
2. When you start the application, it will automatically detect these songs.
3. By default, the app will look for a file named `timeless.mp3`.

## Using the App

1. Start the application:
   ```
   python run.py
   ```

2. The app will automatically show you the songs available in the `songs` folder.

3. Choose from the menu options:
   - Option 1: Start regular Hand DJ (uses MediaPipe for hand tracking)
   - Option 2: Start simple Hand DJ (uses color tracking - better compatibility)
   - Option 5: Select a specific song from the songs folder

## Controlling the Music

### Regular Hand DJ (Option 1)
- **Left hand pinch**: Controls playback speed (tighter pinch = slower)
- **Right hand pinch**: Controls pitch/frequency 
- **Distance between hands**: Controls volume

### Simple Hand DJ (Option 2)
For this version, you need colored objects (like colored paper, markers, or clothing):
- **Blue object**: Controls pitch (left/right position)
- **Red object**: Controls speed (up/down position)
- **Distance between objects**: Controls volume

## Troubleshooting

If you have issues with the MediaPipe version (Option 1):
1. Try the Simple Hand DJ (Option 2) which uses basic color tracking instead
2. Ensure you have good lighting for better hand/color detection
3. Use contrasting backgrounds for better tracking

## Adding Songs

1. Simply place your audio files in the `songs` folder
2. Supported formats: MP3, WAV, OGG, FLAC, AIFF
3. For the best experience, name your main song `timeless.mp3`

---

For detailed installation and dependency information, see the main README.md file. 