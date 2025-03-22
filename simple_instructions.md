# Hand DJ - Instructions

## Overview
Hand DJ is an interactive application that lets you control music parameters using hand gestures through your webcam. The application has two modes:

1. **Regular mode** (uses MediaPipe for advanced hand tracking)
2. **Simple mode** (uses color tracking for better compatibility)

## Requirements
- Webcam
- Python 3.8 or newer
- Required packages (install with `pip install -r requirements.txt`)
- Audio files (MP3 or WAV format) in the `songs` folder

## Quick Start

### Running the Application

```bash
# Basic usage (uses MediaPipe hand tracking)
python run.py

# Simple mode (color tracking)
python run.py --simple

# Specify a song
python run.py --song songname

# Convert MP3 files to WAV before running
python run.py --convert
```

## Using Simple Mode

The simple mode uses color tracking to detect blue and red objects in the webcam feed.

### What You Need
- A **blue object** (like a blue sticky note, marker, or glove)
- A **red object** (like a red sticky note, marker, or glove)

### Controls
1. **Volume Control**: Use the **red object**. Move it up and down to control volume.
   - Higher position = louder
   - Lower position = quieter

2. **Pitch Control**: Use the **blue object**. Move it left and right to control pitch.
   - Left side = lower pitch (down to -12 semitones)
   - Right side = higher pitch (up to +12 semitones)
   - The pitch range is now wider for more musical control

3. **Filter Control**: The **blue object's** vertical position controls a lowpass filter.
   - Higher position = brighter sound
   - Lower position = more filtered sound

4. **Distance Control**: The distance between the red and blue objects affects volume.
   - Wider apart = louder
   - Closer together = quieter
   - The sensitivity has been enhanced for easier control

### Tips for Best Results
- Use bright, saturated colors for better tracking
- Ensure good lighting in the room
- Keep the colored objects separate from each other
- Avoid having other red or blue objects in the camera view

### Keyboard Controls
- Press `ESC` or `Q` to quit the application
- Press `M` to toggle the color mask view (useful for debugging tracking issues)

## Using Regular Mode (MediaPipe)

The regular mode uses MediaPipe to track your actual hands.

### Controls
1. **Volume Control**: Distance between your hands (pinch gestures)
   - Wider = louder
   - Closer = quieter

2. **Pitch Control**: Height of your right hand
   - Higher = higher pitch
   - Lower = lower pitch

3. **Filter Control**: Height of your left hand
   - Higher = brighter sound
   - Lower = more filtered sound

## Enhanced Audio Features

The application now includes several audio quality enhancements:

1. **Enhanced Sound Quality**
   - Added reverb for more natural sound
   - Improved audio processing for richer tone
   - Better handling of MP3 files with higher quality playback

2. **Widened Control Ranges**
   - Pitch range has been expanded from +/-5 to +/-12 semitones
   - Speed control range has been expanded from 0.5-2.0x to 0.25-3.0x
   - More sensitive control mapping for better expressiveness

3. **Improved Sine Wave Synthesis**
   - Added harmonic overtones for a richer sound when no audio file is loaded
   - Better volume balance across frequencies
   - More natural pitch transitions

4. **Dynamic Audio Processing**
   - Reverb changes dynamically based on playback speed
   - Filter cutoff adjusts based on pitch for more musical results
   - Overall louder, clearer output

## Troubleshooting

### MP3 File Issues
If you have trouble playing MP3 files, try these solutions:

1. Convert your MP3 files to WAV format using the included converter:
   ```
   python convert_mp3.py
   ```
   or
   ```
   python run.py --convert
   ```

2. Make sure your MP3 files are in the `songs` folder

### Tracking Issues in Simple Mode
If the color tracking isn't working well:

1. Try adjusting the lighting in the room
2. Use more vivid blue and red objects
3. Press `M` to see the color mask - your colored objects should appear as white areas

### Tracking Issues in Regular Mode
If hand tracking isn't working well:

1. Make sure your hands are clearly visible in the webcam view
2. Try running the calibration tool:
   ```
   python calibrate.py
   ```

### Crash on Startup
If the application crashes on startup:

1. Try using simple mode instead:
   ```
   python run.py --simple
   ```

2. Check that all dependencies are installed properly:
   ```
   pip install -r requirements.txt
   ```

## Enjoy!
Have fun making music with your hands! Experiment with different movements and gestures to create interesting sound effects. 