# Air DJ

Control audio parameters using hand gestures through your webcam!

## Project Overview

Air DJ is an interactive application that lets you control music playback parameters using hand gestures tracked by your laptop's webcam. With Air DJ, you can:

- Adjust volume, pitch, and filter effects in real-time
- Control music using either advanced hand tracking or simple color tracking
- Play your own MP3 or WAV files
- Create a fun, interactive DJ experience without specialized hardware

## Features

- **Two tracking modes**:
  - MediaPipe hand tracking (advanced mode)
  - Simple color tracking (compatibility mode)
- **Real-time audio manipulation**:
  - Volume control
  - Pitch shifting
  - Low-pass filtering
- **User-friendly interface**:
  - Visual feedback of tracked objects
  - Parameter display
  - Audio visualization
- **Audio file support**:
  - MP3 support
  - WAV support (recommended)
  - Built-in MP3 to WAV converter

## Requirements

- Python 3.8 or newer
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository or download the source code:
   ```
   git clone https://github.com/yourusername/air-dj.git
   cd air-dj
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # source venv_py311/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `songs` folder and add your audio files:
   ```
   mkdir -p songs
   # Copy your MP3 or WAV files to the songs folder
   ```

## Usage

### Quick Start

Run the application in regular mode:
```
python air_dj.py
```

Run in simple color tracking mode:
```
python air_dj.py --simple
```

Run with a specific song:
```
python air_dj.py --song songname
```

Convert MP3 files to WAV before running:
```
python air_dj.py --convert
```

### Converting MP3 to WAV

If you have trouble with MP3 playback, you can convert your files to WAV format:
```
python audio_converter.py
```

This requires ffmpeg to be installed on your system.

## Detailed Instructions

For detailed instructions on how to use the application, see [simple_instructions.md](simple_instructions.md).

## Troubleshooting

If you encounter issues:

1. Make sure all dependencies are correctly installed
2. Try the simple mode if the MediaPipe tracking doesn't work
3. Convert MP3 files to WAV format for better compatibility
4. Ensure you have good lighting for better tracking

## Acknowledgments

- Built with [MediaPipe](https://mediapipe.dev/)
- Audio processing using [pyo](http://ajaxsoundstudio.com/software/pyo/)
- Computer vision with [OpenCV](https://opencv.org/)
