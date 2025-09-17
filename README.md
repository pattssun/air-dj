# 🎧 Air DJ - Webcam-Controlled DJ Controller

Control a professional DJ interface using hand gestures through your webcam! Experience the future of DJing with precision hand tracking and professional-grade audio mixing.

## 🎛️ What is Air DJ?

Air DJ transforms your webcam into a professional DJ controller interface. Using advanced hand tracking, you can control audio playback, mix tracks, and manipulate stems with simple pinch gestures - no physical hardware required!

## ✨ Key Features

### 🤏 **Precision Hand Tracking**
- **Pinch Detection**: Use thumb + index finger pinches for button control
- **High Precision**: Refined 0.04 threshold for deliberate actions only
- **No Accidental Triggers**: Professional-grade control sensitivity

### 🎛️ **Professional DJ Interface**
- **Transparent Overlay**: Full DJ controller appears over your camera feed
- **Dual Deck Control**: Independent control of two audio tracks
- **Real-time Visualization**: Track position bars and jog wheel indicators
- **Professional Layout**: Industry-standard DJ controller design

### 🎵 **Advanced Audio Engine**
- **Stem Isolation**: Automatic loading of vocal and instrumental stems
- **Independent Timing**: Each deck has its own timeline (like Rekordbox/Serato)
- **Professional Position Management**: Proper cue, play/pause behavior
- **Real-time Effects**: Volume control and stem toggling

### 🎚️ **Professional Controls**

| Control | Function | Behavior |
|---------|----------|----------|
| **🎯 CUE** | Jump to cue point | Seeks to beginning for preview |
| **▶️ PLAY** | Start playback | Continues from current position |
| **⏸️ PAUSE** | Pause playback | Maintains position for resume |
| **🎤 VOCAL** | Toggle vocals | Real-time on/off at current position |
| **🎶 INSTRUMENTAL** | Toggle instrumental | Real-time on/off at current position |

### 📊 **Visual Feedback**
- **Track Progress Bars**: Real-time position with time display (mm:ss)
- **Jog Wheel Indicators**: Moving position markers
- **Stem Status Display**: Visual confirmation of vocal/instrumental state
- **Independent Visualization**: Each deck shows separate progress

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Audio files with stems (see Audio Setup below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/air-dj.git
   cd air-dj
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv_py311
   source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your music:**
   ```bash
   # Place stem folders in the songs/ directory
   # Each song should have separate vocal/instrumental files
   ```

### Launch Air DJ

```bash
source venv_py311/bin/activate
python air_dj.py
```

## 🎵 Audio Setup

### Supported Formats
- **Stems**: Separate vocal and instrumental tracks
- **File Types**: MP3, WAV
- **Folder Structure**: Each song in its own subfolder with stems

### Example Folder Structure
```
songs/
├── Song1 - Artist/
│   ├── Vocals - Song1.mp3
│   └── Instrumental - Song1.mp3
├── Song2 - Artist/
│   ├── Vocals - Song2.mp3
│   └── Instrumental - Song2.mp3
```

### Getting Stems
- Use AI stem separation tools (Spleeter, LALAL.AI, etc.)
- Download pre-separated stems from music platforms
- The included songs folder has example stems ready to use

## 🎮 How to Use

### Hand Positioning
1. **Position yourself** in front of the camera
2. **Extend your hand** toward the camera
3. **Use thumb + index finger** for pinch gestures
4. **Pinch close together** over buttons to activate

### DJ Workflow
1. **Start with CUE** - Preview tracks from the beginning
2. **Press PLAY** - Start playback from current position
3. **Toggle VOCAL/INST** - Mix stems in real-time
4. **Use PAUSE** - Stop but maintain position
5. **Press PLAY again** - Resume from where you paused
6. **Use CUE** - Jump back to beginning for next mix

### Professional Tips
- **CUE is the only control that resets position** (professional DJ behavior)
- **PLAY/PAUSE maintains position** (no unwanted restarts)
- **VOCAL/INST toggles work at current position** (real-time mixing)
- **Each deck operates independently** (different timelines)

## 🔧 Technical Details

### Hand Tracking Precision
- **MediaPipe Integration**: Advanced hand landmark detection
- **Precision Threshold**: 0.04 distance for pinch confirmation
- **Frame-rate Optimized**: Smooth real-time tracking
- **Expanded Hit Areas**: Larger vocal/instrumental buttons (120x45px)

### Audio Engine Architecture
- **Pyo Audio Server**: Professional audio processing
- **Independent Deck Timing**: Separate timelines per deck
- **Volume-based Control**: Position-preserving playback management
- **Stem Isolation**: Clear vocal/instrumental separation

### Performance Features
- **Professional Position Management**: Like Rekordbox/Serato
- **Independent Track Timing**: Each deck has own timeline
- **Real-time Visualization**: Track progress and stem status
- **Optimized Pinch Detection**: Deliberate actions only

## 🎛️ Controller Layout

```
           DECK 1              CENTER               DECK 2
    
    [Jog Wheel 1]    [Hi] [Hi]   [Jog Wheel 2]
                     [Mid][Mid]
    [CUE] [VOCAL]    [Low][Low]   [CUE] [VOCAL]
    [PLAY][INST.]    [Vol][Vol]   [PLAY][INST.]
                   [Crossfader]
```

### Button Specifications
- **Vocal/Instrumental**: 120x45 pixels (80% larger than standard)
- **Separation**: 60px vertical gap (50% more space)
- **Hit Detection**: 10px margins for precise targeting
- **Visual Feedback**: Active state indication

## 🎯 Professional Features

### Track Position Management
- **Independent Timing**: Each deck maintains separate timeline
- **Professional Cue Behavior**: Only CUE resets position
- **Seamless Resume**: PLAY/PAUSE maintains exact position
- **Real-time Mixing**: Stem controls don't affect position

### Visual Feedback System
- **Progress Bars**: Show playback position with time (mm:ss)
- **Jog Wheel Indicators**: Rotating position markers
- **Stem Status**: Visual confirmation of vocal/instrumental state
- **Color Coding**: Green (playing), Gray (paused), Yellow (cue point)

### Precision Control
- **Deliberate Pinch Detection**: Requires close finger proximity
- **Large Target Buttons**: 80% bigger vocal/instrumental controls
- **Professional Spacing**: Prevents accidental button presses
- **Optimized Margins**: Precise hit detection

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Ensure no other apps are using the camera
- Try reconnecting the camera

**Audio not playing:**
- Verify audio files in songs folder
- Check that stems are properly named
- Ensure virtual environment is activated

**Hand tracking not working:**
- Improve lighting conditions
- Position hand clearly in camera view
- Check MediaPipe installation

**Pinch detection too sensitive/not sensitive enough:**
- The threshold is optimized at 0.04 for precision
- Ensure fingers are close together for activation
- Practice deliberate pinch gestures

### Performance Optimization
- **Good Lighting**: Ensure adequate lighting for hand tracking
- **Camera Position**: Position camera at eye level
- **Clean Background**: Minimize background clutter
- **Steady Hand**: Keep hand steady while pinching

## 📁 Project Structure

```
air-dj/
├── dj_controller.py          # Main DJ controller implementation
├── air_dj.py               # Application launcher
├── requirements.txt         # Python dependencies
├── audio_converter.py       # MP3 to WAV converter
├── hand_tracking_dj.py      # Legacy hand tracking (deprecated)
├── color_tracking_dj.py     # Color tracking fallback
├── demo.py                  # Basic demo script
├── songs/                   # Audio files directory
└── venv_py311/             # Virtual environment
```

## 🎉 What Makes Air DJ Special

### Professional DJ Software Experience
- **Industry Standard Behavior**: Works like Rekordbox, Serato, Virtual DJ
- **True Professional Workflow**: Proper cue, play/pause, and mixing controls
- **Independent Deck Control**: Each deck operates on its own timeline
- **Real-time Stem Mixing**: Live vocal/instrumental control

### Precision Hand Tracking
- **No Hardware Required**: Turn any camera into a DJ controller
- **Deliberate Control**: Precise pinch detection prevents accidents
- **Professional Feel**: Responsive, accurate gesture recognition
- **Optimized Layout**: Larger buttons, better spacing, clear visual feedback

### Advanced Audio Processing
- **Stem Isolation**: Automatic loading of vocal and instrumental tracks
- **Position Management**: Maintains playback position like professional software
- **Real-time Effects**: Instant vocal/instrumental toggling
- **High-Quality Audio**: Professional audio engine with low latency

## 🎵 Ready to DJ?

Launch Air DJ and experience the future of digital DJing:

```bash
source venv_py311/bin/activate
python air_dj.py
```

**Use pinch gestures to control the interface and create professional mixes with precision hand tracking!**

## 🙏 Acknowledgments

- **MediaPipe**: Google's hand tracking technology
- **Pyo**: Professional audio processing library  
- **OpenCV**: Computer vision framework
- **NumPy**: Numerical computing support

---

**🎧 Air DJ - Where hand gestures meet professional DJing! 🎛️**