# 🎧 Air DJ

Control a DJ interface using hand gestures through your webcam.

## Key Features

### **Controls**

| Control | Function | Behavior |
|---------|----------|----------|
| **🎯 CUE** | Jump to cue point | Seeks to beginning for preview |
| **▶️ PLAY** | Start playback | Continues from current position |
| **⏸️ PAUSE** | Pause playback | Maintains position for resume |
| **🎤 VOCAL** | Toggle vocals | Real-time on/off at current position |
| **🎶 INSTRUMENTAL** | Toggle instrumental | Real-time on/off at current position |

### **Visual Feedback**
- **Track Progress Bars**: Real-time position with time display (mm:ss)
- **Jog Wheel Indicators**: Moving position markers
- **Stem Status Display**: Visual confirmation of vocal/instrumental state
- **Independent Visualization**: Each deck shows separate progress

## Quick Start

### Prerequisites
- **Python 3.9 - 3.12** (MediaPipe compatibility requirement)
- **64-bit Operating System** (Windows/macOS/Linux)
- **Webcam** (built-in or external)
- **Audio files with stems** (see "Add Your Music" section below)

⚠️ **Important**: Python 3.13+ and Python 3.8 are not supported due to MediaPipe compatibility.

### Installation

#### **macOS / Linux**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/air-dj.git
   cd air-dj
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### **Windows**

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/yourusername/air-dj.git
   cd air-dj
   ```

2. **Create virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

### **Launch Air DJ**

#### **macOS / Linux**
```bash
source venv/bin/activate
python air_dj.py
```

#### **Windows**
```cmd
venv\Scripts\activate
python air_dj.py
```

## **Song Selection**

### **Add Your Music**

Air DJ works with stem-separated audio files (vocals + instrumental tracks).

**[Complete Music Setup Guide →](songs/MUSIC_SETUP.md)**

**Quick Start:**
1. **Try it immediately**: Run `python air_dj.py` - includes 2 copyright-free example songs!
2. **Add your music**: Get stem-separated songs from [fadr.com](https://fadr.com/stems) (free AI separation)
3. Create folders in `songs/` directory: `Artist - Song Name/`
4. Add required files: `Vocals - Song.mp3` and `Instrumental - Song.mp3`

### **Interactive Selection (Default)**
```bash
python air_dj.py
```
Choose your songs from a numbered menu. **BPM sync enabled by default.**

### **Other Options**
```bash
python air_dj.py --default    # Skip song selection, use preset songs
python air_dj.py --unsync     # Disable BPM synchronization between decks
```

## **Licensing**
Air DJ is available under **[MIT License](LICENSE)**