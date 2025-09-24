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
- Python 3.8+
- Webcam
- Audio files with stems (see installation step 4 below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/air-dj.git
   cd air-dj
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv_py311\Scripts\activate
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
Getting Stems:
- Use AI stem separation tools (https://fadr.com/stems)
- Download pre-separated stems from music platforms
- The included songs folder has example stems ready to use

### Launch Air DJ

```bash
source venv/bin/activate
python air_dj.py
```