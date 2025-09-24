# 🎵 Music Setup Guide for Air DJ

Air DJ works with **stem-separated** audio files. You'll need vocals and instrumental tracks for each song.

## **Step 1: Get Stem-Separated Songs**
- **🤖 AI Separation (Recommended)**: Upload your songs to [fadr.com](https://fadr.com/stems) for free AI stem separation
- **🎵 Alternative Services**: [LALAL.AI](https://lalal.ai/), [Spleeter](https://github.com/deezer/spleeter), or [Ultimate Vocal Remover](https://ultimatevocalremover.com/)

## **Step 2: Organize Your Music**
Create folders in this `songs/` directory using this structure:
```
songs/
├── Artist - Song Name/
│   ├── Vocals - Artist - Song Name.mp3     ← Required
│   ├── Instrumental - Artist - Song Name.mp3  ← Required  
│   ├── Drums - Artist - Song Name.mp3     ← Optional
│   ├── Bass - Artist - Song Name.mp3      ← Optional
│   └── album-art.png                      ← Optional
└── Another Song/
    └── ...
```

## **Step 3: File Naming Requirements**
- **Vocals**: Must contain "Vocals" in filename
- **Instrumental**: Must contain "Instrumental" in filename  
- **Folder Name**: Use "Artist - Song Name" format
- **Supported Formats**: `.mp3`, `.wav`, `.flac`, `.m4a`

## **Example Setup**
```
songs/
├── Avicii - Levels/
│   ├── Vocals - Avicii - Levels.mp3
│   ├── Instrumental - Avicii - Levels.mp3
│   └── cover.png
└── Kesha - Die Young/
    ├── Vocals - Kesha - Die Young.mp3
    └── Instrumental - Kesha - Die Young.mp3
```

## **Tips for Best Results**

### **Getting High-Quality Stems:**
- Use **320kbps MP3** or **lossless formats** (WAV, FLAC) for source material
- **fadr.com** often provides the cleanest AI separation results
- For electronic music, try **Ultimate Vocal Remover** with different models

### **File Organization:**
- Keep consistent naming: `Vocals - [Artist] - [Song].mp3`
- Use album artwork (PNG/JPG) for visual appeal in the DJ interface
- Optional stems (Drums, Bass, Other) add creative control but aren't required

### **Troubleshooting:**
- **Song not appearing?** Check folder name format and required files (Vocals + Instrumental)
- **Poor audio quality?** Use higher bitrate source files for stem separation
- **Stems not isolating properly?** Some songs work better with different AI models

## **Ready to DJ!**

Once your music is set up, run:
```bash
python air_dj.py
```

Your songs will appear in the selection menu. Choose your tracks and start mixing with hand gestures!

---

💡 **Need help?** Check the main [README](../README.md) for installation and usage instructions.
