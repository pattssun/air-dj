# Air DJ — agent reference

Read this first when working in this repo. It captures dev-environment gotchas,
architecture, and conventions so the next session doesn't relearn them.

## Dev environment

- **Python 3.9–3.12 only.** MediaPipe has no 3.13 wheels; do not bump the venv.
- The working venv on this machine is `venv_py311/`. Activate with
  `source venv_py311/bin/activate` (do not delete it; recreating is non-trivial).
- macOS Apple Silicon needs native libs for `pyo`:
  `brew install portaudio portmidi libsndfile liblo`. If `pip install pyo` still
  fails to find headers, export `CFLAGS=-I/opt/homebrew/include` and
  `LDFLAGS=-L/opt/homebrew/lib` and reinstall with `--no-cache-dir`.
- `requirements.txt` uses unified `tensorflow` (since TF 2.16). The legacy
  `tensorflow-macos` package is deprecated — do not reintroduce it.

## Run / debug

- Default: `python air_dj.py` (interactive song picker, BPM sync on).
- Skip picker: `python air_dj.py --default`.
- Disable BPM sync: `python air_dj.py --unsync`.
- In-app keys: `/` toggles keymap overlay, `f` FPS debug, `` ` `` anim status, `Esc` quit.
- Lint / format: `make lint` / `make format` (ruff).

## Architecture (default flow only)

Entry: `air_dj.py` → `DJController.run()`.

All core classes live in **`dj_controller.py`** (~4100 lines — known god-file,
refactor deferred):

| Class | Line | Role |
|---|---|---|
| `AudioEngine` | 696 | pyo-based stem playback, crossfader, tempo, BPM, scratch |
| `HandTracker` | 1550 | MediaPipe pinch detection (thumb+index, middle+index) |
| `TrackLoader` | 1645 | scans `songs/`, parses BPM from folder name |
| `RekordboxStyleVisualizer` | 256 | OpenCV waveform rendering |
| `DJController` | 1787 | orchestrator: routes gestures + keys → AudioEngine |

Camera capture lives in `iphone_camera_integration.py` (used by
`dj_controller.py` ~1861).

### Inactive files

`demo.py`, `hand_tracking_dj.py`, `color_tracking_dj.py`, `audio_converter.py`
are **not in the default flow**. `demo.py` is the only entry that uses them.
README only mentions `air_dj.py`. They are kept pending a user decision; do not
assume any code in them is current. If you add a new feature, add it to the
`DJController` flow, not these files.

## AudioEngine public API

All input sources (gestures, keyboard, future MIDI) should call these — do not
poke `AudioEngine` internals.

| Method | Line | Purpose |
|---|---|---|
| `play_deck(deck)` / `pause_deck(deck)` | 974 / 1038 | start/pause stem playback |
| `cue_deck(deck)` / `stop_cue_deck(deck)` | 1074 / 1085 | seek to cue point / release |
| `set_stem_volume(deck, stem, vol)` | 1103 | toggle/level vocals or instrumental |
| `set_master_volume(deck, vol)` | 1123 | per-deck master volume (0–1) |
| `set_crossfader_position(pos)` | 1136 | 0=deck1, 0.5=mix, 1=deck2 |
| `set_tempo(deck, fader_value)` | 1179 | 0–1 fader value → 0.8–1.2x speed |
| `nudge_track_position(deck, seconds)` | 1441 | jog/seek by ± seconds |
| `set_cue_point(deck, position)` | 1514 | place cue marker |
| `get_deck_info(deck)` | 1522 | dict with current position, bpm, etc. |
| `get_current_bpm(deck)` | 1216 | original BPM × tempo |

## Input → action mapping

Two adapters over the same AudioEngine API:

- **Gestures:** `DJController.process_hand_interactions` at `dj_controller.py:2292`.
  MediaPipe pinches → `handle_button_interaction` → AudioEngine.
- **Keyboard:** `DJController._handle_keyboard` at `dj_controller.py:2292` (just
  above gestures). Mirrored layout — left-hand keys drive Deck 1, right-hand
  keys drive Deck 2. Reuses `handle_button_interaction` so on-screen button
  highlighting stays consistent between input modes.

Keymap (also rendered in-app by `_draw_keymap_overlay`):

```
Action               Deck 1     Deck 2
Cue                  Q          P
Play/Pause           A          ;
Vocal toggle         W          O
Instrumental toggle  E          I
Tempo + / -          R / G      U / J
Seek - / + (back/fwd) S / D     K / L
Master vol + / -     1 / 2      9 / 0
Set cue point        Shift+Q    Shift+P

Crossfader  ← / → / ↓ (center)
Help        /   Quit  Esc   FPS  f   Anim  `
```

## Songs / stems

- Folder format: `songs/Artist - Track (BPM bpm)/` containing
  `Vocals - X.mp3` and `Instrumental - X.mp3`.
- BPM is parsed from the folder name (`(150 bpm)` or `[157bpm]`); falls back to
  librosa beat detection — see `TrackLoader._extract_bpm_from_folder_name`
  / `_detect_bpm_from_audio` (`dj_controller.py:1672` / `:1689`).

## Conventions

- **Adapter pattern:** new control = AudioEngine method first, *then* gesture
  mapping, *then* keyboard mapping. Don't duplicate logic across input layers.
- **No global mode flags.** Keyboard and hands coexist; do not gate one on the
  other.
- **No new windows.** UI overlays append to `draw_controller_overlay`; the
  keymap overlay (`_draw_keymap_overlay`) is the template.
- **Comments:** explain *why* (non-obvious constraints, key-code platform
  quirks, BPM math). Don't narrate *what* — names already do that.

## Common pitfalls

- `pyo` build failure on macOS → missing brew libs (see Dev environment).
- `tensorflow-macos` not found → unified `tensorflow` is now in requirements;
  pull latest, recreate venv if needed.
- MediaPipe import error → you're on Python 3.13. Recreate venv with 3.11/3.12.
- `cv2.waitKey(1) & 0xFF` masks arrow keys. The run loop now uses
  `cv2.waitKeyEx(1)` — do not revert.
- macOS and Linux report different arrow-key codes (63234 vs 65361 etc.);
  `_handle_keyboard` accepts both.

## Deferred work (do not start unprompted)

- Split `dj_controller.py` into an `airdj/` package
  (`audio_engine.py`, `hand_tracker.py`, `track_loader.py`, `visualizer.py`,
  `controller.py`, `keyboard.py`, `models.py`).
- Decide fate of the `demo.py` chain (likely deletable).
- Add pytest smoke tests for `AudioEngine` and `TrackLoader`.
