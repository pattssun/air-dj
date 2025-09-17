#!/usr/bin/env python3
"""
Air DJ Controller - Webcam-controlled DJ interface
A modular DJ controller that overlays on screen and responds to hand gestures
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# Audio processing
try:
    from pyo import *
except ImportError:
    print("Pyo not available. Install with: pip install pyo")
    import pygame.mixer as fallback_audio

@dataclass
class ControllerButton:
    """Represents a clickable button on the DJ controller"""
    name: str
    x: int
    y: int
    width: int
    height: int
    color: Tuple[int, int, int] = (200, 200, 200)
    active_color: Tuple[int, int, int] = (100, 255, 100)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    is_active: bool = False
    is_pressed: bool = False
    button_type: str = "momentary"  # "momentary" or "toggle"

@dataclass
class JogWheel:
    """Represents a jog wheel control"""
    name: str
    center_x: int
    center_y: int
    radius: int
    current_angle: float = 0.0
    is_touching: bool = False

@dataclass
class Fader:
    """Represents a vertical fader control"""
    name: str
    x: int
    y: int
    width: int
    height: int
    value: float = 0.5  # 0.0 to 1.0
    is_dragging: bool = False

class DeckState(Enum):
    """States for each deck"""
    STOPPED = "stopped"
    PLAYING = "playing" 
    CUEING = "cueing"
    PAUSED = "paused"

@dataclass
class Track:
    """Represents a track with stems"""
    name: str
    folder_path: str
    bpm: int
    key: str
    stems: Dict[str, str]  # stem_type -> file_path

class AudioEngine:
    """Handles audio playback and mixing with professional track position management"""
    
    def __init__(self):
        self.server = None
        self.deck1_players = {}  # stem_type -> SfPlayer
        self.deck2_players = {}
        self.deck1_volumes = {}  # stem_type -> volume
        self.deck2_volumes = {}
        self.deck1_state = DeckState.STOPPED
        self.deck2_state = DeckState.STOPPED
        
        # Professional track position management
        self.deck1_cue_point = 0.0      # Cue point position (default: beginning)
        self.deck2_cue_point = 0.0
        self.deck1_play_position = 0.0  # Current playback position in seconds
        self.deck2_play_position = 0.0
        self.deck1_is_playing = False   # True playback state
        self.deck2_is_playing = False
        
        # Independent timing for each deck
        import time
        self.deck1_start_time = 0.0     # When deck 1 started playing
        self.deck2_start_time = 0.0     # When deck 2 started playing
        self.deck1_pause_time = 0.0     # Accumulated pause time for deck 1
        self.deck2_pause_time = 0.0     # Accumulated pause time for deck 2
        self.deck1_last_pause = 0.0     # When deck 1 was last paused
        self.deck2_last_pause = 0.0     # When deck 2 was last paused
        
        self.crossfader_value = 0.5
        # Track which stems are active - only vocal and instrumental for clear isolation
        self.deck1_active_stems = {"vocals": True, "instrumental": True}
        self.deck2_active_stems = {"vocals": True, "instrumental": True}
        
        # Track references for reloading during CUE operations
        self.deck1_track = None
        self.deck2_track = None
        
        self.setup_audio()
    
    def setup_audio(self):
        """Initialize the audio server"""
        try:
            self.server = Server().boot()
            self.server.start()
            print("Audio server initialized")
        except Exception as e:
            print(f"Error initializing audio: {e}")
    
    def load_track(self, deck: int, track: Track):
        """Load a track into the specified deck - only vocal and instrumental for clear isolation"""
        try:
            players = self.deck1_players if deck == 1 else self.deck2_players
            volumes = self.deck1_volumes if deck == 1 else self.deck2_volumes
            
            # Store track reference for CUE operations
            if deck == 1:
                self.deck1_track = track
            else:
                self.deck2_track = track
            
            # Clear existing players
            for player in players.values():
                player.stop()
            players.clear()
            volumes.clear()
            
            # Load only vocal and instrumental stems for clear isolation
            target_stems = ["vocals", "instrumental"]
            
            for stem_type in target_stems:
                if stem_type in track.stems:
                    file_path = track.stems[stem_type]
                    if os.path.exists(file_path):
                        # Create player and immediately start it (but at 0 volume initially)
                        player = SfPlayer(file_path, loop=True, mul=0.0)
                        player.out()  # Start playing immediately at 0 volume
                        players[stem_type] = player
                        volumes[stem_type] = 0.7  # Default volume
                        print(f"Loaded {stem_type} for deck {deck}")
                    else:
                        print(f"Warning: {stem_type} file not found for deck {deck}")
                else:
                    print(f"Warning: {stem_type} not available for this track on deck {deck}")
            
            # Reset position tracking and timing
            import time
            current_time = time.time()
            if deck == 1:
                self.deck1_play_position = 0.0
                self.deck1_start_time = current_time
                self.deck1_pause_time = 0.0
                self.deck1_last_pause = 0.0
            else:
                self.deck2_play_position = 0.0
                self.deck2_start_time = current_time
                self.deck2_pause_time = 0.0
                self.deck2_last_pause = 0.0
            
            # Verify we have both stems
            if len(players) < 2:
                print(f"Warning: Only {len(players)} stems loaded for deck {deck}. Isolation may not work as expected.")
            
            return True
        except Exception as e:
            print(f"Error loading track for deck {deck}: {e}")
            return False
    
    def play_deck(self, deck: int):
        """Start playing the specified deck from current play position (NOT cue point)"""
        import time
        players = self.deck1_players if deck == 1 else self.deck2_players
        volumes = self.deck1_volumes if deck == 1 else self.deck2_volumes
        active_stems = self.deck1_active_stems if deck == 1 else self.deck2_active_stems
        
        current_time = time.time()
        
        # Set playing state and update timing
        if deck == 1:
            self.deck1_is_playing = True
            self.deck1_state = DeckState.PLAYING
            # Set start time accounting for previous playback
            self.deck1_start_time = current_time - self.deck1_play_position
        else:
            self.deck2_is_playing = True
            self.deck2_state = DeckState.PLAYING
            # Set start time accounting for previous playback
            self.deck2_start_time = current_time - self.deck2_play_position
        
        # Resume playback by setting volume (players are already running)
        for stem_type, player in players.items():
            if active_stems.get(stem_type, True):
                # Set to full volume if stem is active
                player.mul = volumes.get(stem_type, 0.7)
            else:
                # Mute if stem is inactive
                player.mul = 0.0
        
        print(f"Deck {deck} playing from position {self.deck1_play_position if deck == 1 else self.deck2_play_position:.1f}s")
    
    def pause_deck(self, deck: int):
        """Pause the specified deck (maintains current play position)"""
        import time
        players = self.deck1_players if deck == 1 else self.deck2_players
        
        current_time = time.time()
        
        # Update play position before pausing
        if deck == 1:
            if self.deck1_is_playing:
                self.deck1_play_position = current_time - self.deck1_start_time
            self.deck1_is_playing = False
            self.deck1_state = DeckState.PAUSED
        else:
            if self.deck2_is_playing:
                self.deck2_play_position = current_time - self.deck2_start_time
            self.deck2_is_playing = False
            self.deck2_state = DeckState.PAUSED
        
        # Mute all players (but keep them running to maintain position)
        for player in players.values():
            player.mul = 0.0
        
        position = self.deck1_play_position if deck == 1 else self.deck2_play_position
        print(f"Deck {deck} paused at position {position:.1f}s")
    
    def cue_deck(self, deck: int):
        """CUE: Jump to cue point and play at low volume for preview (ONLY operation that seeks)"""
        import time
        players = self.deck1_players if deck == 1 else self.deck2_players
        volumes = self.deck1_volumes if deck == 1 else self.deck2_volumes
        active_stems = self.deck1_active_stems if deck == 1 else self.deck2_active_stems
        cue_point = self.deck1_cue_point if deck == 1 else self.deck2_cue_point
        
        current_time = time.time()
        
        # Stop and restart all players to reset position (ONLY operation that does this)
        for stem_type, player in players.items():
            player.stop()
            # Recreate player to reset position to beginning
            file_path = None
            if deck == 1 and self.deck1_track:
                file_path = self.deck1_track.stems.get(stem_type)
            elif deck == 2 and self.deck2_track:
                file_path = self.deck2_track.stems.get(stem_type)
            
            if file_path and os.path.exists(file_path):
                # Recreate player at beginning
                new_player = SfPlayer(file_path, loop=True, mul=0.0)
                new_player.out()
                players[stem_type] = new_player
        
        # Reset timing and set cue state
        if deck == 1:
            self.deck1_state = DeckState.CUEING
            self.deck1_play_position = 0.0
            self.deck1_start_time = current_time  # Reset timing to current
            self.deck1_is_playing = False
        else:
            self.deck2_state = DeckState.CUEING
            self.deck2_play_position = 0.0
            self.deck2_start_time = current_time  # Reset timing to current
            self.deck2_is_playing = False
        
        # Play at lower volume for cueing (only active stems)
        for stem_type, player in players.items():
            if active_stems.get(stem_type, True):
                # Reduce volume for cue preview
                original_vol = volumes.get(stem_type, 0.7)
                player.mul = original_vol * 0.3  # 30% volume for cue
            else:
                player.mul = 0.0
        
        print(f"Deck {deck} CUE: jumped to beginning (timing reset)")
    
    def stop_cue_deck(self, deck: int):
        """Stop cueing and return to stopped state (position remains at cue point)"""
        players = self.deck1_players if deck == 1 else self.deck2_players
        
        # Mute all players (keep them running at current position)
        for player in players.values():
            player.mul = 0.0
        
        # Set stopped state (position remains where cue left off)
        if deck == 1:
            self.deck1_state = DeckState.STOPPED
            self.deck1_is_playing = False
        else:
            self.deck2_state = DeckState.STOPPED
            self.deck2_is_playing = False
        
        print(f"Deck {deck} cue stopped (ready to play from current position)")
    
    def set_stem_volume(self, deck: int, stem_type: str, volume: float):
        """Set volume for a specific stem - applies at current position (NO position change)"""
        players = self.deck1_players if deck == 1 else self.deck2_players
        volumes = self.deck1_volumes if deck == 1 else self.deck2_volumes
        active_stems = self.deck1_active_stems if deck == 1 else self.deck2_active_stems
        is_playing = self.deck1_is_playing if deck == 1 else self.deck2_is_playing
        current_state = self.deck1_state if deck == 1 else self.deck2_state
        
        # Update active stem status
        active_stems[stem_type] = volume > 0.0
        volumes[stem_type] = volume
        
        if stem_type in players:
            # Apply volume change immediately (NO stop/start, just volume control)
            if is_playing:
                # Deck is playing - apply volume immediately
                players[stem_type].mul = volume
            elif current_state == DeckState.CUEING:
                # Deck is cueing - apply reduced volume immediately
                players[stem_type].mul = volume * 0.3
            else:
                # Deck is stopped/paused - set volume for next play
                players[stem_type].mul = 0.0  # Keep muted until play
        
        print(f"Deck {deck} {stem_type}: {'ON' if volume > 0 else 'OFF'} (volume control only)")
    
    def set_cue_point(self, deck: int, position: float = 0.0):
        """Set the cue point for a deck (default: beginning of track)"""
        if deck == 1:
            self.deck1_cue_point = position
        else:
            self.deck2_cue_point = position
        print(f"Deck {deck} cue point set to {position}")
    
    def get_deck_info(self, deck: int) -> dict:
        """Get current deck information for status display"""
        if deck == 1:
            return {
                "state": self.deck1_state.value,
                "is_playing": self.deck1_is_playing,
                "cue_point": self.deck1_cue_point,
                "play_position": self.deck1_play_position,
                "active_stems": self.deck1_active_stems.copy()
            }
        else:
            return {
                "state": self.deck2_state.value,
                "is_playing": self.deck2_is_playing,
                "cue_point": self.deck2_cue_point,
                "play_position": self.deck2_play_position,
                "active_stems": self.deck2_active_stems.copy()
            }
    
    def get_playback_position(self, deck: int) -> float:
        """Get current playback position as a percentage (0.0 to 1.0)"""
        import time
        current_time = time.time()
        track_length = 180.0  # Assuming 3-minute tracks
        
        if deck == 1:
            if self.deck1_is_playing:
                # Calculate current position based on when deck started and current time
                current_position = current_time - self.deck1_start_time
                # Loop the track if it goes beyond track length
                current_position = current_position % track_length
                return current_position / track_length
            else:
                # Return last known position when not playing
                return (self.deck1_play_position % track_length) / track_length
        else:
            if self.deck2_is_playing:
                # Calculate current position based on when deck started and current time  
                current_position = current_time - self.deck2_start_time
                # Loop the track if it goes beyond track length
                current_position = current_position % track_length
                return current_position / track_length
            else:
                # Return last known position when not playing
                return (self.deck2_play_position % track_length) / track_length
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.server:
            self.server.stop()

class HandTracker:
    """Handles hand tracking and pinch detection"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_width = 1280
        self.frame_height = 720
        self.pinch_history = []  # For smoothing pinch detection
        
    def detect_pinch(self, landmarks, hand_landmarks) -> Tuple[bool, Tuple[int, int]]:
        """
        Detect if thumb and index finger are pinched together
        Returns (is_pinched, pinch_position)
        """
        if not landmarks:
            return False, (0, 0)
        
        # Get thumb tip and index tip landmarks
        thumb_tip = landmarks.landmark[4]  # Thumb tip
        index_tip = landmarks.landmark[8]  # Index finger tip
        
        # Calculate distance between thumb and index finger
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + 
            (thumb_tip.y - index_tip.y) ** 2
        )
        
        # More precise pinch threshold - fingers must be closer together
        pinch_threshold = 0.04  # Reduced threshold for more precise pinch detection
        is_pinched = distance < pinch_threshold
        
        # Calculate pinch position (midpoint) - use actual frame dimensions
        pinch_x = int((thumb_tip.x + index_tip.x) * 0.5 * self.frame_width)
        pinch_y = int((thumb_tip.y + index_tip.y) * 0.5 * self.frame_height)
        
        return is_pinched, (pinch_x, pinch_y)
    
    def process_frame(self, frame):
        """
        Process frame and return hand tracking results
        Returns list of (is_pinched, pinch_position) for each detected hand
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        pinch_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                is_pinched, position = self.detect_pinch(hand_landmarks, hand_landmarks)
                pinch_data.append((is_pinched, position))
        
        return pinch_data, results

class TrackLoader:
    """Handles loading tracks from stem folders"""
    
    def __init__(self, songs_folder: str = "songs"):
        self.songs_folder = songs_folder
        self.available_tracks = []
        self.scan_tracks()
    
    def scan_tracks(self):
        """Scan the songs folder for tracks with stems"""
        self.available_tracks = []
        
        if not os.path.exists(self.songs_folder):
            print(f"Songs folder {self.songs_folder} not found")
            return
        
        # Sort folder names alphabetically for consistent track order
        folder_items = sorted(os.listdir(self.songs_folder))
        
        for item in folder_items:
            item_path = os.path.join(self.songs_folder, item)
            
            # Check if it's a folder (potential stem folder)
            if os.path.isdir(item_path):
                track = self._parse_stem_folder(item, item_path)
                if track:
                    self.available_tracks.append(track)
        
        print(f"Found {len(self.available_tracks)} tracks with stems")
    
    def _parse_stem_folder(self, folder_name: str, folder_path: str) -> Optional[Track]:
        """Parse a stem folder and create a Track object"""
        # Look for stem files
        stems = {}
        bpm = 120  # Default BPM
        key = "C"  # Default key
        
        # Expected stem types
        stem_types = ["vocals", "instrumental", "drums", "bass", "other"]
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.mp3', '.wav')):
                file_path = os.path.join(folder_path, file)
                
                # Try to identify stem type from filename
                file_lower = file.lower()
                for stem_type in stem_types:
                    if stem_type in file_lower:
                        stems[stem_type] = file_path
                        
                        # Try to extract BPM and key from filename
                        parts = file.split(' - ')
                        for part in parts:
                            if 'bpm' in part.lower():
                                try:
                                    bpm = int(''.join(filter(str.isdigit, part)))
                                except:
                                    pass
                            if any(note in part for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']) and ('maj' in part or 'min' in part):
                                key = part
                        break
        
        # Only create track if we have at least some stems
        if stems:
            return Track(
                name=folder_name,
                folder_path=folder_path,
                bpm=bpm,
                key=key,
                stems=stems
            )
        return None
    
    def get_track(self, index: int) -> Optional[Track]:
        """Get track by index"""
        if 0 <= index < len(self.available_tracks):
            return self.available_tracks[index]
        return None

class DJController:
    """Main DJ Controller class with transparent overlay"""
    
    def __init__(self):
        # Initialize components
        self.hand_tracker = HandTracker()
        self.audio_engine = AudioEngine()
        self.track_loader = TrackLoader()
        
        # Controller layout configuration
        self.screen_width = 1280
        self.screen_height = 720
        self.overlay_alpha = 0.8
        
        # Initialize controller elements
        self.setup_controller_layout()
        
        # State
        self.current_pinches = []
        self.deck1_track = None
        self.deck2_track = None
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)
    
    def setup_controller_layout(self):
        """Setup the DJ controller layout matching the screenshot"""
        # Center the layout - calculate positions relative to screen center
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Jog wheels - moved more toward center
        self.jog_wheel_1 = JogWheel("jog1", center_x - 320, center_y - 50, 120)
        self.jog_wheel_2 = JogWheel("jog2", center_x + 320, center_y - 50, 120)
        
        # Buttons for Deck 1 (left side) - centered with larger vocal/instrumental buttons
        self.deck1_buttons = {
            "cue": ControllerButton("Cue", center_x - 450, center_y + 160, 80, 40, button_type="momentary"),
            "play_pause": ControllerButton("Play/Pause", center_x - 450, center_y + 210, 80, 40, button_type="toggle"),
            "vocal": ControllerButton("Vocal", center_x - 360, center_y + 150, 120, 45, button_type="toggle"),
            "instrumental": ControllerButton("Instrumental", center_x - 360, center_y + 210, 120, 45, button_type="toggle")
        }
        
        # Buttons for Deck 2 (right side) - centered with larger vocal/instrumental buttons
        self.deck2_buttons = {
            "cue": ControllerButton("Cue", center_x + 370, center_y + 160, 80, 40, button_type="momentary"),
            "play_pause": ControllerButton("Play/Pause", center_x + 370, center_y + 210, 80, 40, button_type="toggle"),
            "vocal": ControllerButton("Vocal", center_x + 240, center_y + 150, 120, 45, button_type="toggle"),
            "instrumental": ControllerButton("Instrumental", center_x + 240, center_y + 210, 120, 45, button_type="toggle")
        }
        
        # Center controls - centered around middle
        self.center_buttons = {
            "hi_l": ControllerButton("Hi", center_x - 110, center_y - 150, 60, 30),
            "hi_r": ControllerButton("Hi", center_x + 50, center_y - 150, 60, 30),
            "mid_l": ControllerButton("Mid", center_x - 110, center_y - 110, 60, 30),
            "mid_r": ControllerButton("Mid", center_x + 50, center_y - 110, 60, 30),
            "low_l": ControllerButton("Low", center_x - 110, center_y - 70, 60, 30),
            "low_r": ControllerButton("Low", center_x + 50, center_y - 70, 60, 30),
            "cfx_l": ControllerButton("CFX", center_x - 110, center_y - 30, 60, 30),
            "cfx_r": ControllerButton("CFX", center_x + 50, center_y - 30, 60, 30)
        }
        
        # Volume faders - centered
        self.volume_fader_1 = Fader("Vol1", center_x - 70, center_y + 40, 30, 150)
        self.volume_fader_2 = Fader("Vol2", center_x + 40, center_y + 40, 30, 150)
        
        # Crossfader - centered
        self.crossfader = Fader("Crossfader", center_x - 100, center_y + 240, 200, 30, value=0.5)
        
        # Tempo controls - centered
        self.tempo_fader_1 = Fader("Tempo1", center_x - 280, center_y + 40, 20, 120)
        self.tempo_fader_2 = Fader("Tempo2", center_x + 260, center_y + 40, 20, 120)
    
    def handle_button_interaction(self, button: ControllerButton, deck: int = 0):
        """Handle button press interactions"""
        if button.name == "Cue":
            if deck == 1:
                self.audio_engine.cue_deck(1)
                button.is_active = True
            elif deck == 2:
                self.audio_engine.cue_deck(2)
                button.is_active = True
        
        elif button.name == "Play/Pause":
            if button.button_type == "toggle":
                button.is_active = not button.is_active
                
                if deck == 1:
                    if button.is_active:
                        self.audio_engine.play_deck(1)
                    else:
                        self.audio_engine.pause_deck(1)
                elif deck == 2:
                    if button.is_active:
                        self.audio_engine.play_deck(2)
                    else:
                        self.audio_engine.pause_deck(2)
        
        elif button.name == "Vocal":
            if button.button_type == "toggle":
                button.is_active = not button.is_active
                # Toggle vocal stem volume
                volume = 0.7 if button.is_active else 0.0
                self.audio_engine.set_stem_volume(deck, "vocals", volume)
                print(f"Deck {deck} vocals: {'ON' if button.is_active else 'OFF'}")
        
        elif button.name == "Instrumental":
            if button.button_type == "toggle":
                button.is_active = not button.is_active
                # Toggle instrumental stem volume
                volume = 0.7 if button.is_active else 0.0
                self.audio_engine.set_stem_volume(deck, "instrumental", volume)
                print(f"Deck {deck} instrumental: {'ON' if button.is_active else 'OFF'}")
    
    def handle_button_release(self, button: ControllerButton, deck: int = 0):
        """Handle button release interactions"""
        if button.name == "Cue" and button.button_type == "momentary":
            button.is_active = False
            if deck == 1:
                self.audio_engine.stop_cue_deck(1)
            elif deck == 2:
                self.audio_engine.stop_cue_deck(2)
    
    def check_button_collision(self, x: int, y: int, button: ControllerButton) -> bool:
        """Check if coordinates collide with button"""
        return (button.x <= x <= button.x + button.width and 
                button.y <= y <= button.y + button.height)
    
    def check_button_collision_expanded(self, x: int, y: int, button: ControllerButton) -> bool:
        """Check if coordinates collide with button using expanded hit area for better reliability"""
        # Smaller margin since buttons are now larger - more precise selection
        margin = 10
        return (button.x - margin <= x <= button.x + button.width + margin and 
                button.y - margin <= y <= button.y + button.height + margin)
    
    def draw_track_visualization(self, overlay, center_x: int, center_y: int):
        """Draw track visualization bars for both decks"""
        # Track visualization bars
        bar_width = 300
        bar_height = 20
        
        # Deck 1 track bar (left side)
        deck1_x = center_x - 350
        deck1_y = center_y - 200
        
        # Background bar
        cv2.rectangle(overlay, (deck1_x, deck1_y), (deck1_x + bar_width, deck1_y + bar_height), (40, 40, 40), -1)
        cv2.rectangle(overlay, (deck1_x, deck1_y), (deck1_x + bar_width, deck1_y + bar_height), (200, 200, 200), 2)
        
        # Position indicator
        position1 = self.audio_engine.get_playback_position(1)
        pos1_x = int(deck1_x + position1 * bar_width)
        
        # Progress bar
        if self.audio_engine.deck1_is_playing:
            cv2.rectangle(overlay, (deck1_x, deck1_y), (pos1_x, deck1_y + bar_height), (0, 200, 0), -1)
        else:
            cv2.rectangle(overlay, (deck1_x, deck1_y), (pos1_x, deck1_y + bar_height), (100, 100, 100), -1)
        
        # Position needle
        cv2.line(overlay, (pos1_x, deck1_y - 5), (pos1_x, deck1_y + bar_height + 5), (255, 255, 255), 2)
        
        # Cue point indicator (at beginning)
        cue_x1 = deck1_x
        cv2.line(overlay, (cue_x1, deck1_y - 10), (cue_x1, deck1_y + bar_height + 10), (255, 255, 0), 3)
        
        # Labels with timing info
        cv2.putText(overlay, "DECK 1 TRACK POSITION", (deck1_x, deck1_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate time in seconds for display
        time_seconds1 = position1 * 180.0  # Assuming 3-minute tracks
        minutes1 = int(time_seconds1 // 60)
        seconds1 = int(time_seconds1 % 60)
        cv2.putText(overlay, f"{position1*100:.1f}% ({minutes1}:{seconds1:02d})", (deck1_x, deck1_y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Deck 1 stem status
        vocal1_status = "🎤 VOCAL" if self.audio_engine.deck1_active_stems.get("vocals", False) else "🎤 vocal"
        inst1_status = "🎶 INST" if self.audio_engine.deck1_active_stems.get("instrumental", False) else "🎶 inst"
        cv2.putText(overlay, f"{vocal1_status} | {inst1_status}", (deck1_x, deck1_y + bar_height + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Deck 2 track bar (right side)
        deck2_x = center_x + 50
        deck2_y = center_y - 200
        
        # Background bar
        cv2.rectangle(overlay, (deck2_x, deck2_y), (deck2_x + bar_width, deck2_y + bar_height), (40, 40, 40), -1)
        cv2.rectangle(overlay, (deck2_x, deck2_y), (deck2_x + bar_width, deck2_y + bar_height), (200, 200, 200), 2)
        
        # Position indicator
        position2 = self.audio_engine.get_playback_position(2)
        pos2_x = int(deck2_x + position2 * bar_width)
        
        # Progress bar
        if self.audio_engine.deck2_is_playing:
            cv2.rectangle(overlay, (deck2_x, deck2_y), (pos2_x, deck2_y + bar_height), (0, 200, 0), -1)
        else:
            cv2.rectangle(overlay, (deck2_x, deck2_y), (pos2_x, deck2_y + bar_height), (100, 100, 100), -1)
        
        # Position needle
        cv2.line(overlay, (pos2_x, deck2_y - 5), (pos2_x, deck2_y + bar_height + 5), (255, 255, 255), 2)
        
        # Cue point indicator (at beginning)
        cue_x2 = deck2_x
        cv2.line(overlay, (cue_x2, deck2_y - 10), (cue_x2, deck2_y + bar_height + 10), (255, 255, 0), 3)
        
        # Labels with timing info
        cv2.putText(overlay, "DECK 2 TRACK POSITION", (deck2_x, deck2_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate time in seconds for display
        time_seconds2 = position2 * 180.0  # Assuming 3-minute tracks
        minutes2 = int(time_seconds2 // 60)
        seconds2 = int(time_seconds2 % 60)
        cv2.putText(overlay, f"{position2*100:.1f}% ({minutes2}:{seconds2:02d})", (deck2_x, deck2_y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Deck 2 stem status
        vocal2_status = "🎤 VOCAL" if self.audio_engine.deck2_active_stems.get("vocals", False) else "🎤 vocal"
        inst2_status = "🎶 INST" if self.audio_engine.deck2_active_stems.get("instrumental", False) else "🎶 inst"
        cv2.putText(overlay, f"{vocal2_status} | {inst2_status}", (deck2_x, deck2_y + bar_height + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_hand_interactions(self, pinch_data):
        """Process hand interactions with controller elements"""
        # Store previous button states
        prev_pressed_states = {}
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button_name, button in buttons.items():
                prev_pressed_states[id(button)] = button.is_pressed
        
        # Reset all pressed states
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button in buttons.values():
                button.is_pressed = False
        
        # Check for new pinch interactions with improved detection
        for is_pinched, (x, y) in pinch_data:
            if is_pinched:
                button_found = False
                
                # Check deck 1 buttons with expanded hit area
                for button in self.deck1_buttons.values():
                    if self.check_button_collision_expanded(x, y, button):
                        button.is_pressed = True
                        # Only trigger interaction if this is a new press
                        if not prev_pressed_states.get(id(button), False):
                            self.handle_button_interaction(button, 1)
                        button_found = True
                        break
                
                # Check deck 2 buttons with expanded hit area
                if not button_found:
                    for button in self.deck2_buttons.values():
                        if self.check_button_collision_expanded(x, y, button):
                            button.is_pressed = True
                            # Only trigger interaction if this is a new press
                            if not prev_pressed_states.get(id(button), False):
                                self.handle_button_interaction(button, 2)
                            button_found = True
                            break
        
        # Handle button releases for momentary buttons
        for buttons in [self.deck1_buttons, self.deck2_buttons]:
            deck = 1 if buttons == self.deck1_buttons else 2
            for button in buttons.values():
                # If button was pressed before but not now, it's a release
                if prev_pressed_states.get(id(button), False) and not button.is_pressed:
                    if button.button_type == "momentary":
                        self.handle_button_release(button, deck)
    
    def draw_controller_overlay(self, frame):
        """Draw the DJ controller overlay on the frame"""
        overlay = frame.copy()
        
        # Draw jog wheels with track visualization
        # Deck 1 jog wheel with position indicator
        cv2.circle(overlay, (self.jog_wheel_1.center_x, self.jog_wheel_1.center_y), 
                  self.jog_wheel_1.radius, (200, 200, 200), 3)
        
        # Add position indicator on jog wheel
        position1 = self.audio_engine.get_playback_position(1)
        angle1 = position1 * 2 * np.pi - np.pi/2  # Start from top
        pos_x1 = int(self.jog_wheel_1.center_x + (self.jog_wheel_1.radius - 20) * np.cos(angle1))
        pos_y1 = int(self.jog_wheel_1.center_y + (self.jog_wheel_1.radius - 20) * np.sin(angle1))
        cv2.circle(overlay, (pos_x1, pos_y1), 8, (0, 255, 0) if self.audio_engine.deck1_is_playing else (255, 255, 0), -1)
        cv2.putText(overlay, "DECK 1", (self.jog_wheel_1.center_x - 25, self.jog_wheel_1.center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Deck 2 jog wheel with position indicator
        cv2.circle(overlay, (self.jog_wheel_2.center_x, self.jog_wheel_2.center_y), 
                  self.jog_wheel_2.radius, (200, 200, 200), 3)
        
        # Add position indicator on jog wheel
        position2 = self.audio_engine.get_playback_position(2)
        angle2 = position2 * 2 * np.pi - np.pi/2  # Start from top
        pos_x2 = int(self.jog_wheel_2.center_x + (self.jog_wheel_2.radius - 20) * np.cos(angle2))
        pos_y2 = int(self.jog_wheel_2.center_y + (self.jog_wheel_2.radius - 20) * np.sin(angle2))
        cv2.circle(overlay, (pos_x2, pos_y2), 8, (0, 255, 0) if self.audio_engine.deck2_is_playing else (255, 255, 0), -1)
        cv2.putText(overlay, "DECK 2", (self.jog_wheel_2.center_x - 25, self.jog_wheel_2.center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw buttons
        for deck_buttons, deck_name in [(self.deck1_buttons, "Deck 1"), (self.deck2_buttons, "Deck 2")]:
            for button in deck_buttons.values():
                color = button.active_color if button.is_active else button.color
                if button.is_pressed:
                    color = (255, 255, 100)  # Highlight when pressed
                
                cv2.rectangle(overlay, (button.x, button.y), 
                            (button.x + button.width, button.y + button.height), color, -1)
                cv2.rectangle(overlay, (button.x, button.y), 
                            (button.x + button.width, button.y + button.height), (255, 255, 255), 2)
                
                # Button text
                text_x = button.x + 5
                text_y = button.y + button.height // 2 + 5
                cv2.putText(overlay, button.name, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, button.text_color, 1)
        
        # Draw center EQ controls using actual button positions
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        for button_name, button in self.center_buttons.items():
            color = button.active_color if button.is_active else (200, 200, 200)
            cv2.circle(overlay, (button.x + button.width//2, button.y + button.height//2), 20, color, -1)
            cv2.circle(overlay, (button.x + button.width//2, button.y + button.height//2), 20, (255, 255, 255), 2)
            cv2.putText(overlay, button.name, (button.x - 10, button.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw track visualization bars
        self.draw_track_visualization(overlay, center_x, center_y)
        
        # Draw volume section - centered
        vol_rect = (center_x - 85, center_y + 20, 170, 180)
        cv2.rectangle(overlay, (vol_rect[0], vol_rect[1]), 
                     (vol_rect[0] + vol_rect[2], vol_rect[1] + vol_rect[3]), (50, 50, 50), -1)
        cv2.rectangle(overlay, (vol_rect[0], vol_rect[1]), 
                     (vol_rect[0] + vol_rect[2], vol_rect[1] + vol_rect[3]), (255, 255, 255), 2)
        cv2.putText(overlay, "Volume", (center_x - 25, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Volume faders
        for fader in [self.volume_fader_1, self.volume_fader_2]:
            cv2.rectangle(overlay, (fader.x, fader.y), 
                         (fader.x + fader.width, fader.y + fader.height), (100, 100, 100), -1)
            
            # Fader handle
            handle_y = int(fader.y + fader.height * (1 - fader.value))
            cv2.rectangle(overlay, (fader.x - 5, handle_y - 10), 
                         (fader.x + fader.width + 5, handle_y + 10), (255, 255, 255), -1)
        
        # Crossfader - using actual fader position
        cf_rect = (self.crossfader.x, self.crossfader.y, self.crossfader.width, self.crossfader.height)
        cv2.rectangle(overlay, (cf_rect[0], cf_rect[1]), 
                     (cf_rect[0] + cf_rect[2], cf_rect[1] + cf_rect[3]), (100, 100, 100), -1)
        cv2.putText(overlay, "Crossfader", (cf_rect[0] + 50, cf_rect[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Crossfader handle
        handle_x = int(cf_rect[0] + cf_rect[2] * self.crossfader.value)
        cv2.rectangle(overlay, (handle_x - 10, cf_rect[1] - 5), 
                     (handle_x + 10, cf_rect[1] + cf_rect[3] + 5), (255, 255, 255), -1)
        
        # Tempo controls
        for fader, side in [(self.tempo_fader_1, "left"), (self.tempo_fader_2, "right")]:
            cv2.rectangle(overlay, (fader.x, fader.y), 
                         (fader.x + fader.width, fader.y + fader.height), (100, 100, 100), -1)
            cv2.putText(overlay, "Tempo", (fader.x - 20, fader.y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Tempo handle
            handle_y = int(fader.y + fader.height * (1 - fader.value))
            cv2.rectangle(overlay, (fader.x - 3, handle_y - 8), 
                         (fader.x + fader.width + 3, handle_y + 8), (255, 255, 255), -1)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)
        
        return frame
    
    def load_default_tracks(self):
        """Load default tracks into both decks with professional setup"""
        if len(self.track_loader.available_tracks) >= 1:
            track1 = self.track_loader.get_track(0)
            if track1:
                self.deck1_track = track1
                self.audio_engine.load_track(1, track1)
                # Set cue point at beginning (professional default)
                self.audio_engine.set_cue_point(1, 0.0)
                # Set default buttons active for deck 1 - both vocal and instrumental ON for full track
                self.deck1_buttons["vocal"].is_active = True
                self.deck1_buttons["instrumental"].is_active = True
                # Ensure audio engine reflects these settings (but start muted)
                self.audio_engine.deck1_active_stems["vocals"] = True
                self.audio_engine.deck1_active_stems["instrumental"] = True
                print(f"Loaded '{track1.name}' into Deck 1 (cue point: beginning)")
        
        if len(self.track_loader.available_tracks) >= 2:
            track2 = self.track_loader.get_track(1)
            if track2:
                self.deck2_track = track2
                self.audio_engine.load_track(2, track2)
                # Set cue point at beginning (professional default)
                self.audio_engine.set_cue_point(2, 0.0)
                # Set default buttons active for deck 2 - both vocal and instrumental ON for full track
                self.deck2_buttons["vocal"].is_active = True
                self.deck2_buttons["instrumental"].is_active = True
                # Ensure audio engine reflects these settings (but start muted)
                self.audio_engine.deck2_active_stems["vocals"] = True
                self.audio_engine.deck2_active_stems["instrumental"] = True
                print(f"Loaded '{track2.name}' into Deck 2 (cue point: beginning)")
    
    def run(self):
        """Main application loop"""
        print("DJ Controller starting...")
        print("Use pinch gestures (thumb + index finger) to interact with controls")
        
        # Load default tracks
        self.load_default_tracks()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Update hand tracker frame dimensions
                self.hand_tracker.frame_width = frame.shape[1]
                self.hand_tracker.frame_height = frame.shape[0]
                
                # Process hand tracking
                pinch_data, hand_results = self.hand_tracker.process_frame(frame)
                
                # Draw hand landmarks
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.hand_tracker.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.hand_tracker.mp_hands.HAND_CONNECTIONS)
                
                # Process interactions
                self.process_hand_interactions(pinch_data)
                
                # Draw controller overlay
                frame = self.draw_controller_overlay(frame)
                
                # Show pinch points and button detection
                for is_pinched, (x, y) in pinch_data:
                    if is_pinched:
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                        cv2.putText(frame, "PINCH", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Show which button is being targeted (if any)
                        for deck_buttons, deck_name in [(self.deck1_buttons, "D1"), (self.deck2_buttons, "D2")]:
                            for button_name, button in deck_buttons.items():
                                if self.check_button_collision_expanded(x, y, button):
                                    cv2.putText(frame, f"{deck_name}-{button_name}", (x + 15, y + 20), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Professional status info with deck management details
                status_y = 30
                deck1_info = self.audio_engine.get_deck_info(1)
                deck2_info = self.audio_engine.get_deck_info(2)
                
                # Deck 1 status with professional info
                deck1_status = f"Deck 1: {deck1_info['state']}"
                if deck1_info['is_playing']:
                    deck1_status += " (LIVE)"
                cv2.putText(frame, deck1_status, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Deck 2 status with professional info  
                deck2_status = f"Deck 2: {deck2_info['state']}"
                if deck2_info['is_playing']:
                    deck2_status += " (LIVE)"
                cv2.putText(frame, deck2_status, (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Track names
                if self.deck1_track:
                    cv2.putText(frame, f"Track 1: {self.deck1_track.name[:30]}", 
                               (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if self.deck2_track:
                    cv2.putText(frame, f"Track 2: {self.deck2_track.name[:30]}", 
                               (10, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Professional DJ behavior info
                cv2.putText(frame, "CUE: Jumps to beginning | PLAY/PAUSE: Volume control (no restart)", 
                           (10, status_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "VOCAL/INST: Real-time volume toggle (position maintained)", 
                           (10, status_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "🎛️ INDEPENDENT TIMING: Each deck has its own timeline", 
                           (10, status_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                cv2.putText(frame, "🤏 PRECISE PINCH: Closer fingers = more accurate selection", 
                           (10, status_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                cv2.putText(frame, "🎛️ LARGER BUTTONS: Vocal/Instrumental buttons bigger and separated", 
                           (10, status_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
                cv2.putText(frame, "Press 'q' to quit", (10, self.screen_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Air DJ Controller', frame)
                
                # Handle key presses
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio_engine.cleanup()

if __name__ == "__main__":
    controller = DJController()
    controller.run()
