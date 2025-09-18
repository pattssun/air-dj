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
        
        # Professional volume control - master volume for each deck
        self.deck1_master_volume = 0.8  # 0.0 to 1.0 (default 80%)
        self.deck2_master_volume = 0.8  # 0.0 to 1.0 (default 80%)
        
        
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
        
        # Crossfader mixing - professional DJ crossfader behavior
        self.crossfader_position = 0.5  # 0.0 = full left (deck1), 1.0 = full right (deck2)
        
        # Track which stems are active - only vocal and instrumental for clear isolation
        self.deck1_active_stems = {"vocals": True, "instrumental": True}
        self.deck2_active_stems = {"vocals": True, "instrumental": True}
        
        # Track references for reloading during CUE operations
        self.deck1_track = None
        self.deck2_track = None
        
        self.setup_audio()
    
    def setup_audio(self, preferred_device=None):
        """Initialize the audio server for Mac with device selection"""
        try:
            # Get available devices first
            devices = []
            device_info = {}
            try:
                from pyo import pa_list_devices, pa_get_output_devices
                print("🔍 Available audio devices:")
                pa_list_devices()
                
                # Get output devices specifically
                output_devices = pa_get_output_devices()
                for device in output_devices:
                    device_id = device[0]  # Device ID
                    device_name = device[1]  # Device name is at index 1
                    devices.append((device_id, device_name))
                    device_info[device_id] = device_name
                    
            except Exception as device_error:
                print(f"Device enumeration: {device_error}")
            
            # Let user choose device if multiple outputs available
            output_device = None
            if len(devices) > 1:
                print(f"\n🎧 Found {len(devices)} output devices:")
                for i, (dev_id, dev_name) in enumerate(devices):
                    marker = " ← BUILT-IN" if "MacBook" in dev_name and "Speakers" in dev_name else ""
                    marker = " ← AIRPODS" if "AirPods" in dev_name else marker
                    print(f"   {i}: {dev_name}{marker}")
                
                # Try to use MacBook speakers by default (most reliable)
                macbook_speakers = None
                for dev_id, dev_name in devices:
                    if "MacBook" in dev_name and "Speakers" in dev_name:
                        macbook_speakers = dev_id
                        break
                
                if macbook_speakers is not None:
                    output_device = macbook_speakers
                    print(f"🔊 Auto-selected: {device_info[macbook_speakers]} (built-in speakers)")
                else:
                    # Check for AirPods and warn user
                    airpods_device = None
                    for dev_id, dev_name in devices:
                        if "AirPods" in dev_name:
                            airpods_device = dev_id
                            break
                    
                    if airpods_device is not None:
                        print(f"⚠️  AUDIO ROUTING TO AIRPODS: {device_info[airpods_device]}")
                        print("💡 If you can't hear audio:")
                        print("   • Check your AirPods are connected")
                        print("   • Or disconnect AirPods to use MacBook speakers")
                        output_device = airpods_device
                    else:
                        # Use first device
                        output_device = devices[0][0]
                        print(f"🔊 Using: {device_info[output_device]}")
                    
            # Configure server with specific device
            if output_device is not None:
                self.server = Server(
                    sr=44100,
                    nchnls=2,
                    buffersize=512,
                    duplex=0,
                    audio='portaudio',
                    jackname='',
                    ichnls=0,  # No input channels
                    )
                # Set output device after creation
                try:
                    self.server.setOutputDevice(output_device)
                    print(f"✅ Set output device to: {device_info[output_device]}")
                except:
                    print("⚠️  Could not set specific device, using default")
            else:
                # Default configuration
                self.server = Server(
                    sr=44100,
                    nchnls=2,
                    buffersize=512,
                    duplex=0,
                    audio='portaudio'
                )
            
            # Boot and start the server
            self.server.boot()
            self.server.start()
            print("✅ Audio server initialized")
            print(f"   Sample Rate: {self.server.getSamplingRate()}Hz")
            print(f"   Channels: {self.server.getNchnls()}")
            print(f"   Buffer Size: {self.server.getBufferSize()}")
            
        except Exception as e:
            print(f"❌ Error initializing audio: {e}")
            print("🔧 Trying basic audio configuration...")
            
            # Very basic fallback
            try:
                self.server = Server(sr=44100, nchnls=2)
                self.server.boot()
                self.server.start()
                print("✅ Basic audio server started")
            except Exception as fallback_error:
                print(f"❌ All audio configurations failed: {fallback_error}")
                self.server = None
    
    
    
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
                        player.out()  # Direct output - no EQ chain
                        
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
        
        # Resume playback by setting volume (players are already running) with master volume
        self._update_all_stem_volumes(deck)
        
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
                new_player.out()  # Direct output - no EQ
                
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
        
        # Play at lower volume for cueing (only active stems) with master volume
        self._update_all_stem_volumes(deck)
        
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
            # Apply volume change immediately with master volume (NO stop/start, just volume control)
            master_volume = self.deck1_master_volume if deck == 1 else self.deck2_master_volume
            
            # Calculate final volume
            if is_playing:
                # Deck is playing - apply volume with master volume
                final_volume = volume * master_volume if volume > 0 else 0.0
            elif current_state == DeckState.CUEING:
                # Deck is cueing - apply reduced volume with master volume
                final_volume = volume * master_volume * 0.3 if volume > 0 else 0.0
            else:
                # Deck is stopped/paused - set volume for next play
                final_volume = 0.0
            
            # Apply volume directly to player
            players[stem_type].mul = final_volume
        
        print(f"Deck {deck} {stem_type}: {'ON' if volume > 0 else 'OFF'} (volume control only)")
    
    def set_master_volume(self, deck: int, volume: float):
        """Set the master volume for a deck (0.0 to 1.0) - like Rekordbox volume fader"""
        volume = max(0.0, min(1.0, volume))  # Clamp between 0.0 and 1.0
        
        if deck == 1:
            self.deck1_master_volume = volume
        else:
            self.deck2_master_volume = volume
        
        # Update all currently playing stems with new master volume
        self._update_all_stem_volumes(deck)
        print(f"Deck {deck} master volume: {volume*100:.0f}%")
    
    def set_crossfader_position(self, position: float):
        """Set crossfader position - professional DJ crossfader mixing"""
        position = max(0.0, min(1.0, position))  # Clamp to valid range
        self.crossfader_position = position
        
        # Update both decks to apply crossfader mixing
        self._update_all_stem_volumes(1)
        self._update_all_stem_volumes(2)
        
        # Optional: print crossfader position for debugging
        if position <= 0.1:
            position_text = "DECK1"
        elif position >= 0.9:
            position_text = "DECK2"
        else:
            position_text = f"MIX-{int(position*100)}%"
        print(f"Crossfader: {position_text}")
    
    def _calculate_crossfader_gain(self, deck: int) -> float:
        """Calculate crossfader gain for a deck using professional power curve"""
        # Professional DJ crossfader uses a power curve for smooth mixing
        # 0.0 = full left (deck1), 1.0 = full right (deck2)
        
        if deck == 1:
            # Deck 1: Full volume at position 0.0, silent at position 1.0
            # Use power curve for smooth transition
            gain = (1.0 - self.crossfader_position) ** 0.5
        elif deck == 2:
            # Deck 2: Silent at position 0.0, full volume at position 1.0  
            # Use power curve for smooth transition
            gain = self.crossfader_position ** 0.5
        else:
            gain = 0.0
        
        return gain
    
    def _update_all_stem_volumes(self, deck: int):
        """Update all stem volumes for a deck using current master volume"""
        players = self.deck1_players if deck == 1 else self.deck2_players
        volumes = self.deck1_volumes if deck == 1 else self.deck2_volumes
        active_stems = self.deck1_active_stems if deck == 1 else self.deck2_active_stems
        is_playing = self.deck1_is_playing if deck == 1 else self.deck2_is_playing
        current_state = self.deck1_state if deck == 1 else self.deck2_state
        master_volume = self.deck1_master_volume if deck == 1 else self.deck2_master_volume
        
        for stem_type, player in players.items():
            stem_volume = volumes.get(stem_type, 0.7)
            stem_active = active_stems.get(stem_type, True)
            
            # Calculate final volume
            if is_playing and stem_active:
                # Apply master volume to active stem
                final_volume = stem_volume * master_volume
            elif current_state == DeckState.CUEING and stem_active:
                # Apply master volume to cue preview (reduced)
                final_volume = stem_volume * master_volume * 0.3
            else:
                # Muted
                final_volume = 0.0
            
            # Apply crossfader gain - professional DJ mixing
            crossfader_gain = self._calculate_crossfader_gain(deck)
            final_volume *= crossfader_gain
            
            # Apply volume directly to player
            player.mul = final_volume
    
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
        
        # Center controls (effects, etc.)
        self.center_buttons = {
            "cfx_l": ControllerButton("CFX", center_x - 110, center_y - 30, 60, 30),
            "cfx_r": ControllerButton("CFX", center_x + 50, center_y - 30, 60, 30)
        }
        
        # Volume faders - centered (set initial values to match audio engine)
        self.volume_fader_1 = Fader("Vol1", center_x - 70, center_y + 40, 30, 150, value=0.8)  # Match deck1_master_volume
        self.volume_fader_2 = Fader("Vol2", center_x + 40, center_y + 40, 30, 150, value=0.8)   # Match deck2_master_volume
        
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
    
    def check_fader_collision(self, x: int, y: int, fader: Fader) -> bool:
        """Check if coordinates collide with fader (expanded area for easier interaction)"""
        margin = 15  # Larger margin for faders to make them easier to grab
        return (fader.x - margin <= x <= fader.x + fader.width + margin and 
                fader.y - margin <= y <= fader.y + fader.height + margin)
    
    def handle_fader_interaction(self, x: int, y: int, fader: Fader, deck: int = 0):
        """Handle fader drag interaction - convert Y position to fader value"""
        # Calculate relative position within fader (0.0 at bottom, 1.0 at top)
        relative_y = (y - fader.y) / fader.height
        # Invert since fader value 1.0 should be at top (lower Y value)
        fader_value = max(0.0, min(1.0, 1.0 - relative_y))
        
        # Update fader value
        fader.value = fader_value
        
        # Apply to audio engine based on fader type
        if fader.name == "Vol1":
            self.audio_engine.set_master_volume(1, fader_value)
        elif fader.name == "Vol2":
            self.audio_engine.set_master_volume(2, fader_value)
        
        print(f"Volume fader {deck}: {fader_value*100:.0f}%")
    
    def handle_crossfader_interaction(self, x: int, y: int):
        """Handle crossfader drag interaction - convert X position to crossfader value"""
        # Calculate relative position within crossfader (0.0 at left, 1.0 at right)
        relative_x = (x - self.crossfader.x) / self.crossfader.width
        crossfader_value = max(0.0, min(1.0, relative_x))
        
        # Update crossfader value
        self.crossfader.value = crossfader_value
        
        # Apply to audio engine
        self.audio_engine.set_crossfader_position(crossfader_value)
    
    def check_crossfader_collision(self, x: int, y: int) -> bool:
        """Check if coordinates collide with crossfader (expanded area for easier interaction)"""
        margin = 15  # Larger margin for easier interaction
        return (self.crossfader.x - margin <= x <= self.crossfader.x + self.crossfader.width + margin and 
                self.crossfader.y - margin <= y <= self.crossfader.y + self.crossfader.height + margin)
    
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
                interaction_found = False
                
                # Check volume faders and crossfader first (priority over buttons for mixing control)
                if self.check_fader_collision(x, y, self.volume_fader_1):
                    self.handle_fader_interaction(x, y, self.volume_fader_1, 1)
                    self.volume_fader_1.is_dragging = True
                    interaction_found = True
                elif self.check_fader_collision(x, y, self.volume_fader_2):
                    self.handle_fader_interaction(x, y, self.volume_fader_2, 2)
                    self.volume_fader_2.is_dragging = True
                    interaction_found = True
                elif self.check_crossfader_collision(x, y):
                    self.handle_crossfader_interaction(x, y)
                    self.crossfader.is_dragging = True
                    interaction_found = True
                
                # Check deck 1 buttons with expanded hit area (only if no fader interaction)
                if not interaction_found:
                    for button in self.deck1_buttons.values():
                        if self.check_button_collision_expanded(x, y, button):
                            button.is_pressed = True
                            # Only trigger interaction if this is a new press
                            if not prev_pressed_states.get(id(button), False):
                                self.handle_button_interaction(button, 1)
                            interaction_found = True
                            break
                
                # Check deck 2 buttons with expanded hit area (only if no other interaction)
                if not interaction_found:
                    for button in self.deck2_buttons.values():
                        if self.check_button_collision_expanded(x, y, button):
                            button.is_pressed = True
                            # Only trigger interaction if this is a new press
                            if not prev_pressed_states.get(id(button), False):
                                self.handle_button_interaction(button, 2)
                            interaction_found = True
                            break
                
        
        # Reset fader dragging states when no pinch is detected
        if not any(is_pinched for is_pinched, _ in pinch_data):
            self.volume_fader_1.is_dragging = False
            self.volume_fader_2.is_dragging = False
            self.crossfader.is_dragging = False
        
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
        
        # Draw center controls (effects, etc.)
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
        
        # Volume faders with professional-style visualization
        for i, fader in enumerate([self.volume_fader_1, self.volume_fader_2]):
            deck_num = i + 1
            
            # Fader track (background)
            cv2.rectangle(overlay, (fader.x, fader.y), 
                         (fader.x + fader.width, fader.y + fader.height), (60, 60, 60), -1)
            cv2.rectangle(overlay, (fader.x, fader.y), 
                         (fader.x + fader.width, fader.y + fader.height), (200, 200, 200), 2)
            
            # Volume level indicator (filled portion)
            fill_height = int(fader.height * fader.value)
            fill_y = fader.y + fader.height - fill_height
            if fill_height > 0:
                # Color based on volume level (green for normal, yellow for high, red for max)
                if fader.value < 0.7:
                    color = (0, 200, 0)  # Green
                elif fader.value < 0.9:
                    color = (0, 200, 200)  # Yellow
                else:
                    color = (0, 100, 255)  # Red (near max)
                
                cv2.rectangle(overlay, (fader.x + 2, fill_y), 
                             (fader.x + fader.width - 2, fader.y + fader.height - 2), color, -1)
            
            # Fader handle
            handle_y = int(fader.y + fader.height * (1 - fader.value))
            handle_color = (255, 255, 100) if fader.is_dragging else (255, 255, 255)
            cv2.rectangle(overlay, (fader.x - 8, handle_y - 12), 
                         (fader.x + fader.width + 8, handle_y + 12), handle_color, -1)
            cv2.rectangle(overlay, (fader.x - 8, handle_y - 12), 
                         (fader.x + fader.width + 8, handle_y + 12), (0, 0, 0), 2)
            
            # Volume percentage label
            vol_percent = int(fader.value * 100)
            label_x = fader.x + fader.width + 15
            label_y = handle_y + 5
            cv2.putText(overlay, f"{vol_percent}%", (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Deck label
            cv2.putText(overlay, f"VOL{deck_num}", (fader.x - 5, fader.y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Crossfader - Professional DJ controller style
        cf_rect = (self.crossfader.x, self.crossfader.y, self.crossfader.width, self.crossfader.height)
        
        # Crossfader track background
        cv2.rectangle(overlay, (cf_rect[0], cf_rect[1]), 
                     (cf_rect[0] + cf_rect[2], cf_rect[1] + cf_rect[3]), (60, 60, 60), -1)
        
        # Crossfader position indicators
        cf_pos = self.crossfader.value
        
        # Color coding for crossfader position
        if cf_pos <= 0.1:
            # Full Deck 1
            track_color = (0, 100, 255)  # Blue for Deck 1
            pos_text = "DECK 1"
        elif cf_pos >= 0.9:
            # Full Deck 2  
            track_color = (255, 100, 0)  # Orange for Deck 2
            pos_text = "DECK 2"
        else:
            # Mixed
            track_color = (150, 0, 255)  # Purple for mix
            pos_text = f"MIX {int(cf_pos*100)}%"
        
        # Draw crossfader track with position color
        cv2.rectangle(overlay, (cf_rect[0], cf_rect[1]), 
                     (cf_rect[0] + cf_rect[2], cf_rect[1] + cf_rect[3]), track_color, 2)
        
        # Crossfader labels
        cv2.putText(overlay, "A", (cf_rect[0] - 15, cf_rect[1] + cf_rect[3] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)  # Deck 1 label
        cv2.putText(overlay, "B", (cf_rect[0] + cf_rect[2] + 5, cf_rect[1] + cf_rect[3] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)  # Deck 2 label
        
        # Crossfader handle
        handle_x = int(cf_rect[0] + cf_rect[2] * cf_pos)
        handle_color = (255, 255, 255) if not self.crossfader.is_dragging else (255, 255, 0)
        cv2.rectangle(overlay, (handle_x - 8, cf_rect[1] - 3), 
                     (handle_x + 8, cf_rect[1] + cf_rect[3] + 3), handle_color, -1)
        
        # Center position indicator
        center_x = cf_rect[0] + cf_rect[2] // 2
        cv2.line(overlay, (center_x, cf_rect[1] - 5), (center_x, cf_rect[1] + cf_rect[3] + 5), 
                (200, 200, 200), 1)
        
        # Crossfader position text
        cv2.putText(overlay, pos_text, (cf_rect[0] + 30, cf_rect[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
        
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
                        
                        # Show which element is being targeted
                        target_text = ""
                        target_color = (0, 255, 255)
                        
                        # Check faders first
                        if self.check_fader_collision(x, y, self.volume_fader_1):
                            target_text = f"VOL1-{int(self.volume_fader_1.value*100)}%"
                            target_color = (0, 255, 0)
                        elif self.check_fader_collision(x, y, self.volume_fader_2):
                            target_text = f"VOL2-{int(self.volume_fader_2.value*100)}%"
                            target_color = (0, 255, 0)
                        elif self.check_crossfader_collision(x, y):
                            cf_pos = int(self.crossfader.value * 100)
                            if cf_pos <= 10:
                                cf_text = "DECK1"
                            elif cf_pos >= 90:
                                cf_text = "DECK2"
                            else:
                                cf_text = f"MIX-{cf_pos}%"
                            target_text = f"CROSSFADER-{cf_text}"
                            target_color = (255, 0, 255)  # Purple for crossfader
                        
                        
                        # Check buttons if no fader or EQ interaction
                        if not target_text:
                            for deck_buttons, deck_name in [(self.deck1_buttons, "D1"), (self.deck2_buttons, "D2")]:
                                for button_name, button in deck_buttons.items():
                                    if self.check_button_collision_expanded(x, y, button):
                                        target_text = f"{deck_name}-{button_name}"
                                        break
                        
                        if target_text:
                            cv2.putText(frame, target_text, (x + 15, y + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1)
                
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
                
                # Crossfader status
                cf_pos = self.audio_engine.crossfader_position
                if cf_pos <= 0.1:
                    cf_status = "Crossfader: DECK 1"
                    cf_color = (0, 100, 255)
                elif cf_pos >= 0.9:
                    cf_status = "Crossfader: DECK 2"
                    cf_color = (255, 100, 0)
                else:
                    cf_status = f"Crossfader: MIX {int(cf_pos*100)}%"
                    cf_color = (150, 0, 255)
                cv2.putText(frame, cf_status, (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cf_color, 2)
                
                # Track names
                if self.deck1_track:
                    cv2.putText(frame, f"Track 1: {self.deck1_track.name[:30]}", 
                               (10, status_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if self.deck2_track:
                    cv2.putText(frame, f"Track 2: {self.deck2_track.name[:30]}", 
                               (10, status_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Volume levels
                vol1_percent = int(self.audio_engine.deck1_master_volume * 100)
                vol2_percent = int(self.audio_engine.deck2_master_volume * 100)
                cv2.putText(frame, f"Volume 1: {vol1_percent}% | Volume 2: {vol2_percent}%", 
                           (10, status_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                
                # Professional DJ behavior info
                cv2.putText(frame, "CUE: Jumps to beginning | PLAY/PAUSE: Volume control (no restart)", 
                           (10, status_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "VOCAL/INST: Real-time volume toggle | VOLUME FADERS: Independent deck volume", 
                           (10, status_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "🎚️ PROFESSIONAL MIXING: Independent volume, timing, and stem control per deck", 
                           (10, status_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
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
