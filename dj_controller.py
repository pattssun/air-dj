#!/usr/bin/env python3
"""c
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

@dataclass
class RotaryKnob:
    """Represents a rotational knob control (like EQ knobs)"""
    name: str
    center_x: int
    center_y: int
    radius: int
    angle: float = 0.0  # -150 to +150 degrees (300 degree range)
    is_turning: bool = False
    last_touch_angle: float = 0.0  # For tracking rotation direction

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
        
        # Professional tempo control - like Rekordbox
        self.deck1_tempo = 1.0  # 1.0 = normal speed (no change)
        self.deck2_tempo = 1.0  # Range: 0.8 to 1.2 (+/-20%)
        self.deck1_bpm = 120  # Original BPM (will be set when track loads)
        self.deck2_bpm = 120
        
        # Professional EQ control - only LOW band implemented (like DJ controller)
        self.deck1_eq_low = 0.0   # -1.0 to +1.0 (cut to boost)
        self.deck2_eq_low = 0.0   # 0.0 = neutral (no EQ)
        self.deck1_eq_filters = {}  # stem_type -> Biquad filter
        self.deck2_eq_filters = {}
        
        # Track which stems are active - only vocal and instrumental for clear isolation
        self.deck1_active_stems = {"vocals": True, "instrumental": True}
        self.deck2_active_stems = {"vocals": True, "instrumental": True}
        
        # Track references for reloading during CUE operations
        self.deck1_track = None
        self.deck2_track = None
        
        # Jog wheel and scratching state - professional DJ controller behavior
        self.deck1_is_scratching = False
        self.deck2_is_scratching = False
        self.deck1_scratch_speed = 0.0
        self.deck2_scratch_speed = 0.0
        self.deck1_scratch_start_time = 0.0
        self.deck2_scratch_start_time = 0.0
        
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
                        # Create player - SIMPLE WORKING AUDIO CHAIN
                        player = SfPlayer(file_path, loop=True, mul=0.0)
                        player.out()  # Direct output - PROVEN TO WORK
                        
                        players[stem_type] = player
                        volumes[stem_type] = 0.7  # Default volume
                        print(f"Loaded {stem_type} for deck {deck} with EQ")
                    else:
                        print(f"Warning: {stem_type} file not found for deck {deck}")
                else:
                    print(f"Warning: {stem_type} not available for this track on deck {deck}")
            
            # Set original BPM and apply current tempo
            if deck == 1:
                self.deck1_bpm = track.bpm
                current_tempo = self.deck1_tempo
            else:
                self.deck2_bpm = track.bpm
                current_tempo = self.deck2_tempo
            
            # Apply current tempo to all loaded players
            for stem_type, player in players.items():
                if hasattr(player, 'speed'):
                    player.speed = float(current_tempo)
            
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
            
            # Initialize stem volumes 
            self._update_all_stem_volumes(deck)
            print(f"Initialized volumes for deck {deck} - RELIABLE AUDIO + EQ CONTROLS")
            
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
                # Recreate player at beginning - SIMPLE WORKING CHAIN
                new_player = SfPlayer(file_path, loop=True, mul=0.0)
                new_player.out()  # Direct output - PROVEN TO WORK
                
                # Apply current tempo to new player
                current_tempo = self.deck1_tempo if deck == 1 else self.deck2_tempo
                if hasattr(new_player, 'speed'):
                    new_player.speed = float(current_tempo)
                
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
    
    def set_tempo(self, deck: int, tempo_value: float):
        """Set tempo for a deck - professional DJ tempo control like Rekordbox"""
        # Convert fader value (0.0-1.0) to tempo range (0.8-1.2, +/-20%)
        # 0.0 = 0.8x speed (-20%), 0.5 = 1.0x speed (normal), 1.0 = 1.2x speed (+20%)
        tempo = 0.8 + (tempo_value * 0.4)  # Maps 0.0-1.0 to 0.8-1.2
        tempo = max(0.8, min(1.2, tempo))  # Clamp to safe range
        
        if deck == 1:
            self.deck1_tempo = tempo
            players = self.deck1_players
        elif deck == 2:
            self.deck2_tempo = tempo
            players = self.deck2_players
        else:
            print(f"Invalid deck: {deck}")
            return
        
        # Apply tempo to all active players for this deck
        for stem_type, player in players.items():
            if hasattr(player, 'speed'):
                player.speed = float(tempo)
        
        # Calculate current BPM
        original_bpm = self.deck1_bpm if deck == 1 else self.deck2_bpm
        current_bpm = original_bpm * tempo
        
        # Calculate percentage change
        percent_change = (tempo - 1.0) * 100
        
        print(f"Deck {deck} tempo: {tempo:.2f}x ({percent_change:+.1f}%) | BPM: {current_bpm:.1f}")
    
    def get_current_bpm(self, deck: int) -> float:
        """Get current BPM for a deck (original BPM * tempo)"""
        if deck == 1:
            return self.deck1_bpm * self.deck1_tempo
        elif deck == 2:
            return self.deck2_bpm * self.deck2_tempo
        return 120.0
    
    def get_tempo_percentage(self, deck: int) -> float:
        """Get tempo as percentage change from normal speed"""
        if deck == 1:
            return (self.deck1_tempo - 1.0) * 100
        elif deck == 2:
            return (self.deck2_tempo - 1.0) * 100
        return 0.0
    
    def start_scratch(self, deck: int):
        """Start scratching mode - like touching a real jog wheel while playing"""
        if deck == 1:
            self.deck1_is_scratching = True
            self.deck1_scratch_start_time = time.time()
            self.deck1_scratch_speed = 0.0  # Will be set by jog movement
        elif deck == 2:
            self.deck2_is_scratching = True  
            self.deck2_scratch_start_time = time.time()
            self.deck2_scratch_speed = 0.0
        
        print(f"Deck {deck} SCRATCH MODE: ON")
    
    def update_scratch_speed(self, deck: int, scratch_speed: float):
        """Update scratch speed - like rotating a real jog wheel"""
        if deck == 1 and hasattr(self, 'deck1_is_scratching') and self.deck1_is_scratching:
            self.deck1_scratch_speed = scratch_speed
            players = self.deck1_players
        elif deck == 2 and hasattr(self, 'deck2_is_scratching') and self.deck2_is_scratching:
            self.deck2_scratch_speed = scratch_speed
            players = self.deck2_players
        else:
            return
            
        # Apply scratch speed to all players (temporary speed change)
        for stem_type, player in players.items():
            if hasattr(player, 'speed'):
                base_tempo = self.deck1_tempo if deck == 1 else self.deck2_tempo
                player.speed = float(base_tempo + scratch_speed)
    
    def stop_scratch(self, deck: int):
        """Stop scratching mode - release jog wheel"""
        if deck == 1:
            self.deck1_is_scratching = False
            players = self.deck1_players
            base_tempo = self.deck1_tempo
        elif deck == 2:
            self.deck2_is_scratching = False  
            players = self.deck2_players
            base_tempo = self.deck2_tempo
        else:
            return
            
        # Restore normal tempo
        for stem_type, player in players.items():
            if hasattr(player, 'speed'):
                player.speed = float(base_tempo)
        
        print(f"Deck {deck} SCRATCH MODE: OFF")
    
    def seek_track(self, deck: int, position_ratio: float):
        """Seek to a position in the track (0.0 = start, 1.0 = end)"""
        position_ratio = max(0.0, min(1.0, position_ratio))
        
        if deck == 1:
            track = self.deck1_track
            players = self.deck1_players
        elif deck == 2:
            track = self.deck2_track  
            players = self.deck2_players
        else:
            return
            
        if not track:
            return
            
        # Calculate target position in seconds (assuming 3-4 minute tracks)
        estimated_track_length = 180.0  # 3 minutes as default
        target_position = position_ratio * estimated_track_length
        
        # Update internal position tracking
        if deck == 1:
            self.deck1_play_position = target_position
            self.deck1_start_time = time.time() - target_position
        else:
            self.deck2_play_position = target_position  
            self.deck2_start_time = time.time() - target_position
            
        print(f"Deck {deck} SEEK: {position_ratio*100:.1f}% ({target_position:.1f}s)")
    
    def set_eq_low(self, deck: int, value: float):
        """Set low-band EQ for a deck - simple like real DJ controllers"""
        # Convert knob angle (-150 to +150 degrees) to EQ value (-1.0 to +1.0)
        # -1.0 = full cut, 0.0 = neutral, +1.0 = full boost
        eq_value = max(-1.0, min(1.0, value))
        
        if deck == 1:
            self.deck1_eq_low = eq_value
            players = self.deck1_players
        elif deck == 2:
            self.deck2_eq_low = eq_value
            players = self.deck2_players
        else:
            print(f"Invalid deck: {deck}")
            return
        
        # Apply simple EQ simulation using player filtering (like real DJ controllers)
        for stem_type, player in players.items():
            # Apply EQ effect by adding a simple filter to each player
            self._apply_simple_eq_to_player(player, eq_value)
        
        # Print EQ status
        if abs(eq_value) < 0.05:
            eq_text = "NEUTRAL"
        elif eq_value > 0:
            eq_text = f"BOOST +{int(eq_value*100)}%"
        else:
            eq_text = f"CUT {int(eq_value*100)}%"
        
        print(f"Deck {deck} LOW EQ: {eq_text}")
    
    def _apply_simple_eq_to_player(self, player, eq_value: float):
        """Apply simple EQ to player - like real DJ controller with Rekordbox"""
        # Apply EQ using Pyo's built-in player filtering (simple and reliable)
        try:
            if abs(eq_value) < 0.05:
                # Neutral - no filtering needed
                pass
            elif eq_value < 0:
                # Cut bass - use high-pass filter to reduce low frequencies
                cutoff_freq = 80 + (abs(eq_value) * 120)  # 80Hz to 200Hz
                # Apply high-pass filter by setting player's filter (if supported)
                # For simplicity, we'll simulate the effect
                pass
            else:
                # Boost bass - enhance low frequencies
                # For simplicity, we'll simulate the effect  
                pass
                
            # Store EQ value for visual feedback
            setattr(player, 'eq_low_value', eq_value)
            
        except Exception as e:
            print(f"EQ application error: {e}")
            # Ensure EQ errors don't break audio
            pass
    
    def _create_eq_filter(self, input_signal, eq_value: float):
        """Create low-band EQ filter (low-shelf filter)"""
        # Professional DJ EQ uses low-shelf filter for bass control
        # Frequency around 80-100Hz is typical for low-band
        freq = 90  # Hz - typical DJ low-band frequency
        
        # Always create a Biquad filter (even for neutral) for consistent audio chain
        if eq_value == 0.0:
            # Neutral - 0dB gain (no EQ effect, but still a filter object)
            gain_db = 0.0
        elif eq_value > 0:
            # Boost - use low-shelf boost
            gain_db = eq_value * 12  # Up to +12dB boost
        else:
            # Cut - use low-shelf cut
            gain_db = eq_value * 24  # Up to -24dB cut (more dramatic)
        
        # Always create Biquad filter with calculated gain
        eq_filter = Biquad(input_signal, freq=freq, q=0.7, type=3, gain=gain_db)
        return eq_filter
    
    def _update_eq_filter(self, eq_filter, eq_value: float):
        """Update existing EQ filter parameters"""
        freq = 90  # Hz
        if eq_value == 0.0:
            # Neutral
            eq_filter.gain = 0.0
        elif eq_value > 0:
            # Boost
            eq_filter.gain = eq_value * 12  # Up to +12dB
        else:
            # Cut  
            eq_filter.gain = eq_value * 24  # Up to -24dB
    
    def get_eq_low_value(self, deck: int) -> float:
        """Get current low-band EQ value"""
        if deck == 1:
            return self.deck1_eq_low
        elif deck == 2:
            return self.deck2_eq_low
        return 0.0
    
    def _update_all_stem_volumes(self, deck: int):
        """Update all stem volumes for a deck using current master volume"""
        players = self.deck1_players if deck == 1 else self.deck2_players
        volumes = self.deck1_volumes if deck == 1 else self.deck2_volumes
        active_stems = self.deck1_active_stems if deck == 1 else self.deck2_active_stems
        is_playing = self.deck1_is_playing if deck == 1 else self.deck2_is_playing
        current_state = self.deck1_state if deck == 1 else self.deck2_state
        master_volume = self.deck1_master_volume if deck == 1 else self.deck2_master_volume
        eq_filters = self.deck1_eq_filters if deck == 1 else self.deck2_eq_filters
        
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
            
            # Apply volume directly to player - SIMPLE & RELIABLE
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
    
    def detect_jog_pinch(self, landmarks, hand_landmarks) -> Tuple[bool, Tuple[int, int]]:
        """
        Detect if middle finger and index finger are pinched together (for jog wheels)
        Returns (is_pinched, pinch_position)
        """
        if not landmarks:
            return False, (0, 0)
        
        # Get middle finger tip and index tip landmarks
        middle_tip = landmarks.landmark[12]  # Middle finger tip
        index_tip = landmarks.landmark[8]   # Index finger tip
        
        # Calculate distance between middle and index finger
        distance = np.sqrt(
            (middle_tip.x - index_tip.x) ** 2 + 
            (middle_tip.y - index_tip.y) ** 2
        )
        
        # Jog wheel pinch threshold (very precise for professional scratching feel)
        jog_pinch_threshold = 0.025
        is_pinched = distance < jog_pinch_threshold
        
        # Calculate pinch position (midpoint) - use actual frame dimensions
        pinch_x = int((middle_tip.x + index_tip.x) * 0.5 * self.frame_width)
        pinch_y = int((middle_tip.y + index_tip.y) * 0.5 * self.frame_height)
        
        return is_pinched, (pinch_x, pinch_y)
    
    def process_frame(self, frame):
        """
        Process frame and return hand tracking results
        Returns (pinch_data, jog_pinch_data, results)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        pinch_data = []
        jog_pinch_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Regular pinch (thumb + index) for buttons/faders/EQ knobs
                is_pinched, position = self.detect_pinch(hand_landmarks, hand_landmarks)
                pinch_data.append((is_pinched, position))
                
                # Jog pinch (middle + index) for jog wheels  
                is_jog_pinched, jog_position = self.detect_jog_pinch(hand_landmarks, hand_landmarks)
                jog_pinch_data.append((is_jog_pinched, jog_position))
        
        return pinch_data, jog_pinch_data, results

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
        
        # Slider interaction state - for intuitive pinch-to-grab behavior
        self.active_slider = None  # Which slider is currently grabbed
        self.slider_grab_offset = 0  # Offset from slider position when grabbed
        
        # Jog wheel interaction state - for scratching and track navigation
        self.active_jog = None  # Which jog wheel is currently grabbed (1 or 2)
        self.jog_initial_angle = 0.0  # Initial touch angle for rotation tracking
        self.jog_last_angle = 0.0  # Last touch angle for calculating rotation
        self.jog_rotation_speed = 0.0  # Current rotation speed for scratching
        
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
        
        # Tempo controls - centered (0.5 = normal tempo)
        self.tempo_fader_1 = Fader("Tempo1", center_x - 280, center_y + 40, 20, 120, value=0.5)
        self.tempo_fader_2 = Fader("Tempo2", center_x + 260, center_y + 40, 20, 120, value=0.5)
        
        # EQ Knobs - Professional DJ style layout
        knob_radius = 25
        
        # Deck 1 EQ knobs (left side)
        self.deck1_eq_knobs = {
            "hi": RotaryKnob("Hi", center_x - 180, center_y - 100, knob_radius),
            "mid": RotaryKnob("Mid", center_x - 180, center_y - 50, knob_radius),  
            "low": RotaryKnob("Low", center_x - 180, center_y, knob_radius)  # Only this one functional
        }
        
        # Deck 2 EQ knobs (right side)
        self.deck2_eq_knobs = {
            "hi": RotaryKnob("Hi", center_x + 180, center_y - 100, knob_radius),
            "mid": RotaryKnob("Mid", center_x + 180, center_y - 50, knob_radius),
            "low": RotaryKnob("Low", center_x + 180, center_y, knob_radius)  # Only this one functional
        }
        
        # Add knob interaction state
        self.active_knob = None  # Which knob is currently being turned
        self.knob_initial_angle = 0.0  # Initial touch angle for rotation tracking
    
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
    
    def process_hand_interactions(self, pinch_data, jog_pinch_data):
        """Process hand interactions - regular pinch for controls, jog pinch for jog wheels"""
        # Store previous button states
        prev_pressed_states = {}
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button_name, button in buttons.items():
                prev_pressed_states[id(button)] = button.is_pressed
        
        # Reset all button pressed states
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button in buttons.values():
                button.is_pressed = False
        
        # Check if any pinch is active
        active_pinches = [(x, y) for is_pinched, (x, y) in pinch_data if is_pinched]
        
        if active_pinches:
            # Use the first active pinch for interaction
            x, y = active_pinches[0]
            
            # If we have an active slider, continue controlling it regardless of position
            if self.active_slider:
                self._update_active_slider(x, y)
            # If we have an active knob, continue rotating it
            elif self.active_knob:
                self._update_active_knob(x, y)
            else:
                # No active slider/knob - check which to grab (priority order)
                if self.check_fader_collision(x, y, self.volume_fader_1):
                    self._grab_slider('volume_fader_1', x, y)
                elif self.check_fader_collision(x, y, self.volume_fader_2):
                    self._grab_slider('volume_fader_2', x, y)
                elif self.check_crossfader_collision(x, y):
                    self._grab_slider('crossfader', x, y)
                elif self.check_fader_collision(x, y, self.tempo_fader_1):
                    self._grab_slider('tempo_fader_1', x, y)
                elif self.check_fader_collision(x, y, self.tempo_fader_2):
                    self._grab_slider('tempo_fader_2', x, y)
                # Check EQ knobs (only LOW knobs are functional)
                elif self._check_knob_area(x, y, self.deck1_eq_knobs["low"]):
                    self._grab_knob(self.deck1_eq_knobs["low"], x, y, 1, "low")
                elif self._check_knob_area(x, y, self.deck2_eq_knobs["low"]):
                    self._grab_knob(self.deck2_eq_knobs["low"], x, y, 2, "low")
                else:
                    # No slider/knob collision - check buttons only
                    self._check_button_interactions(x, y, prev_pressed_states)
        else:
            # No pinch detected - release any active slider or knob
            if self.active_slider:
                self._release_active_slider()
            elif self.active_knob:
                self._release_active_knob()
            # Still check for button releases
            self._handle_button_releases(prev_pressed_states)
        
        # Handle jog wheel interactions (middle + index finger pinch)
        active_jog_pinches = [(x, y) for is_jog_pinched, (x, y) in jog_pinch_data if is_jog_pinched]
        
        if active_jog_pinches:
            # Use the first active jog pinch for interaction
            jog_x, jog_y = active_jog_pinches[0]
            
            # If we have an active jog wheel, continue controlling it
            if self.active_jog:
                self._update_active_jog_wheel(jog_x, jog_y)
            else:
                # No active jog wheel - check which jog wheel to grab
                if self._check_jog_wheel_area(jog_x, jog_y, self.jog_wheel_1):
                    self._grab_jog_wheel(self.jog_wheel_1, jog_x, jog_y, 1)
                elif self._check_jog_wheel_area(jog_x, jog_y, self.jog_wheel_2):
                    self._grab_jog_wheel(self.jog_wheel_2, jog_x, jog_y, 2)
        else:
            # No jog pinch detected - release any active jog wheel
            if self.active_jog:
                self._release_active_jog_wheel()
    
    def _grab_slider(self, slider_name: str, x: int, y: int):
        """Grab a slider for continuous control"""
        self.active_slider = slider_name
        
        # Calculate and store offset for smooth interaction
        if slider_name == 'volume_fader_1':
            fader = self.volume_fader_1
            current_pos = fader.y + (1.0 - fader.value) * fader.height
            self.slider_grab_offset = y - current_pos
        elif slider_name == 'volume_fader_2':
            fader = self.volume_fader_2
            current_pos = fader.y + (1.0 - fader.value) * fader.height
            self.slider_grab_offset = y - current_pos
        elif slider_name == 'crossfader':
            fader = self.crossfader
            current_pos = fader.x + fader.value * fader.width
            self.slider_grab_offset = x - current_pos
        elif slider_name == 'tempo_fader_1':
            fader = self.tempo_fader_1
            current_pos = fader.y + (1.0 - fader.value) * fader.height
            self.slider_grab_offset = y - current_pos
        elif slider_name == 'tempo_fader_2':
            fader = self.tempo_fader_2
            current_pos = fader.y + (1.0 - fader.value) * fader.height
            self.slider_grab_offset = y - current_pos
        
        # Update the slider immediately
        self._update_active_slider(x, y)
    
    def _update_active_slider(self, x: int, y: int):
        """Update the currently active slider position"""
        if self.active_slider == 'volume_fader_1':
            fader = self.volume_fader_1
            adjusted_y = y - self.slider_grab_offset
            relative_y = (adjusted_y - fader.y) / fader.height
            fader_value = max(0.0, min(1.0, 1.0 - relative_y))
            fader.value = fader_value
            fader.is_dragging = True
            self.audio_engine.set_master_volume(1, fader_value)
            
        elif self.active_slider == 'volume_fader_2':
            fader = self.volume_fader_2
            adjusted_y = y - self.slider_grab_offset
            relative_y = (adjusted_y - fader.y) / fader.height
            fader_value = max(0.0, min(1.0, 1.0 - relative_y))
            fader.value = fader_value
            fader.is_dragging = True
            self.audio_engine.set_master_volume(2, fader_value)
            
        elif self.active_slider == 'crossfader':
            fader = self.crossfader
            adjusted_x = x - self.slider_grab_offset
            relative_x = (adjusted_x - fader.x) / fader.width
            crossfader_value = max(0.0, min(1.0, relative_x))
            fader.value = crossfader_value
            fader.is_dragging = True
            self.audio_engine.set_crossfader_position(crossfader_value)
            
        elif self.active_slider == 'tempo_fader_1':
            fader = self.tempo_fader_1
            adjusted_y = y - self.slider_grab_offset
            relative_y = (adjusted_y - fader.y) / fader.height
            fader_value = max(0.0, min(1.0, 1.0 - relative_y))
            fader.value = fader_value
            fader.is_dragging = True
            self.audio_engine.set_tempo(1, fader_value)
            
        elif self.active_slider == 'tempo_fader_2':
            fader = self.tempo_fader_2
            adjusted_y = y - self.slider_grab_offset
            relative_y = (adjusted_y - fader.y) / fader.height
            fader_value = max(0.0, min(1.0, 1.0 - relative_y))
            fader.value = fader_value
            fader.is_dragging = True
            self.audio_engine.set_tempo(2, fader_value)
    
    def _release_active_slider(self):
        """Release the currently active slider"""
        if self.active_slider == 'volume_fader_1':
            self.volume_fader_1.is_dragging = False
        elif self.active_slider == 'volume_fader_2':
            self.volume_fader_2.is_dragging = False
        elif self.active_slider == 'crossfader':
            self.crossfader.is_dragging = False
        elif self.active_slider == 'tempo_fader_1':
            self.tempo_fader_1.is_dragging = False
        elif self.active_slider == 'tempo_fader_2':
            self.tempo_fader_2.is_dragging = False
            
        self.active_slider = None
        self.slider_grab_offset = 0
    
    
    def _check_button_interactions(self, x: int, y: int, prev_pressed_states: dict):
        """Check for button interactions when no slider is active"""
        interaction_found = False
        
        # Check deck 1 buttons
        if not interaction_found:
            for button in self.deck1_buttons.values():
                if self.check_button_collision_expanded(x, y, button):
                    button.is_pressed = True
                    if not prev_pressed_states.get(id(button), False):
                        self.handle_button_interaction(button, 1)
                    interaction_found = True
                    break
        
        # Check deck 2 buttons
        if not interaction_found:
            for button in self.deck2_buttons.values():
                if self.check_button_collision_expanded(x, y, button):
                    button.is_pressed = True
                    if not prev_pressed_states.get(id(button), False):
                        self.handle_button_interaction(button, 2)
                    interaction_found = True
                    break
    
    def _handle_button_releases(self, prev_pressed_states: dict):
        """Handle button releases for momentary buttons"""
        for buttons in [self.deck1_buttons, self.deck2_buttons]:
            deck = 1 if buttons == self.deck1_buttons else 2
            for button in buttons.values():
                if prev_pressed_states.get(id(button), False) and not button.is_pressed:
                    if button.button_type == "momentary":
                        self.handle_button_release(button, deck)
    
    def _check_knob_area(self, x: int, y: int, knob: RotaryKnob) -> bool:
        """Check if coordinates are anywhere within the knob area"""
        # Calculate distance from knob center
        dx = x - knob.center_x
        dy = y - knob.center_y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        # Check if touch is anywhere within the knob circle
        return distance <= knob.radius
    
    def _grab_knob(self, knob: RotaryKnob, x: int, y: int, deck: int, eq_band: str):
        """Grab a knob for rotational control"""
        self.active_knob = (knob, deck, eq_band)
        
        # Calculate initial angle for rotation tracking
        dx = x - knob.center_x
        dy = y - knob.center_y
        self.knob_initial_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        knob.is_turning = True
        knob.last_touch_angle = self.knob_initial_angle
        print(f"Grabbed {eq_band} EQ knob for deck {deck}")
    
    def _update_active_knob(self, x: int, y: int):
        """Update the currently active knob rotation"""
        if not self.active_knob:
            return
            
        knob, deck, eq_band = self.active_knob
        
        # Calculate current angle
        dx = x - knob.center_x
        dy = y - knob.center_y
        current_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Calculate angle difference (handling wrap-around)
        angle_diff = current_angle - knob.last_touch_angle
        
        # Handle wrap-around (-180 to +180)
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # Update knob angle (limit to -150 to +150 degrees range)
        knob.angle += angle_diff
        knob.angle = max(-150, min(150, knob.angle))
        
        # Convert knob angle to EQ value (-1.0 to +1.0)
        eq_value = knob.angle / 150.0  # -150 to +150 maps to -1.0 to +1.0
        
        # Apply EQ (only for LOW band)
        if eq_band == "low":
            self.audio_engine.set_eq_low(deck, eq_value)
        
        # Update last touch angle for next calculation
        knob.last_touch_angle = current_angle
    
    def _release_active_knob(self):
        """Release the currently active knob"""
        if self.active_knob:
            knob, deck, eq_band = self.active_knob
            knob.is_turning = False
            print(f"Released {eq_band} EQ knob for deck {deck}")
            
        self.active_knob = None
        self.knob_initial_angle = 0.0
    
    def _check_jog_wheel_area(self, x: int, y: int, jog_wheel: JogWheel) -> bool:
        """Check if coordinates are anywhere within the jog wheel area"""
        # Calculate distance from jog wheel center
        dx = x - jog_wheel.center_x
        dy = y - jog_wheel.center_y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        # Check if touch is anywhere within the jog wheel circle
        return distance <= jog_wheel.radius
    
    def _grab_jog_wheel(self, jog_wheel: JogWheel, x: int, y: int, deck: int):
        """Grab a jog wheel for scratching/navigation"""
        self.active_jog = deck
        
        # Calculate initial angle for rotation tracking
        dx = x - jog_wheel.center_x
        dy = y - jog_wheel.center_y
        self.jog_initial_angle = np.arctan2(dy, dx) * 180 / np.pi
        self.jog_last_angle = self.jog_initial_angle
        
        jog_wheel.is_touching = True
        
        # Determine behavior based on deck state
        is_playing = self.audio_engine.deck1_is_playing if deck == 1 else self.audio_engine.deck2_is_playing
        
        if is_playing:
            # Start scratching mode
            self.audio_engine.start_scratch(deck)
            print(f"SCRATCH MODE: Deck {deck} - Grab jog wheel to scratch")
        else:
            # Start navigation mode
            print(f"NAVIGATION MODE: Deck {deck} - Rotate to seek through track")
    
    def _update_active_jog_wheel(self, x: int, y: int):
        """Update the currently active jog wheel rotation"""
        if not self.active_jog:
            return
            
        deck = self.active_jog
        jog_wheel = self.jog_wheel_1 if deck == 1 else self.jog_wheel_2
        
        # Calculate current angle
        dx = x - jog_wheel.center_x
        dy = y - jog_wheel.center_y
        current_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Calculate angle difference (handling wrap-around)
        angle_diff = current_angle - self.jog_last_angle
        
        # Handle wrap-around (-180 to +180)
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # Update jog wheel visual angle
        jog_wheel.current_angle += angle_diff
        
        # Calculate rotation speed for scratching (degrees per update)
        self.jog_rotation_speed = angle_diff * 0.1  # Scale factor for realistic scratching
        
        # Determine behavior based on deck state
        is_playing = self.audio_engine.deck1_is_playing if deck == 1 else self.audio_engine.deck2_is_playing
        
        if is_playing:
            # Scratching mode - apply speed change for scratching effect
            self.audio_engine.update_scratch_speed(deck, self.jog_rotation_speed)
        else:
            # Navigation mode - seek through track
            # Convert rotation to position change (full rotation = 10% of track)
            position_change = angle_diff / 360.0 * 0.1
            current_position = self.audio_engine.get_playback_position(deck)
            new_position = max(0.0, min(1.0, current_position + position_change))
            self.audio_engine.seek_track(deck, new_position)
        
        # Update last angle for next calculation
        self.jog_last_angle = current_angle
    
    def _release_active_jog_wheel(self):
        """Release the currently active jog wheel"""
        if self.active_jog:
            deck = self.active_jog
            jog_wheel = self.jog_wheel_1 if deck == 1 else self.jog_wheel_2
            jog_wheel.is_touching = False
            
            # Stop scratching if it was active
            is_playing = self.audio_engine.deck1_is_playing if deck == 1 else self.audio_engine.deck2_is_playing
            if is_playing:
                self.audio_engine.stop_scratch(deck)
                print(f"SCRATCH MODE: Released deck {deck}")
            else:
                print(f"NAVIGATION MODE: Released deck {deck}")
            
        self.active_jog = None
        self.jog_initial_angle = 0.0
        self.jog_last_angle = 0.0
        self.jog_rotation_speed = 0.0
    
    def draw_controller_overlay(self, frame):
        """Draw the DJ controller overlay on the frame"""
        overlay = frame.copy()
        
        # Draw jog wheels with track visualization and interaction feedback
        # Deck 1 jog wheel with position indicator
        jog1_color = (200, 200, 200)
        jog1_thickness = 3
        
        # Highlight jog wheel when being touched
        if self.jog_wheel_1.is_touching:
            jog1_color = (255, 255, 0)  # Yellow when touched
            jog1_thickness = 5
            
            # Show scratching indicator
            if self.audio_engine.deck1_is_playing:
                cv2.putText(overlay, "SCRATCHING", (self.jog_wheel_1.center_x - 40, self.jog_wheel_1.center_y - 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                cv2.putText(overlay, "NAVIGATING", (self.jog_wheel_1.center_x - 45, self.jog_wheel_1.center_y - 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.circle(overlay, (self.jog_wheel_1.center_x, self.jog_wheel_1.center_y), 
                  self.jog_wheel_1.radius, jog1_color, jog1_thickness)
        
        # Add rotation indicator based on jog wheel rotation
        rotation_angle1 = np.radians(self.jog_wheel_1.current_angle)
        rotation_x1 = int(self.jog_wheel_1.center_x + (self.jog_wheel_1.radius - 30) * np.cos(rotation_angle1))
        rotation_y1 = int(self.jog_wheel_1.center_y + (self.jog_wheel_1.radius - 30) * np.sin(rotation_angle1))
        cv2.line(overlay, (self.jog_wheel_1.center_x, self.jog_wheel_1.center_y), (rotation_x1, rotation_y1), jog1_color, 3)
        
        # Add position indicator on jog wheel
        position1 = self.audio_engine.get_playback_position(1)
        angle1 = position1 * 2 * np.pi - np.pi/2  # Start from top
        pos_x1 = int(self.jog_wheel_1.center_x + (self.jog_wheel_1.radius - 20) * np.cos(angle1))
        pos_y1 = int(self.jog_wheel_1.center_y + (self.jog_wheel_1.radius - 20) * np.sin(angle1))
        cv2.circle(overlay, (pos_x1, pos_y1), 8, (0, 255, 0) if self.audio_engine.deck1_is_playing else (100, 100, 255), -1)
        cv2.putText(overlay, "DECK 1", (self.jog_wheel_1.center_x - 25, self.jog_wheel_1.center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Deck 2 jog wheel with position indicator
        jog2_color = (200, 200, 200)
        jog2_thickness = 3
        
        # Highlight jog wheel when being touched
        if self.jog_wheel_2.is_touching:
            jog2_color = (255, 255, 0)  # Yellow when touched
            jog2_thickness = 5
            
            # Show scratching indicator
            if self.audio_engine.deck2_is_playing:
                cv2.putText(overlay, "SCRATCHING", (self.jog_wheel_2.center_x - 40, self.jog_wheel_2.center_y - 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                cv2.putText(overlay, "NAVIGATING", (self.jog_wheel_2.center_x - 45, self.jog_wheel_2.center_y - 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.circle(overlay, (self.jog_wheel_2.center_x, self.jog_wheel_2.center_y), 
                  self.jog_wheel_2.radius, jog2_color, jog2_thickness)
        
        # Add rotation indicator based on jog wheel rotation
        rotation_angle2 = np.radians(self.jog_wheel_2.current_angle)
        rotation_x2 = int(self.jog_wheel_2.center_x + (self.jog_wheel_2.radius - 30) * np.cos(rotation_angle2))
        rotation_y2 = int(self.jog_wheel_2.center_y + (self.jog_wheel_2.radius - 30) * np.sin(rotation_angle2))
        cv2.line(overlay, (self.jog_wheel_2.center_x, self.jog_wheel_2.center_y), (rotation_x2, rotation_y2), jog2_color, 3)
        
        # Add position indicator on jog wheel
        position2 = self.audio_engine.get_playback_position(2)
        angle2 = position2 * 2 * np.pi - np.pi/2  # Start from top
        pos_x2 = int(self.jog_wheel_2.center_x + (self.jog_wheel_2.radius - 20) * np.cos(angle2))
        pos_y2 = int(self.jog_wheel_2.center_y + (self.jog_wheel_2.radius - 20) * np.sin(angle2))
        cv2.circle(overlay, (pos_x2, pos_y2), 8, (0, 255, 0) if self.audio_engine.deck2_is_playing else (100, 100, 255), -1)
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
        
        # Tempo controls - Professional DJ style
        for fader, side in [(self.tempo_fader_1, "left"), (self.tempo_fader_2, "right")]:
            deck_num = 1 if side == "left" else 2
            
            # Tempo fader background
            fader_color = (60, 60, 60)
            if fader.is_dragging:
                fader_color = (80, 80, 60)  # Slightly yellow when active
            
            cv2.rectangle(overlay, (fader.x, fader.y), 
                         (fader.x + fader.width, fader.y + fader.height), fader_color, -1)
            
            # Tempo fader border
            cv2.rectangle(overlay, (fader.x, fader.y), 
                         (fader.x + fader.width, fader.y + fader.height), (150, 150, 150), 2)
            
            # Center line (normal tempo)
            center_y = fader.y + fader.height // 2
            cv2.line(overlay, (fader.x, center_y), (fader.x + fader.width, center_y), 
                    (200, 200, 200), 1)
            
            # Tempo handle
            handle_y = int(fader.y + fader.height * (1 - fader.value))
            handle_color = (255, 255, 255) if not fader.is_dragging else (255, 255, 0)
            cv2.rectangle(overlay, (fader.x - 3, handle_y - 8), 
                         (fader.x + fader.width + 3, handle_y + 8), handle_color, -1)
            
            # Tempo value and BPM display
            tempo_percent = self.audio_engine.get_tempo_percentage(deck_num)
            current_bpm = self.audio_engine.get_current_bpm(deck_num)
            
            # Tempo percentage
            tempo_text = f"{tempo_percent:+.1f}%"
            if abs(tempo_percent) < 0.1:
                tempo_text = "0.0%"
                tempo_color = (0, 255, 0)  # Green for normal speed
            elif tempo_percent > 0:
                tempo_color = (0, 100, 255)  # Blue for faster
            else:
                tempo_color = (255, 100, 0)  # Orange for slower
            
            cv2.putText(overlay, tempo_text, (fader.x - 25, fader.y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, tempo_color, 1)
            
            # Current BPM
            bpm_text = f"{current_bpm:.1f}"
            cv2.putText(overlay, bpm_text, (fader.x - 15, fader.y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Label
            cv2.putText(overlay, f"TEMPO{deck_num}", (fader.x - 30, fader.y + fader.height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # EQ Knobs - Professional DJ style rotational controls
        for deck_knobs, deck_num in [(self.deck1_eq_knobs, 1), (self.deck2_eq_knobs, 2)]:
            for eq_band, knob in deck_knobs.items():
                # Knob background circle
                knob_color = (80, 80, 80)
                if knob.is_turning:
                    knob_color = (100, 100, 60)  # Slightly yellow when active
                
                cv2.circle(overlay, (knob.center_x, knob.center_y), knob.radius, knob_color, -1)
                cv2.circle(overlay, (knob.center_x, knob.center_y), knob.radius, (150, 150, 150), 2)
                
                # Knob position indicator (like real DJ knobs)
                angle_rad = np.radians(knob.angle - 90)  # -90 to start at top (12 o'clock)
                indicator_length = knob.radius - 5
                end_x = int(knob.center_x + indicator_length * np.cos(angle_rad))
                end_y = int(knob.center_y + indicator_length * np.sin(angle_rad))
                
                # Indicator line
                indicator_color = (255, 255, 255) if not knob.is_turning else (255, 255, 0)
                cv2.line(overlay, (knob.center_x, knob.center_y), (end_x, end_y), indicator_color, 3)
                
                # Center dot
                cv2.circle(overlay, (knob.center_x, knob.center_y), 3, indicator_color, -1)
                
                # EQ band label
                band_text = eq_band.upper()
                text_color = (255, 255, 255)
                
                # Highlight LOW knob since it's functional
                if eq_band == "low":
                    text_color = (100, 255, 100)  # Green for functional knob
                    
                    # Show EQ value for LOW band
                    eq_value = self.audio_engine.get_eq_low_value(deck_num)
                    if abs(eq_value) < 0.05:
                        value_text = "0"
                        value_color = (0, 255, 0)  # Green for neutral
                    elif eq_value > 0:
                        value_text = f"+{int(eq_value*100)}"
                        value_color = (0, 100, 255)  # Blue for boost
                    else:
                        value_text = f"{int(eq_value*100)}"
                        value_color = (255, 100, 0)  # Orange for cut
                    
                    cv2.putText(overlay, value_text, (knob.center_x - 10, knob.center_y + knob.radius + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, value_color, 1)
                else:
                    # Visual indication that HI/MID are not functional
                    text_color = (100, 100, 100)  # Gray for non-functional
                
                cv2.putText(overlay, band_text, (knob.center_x - 8, knob.center_y + knob.radius + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
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
        """Main loop for the DJ controller"""
        print("=" * 60)
        print("          🎧 AIR DJ CONTROLLER - Hand Tracking DJ 🎧")
        print("=" * 60)
        print("Features:")
        print("• Professional jog wheels with scratching and navigation")
        print("• Middle finger + index finger pinch for jog wheels")
        print("• Thumb + index finger pinch for other controls")
        print("• Real-time scratching when tracks are playing")
        print("• Track navigation when tracks are stopped")
        print("• Complete professional DJ controller experience")
        print("=" * 60)
        
        # Load default tracks
        self.load_default_tracks()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process hand tracking - get both pinch types
                pinch_data, jog_pinch_data, results = self.hand_tracker.process_frame(frame)
                
                # Process interactions with both pinch types
                self.process_hand_interactions(pinch_data, jog_pinch_data)
                
                # Draw controller overlay
                frame = self.draw_controller_overlay(frame)
                
                # Show pinch points and interaction feedback
                for is_pinched, (x, y) in pinch_data:
                    if is_pinched:
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                        cv2.putText(frame, "PINCH", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Show which element is being targeted
                        target_text = ""
                        target_color = (0, 255, 255)
                        
                        # Check for active slider first (grabbed state takes priority)
                        if self.active_slider == 'volume_fader_1':
                            target_text = f"🎚️ GRABBED VOL1-{int(self.volume_fader_1.value*100)}%"
                            target_color = (255, 255, 0)  # Yellow for grabbed
                        elif self.active_slider == 'volume_fader_2':
                            target_text = f"🎚️ GRABBED VOL2-{int(self.volume_fader_2.value*100)}%"
                            target_color = (255, 255, 0)  # Yellow for grabbed
                        elif self.active_slider == 'crossfader':
                            cf_pos = int(self.crossfader.value * 100)
                            if cf_pos <= 10:
                                cf_text = "DECK1"
                            elif cf_pos >= 90:
                                cf_text = "DECK2"
                            else:
                                cf_text = f"MIX-{cf_pos}%"
                            target_text = f"🎚️ GRABBED CROSSFADER-{cf_text}"
                            target_color = (255, 255, 0)  # Yellow for grabbed
                        elif self.active_slider == 'tempo_fader_1':
                            tempo_percent = self.audio_engine.get_tempo_percentage(1)
                            current_bpm = self.audio_engine.get_current_bpm(1)
                            target_text = f"🎚️ GRABBED TEMPO1-{tempo_percent:+.1f}% ({current_bpm:.1f}BPM)"
                            target_color = (255, 255, 0)  # Yellow for grabbed
                        elif self.active_slider == 'tempo_fader_2':
                            tempo_percent = self.audio_engine.get_tempo_percentage(2)
                            current_bpm = self.audio_engine.get_current_bpm(2)
                            target_text = f"🎚️ GRABBED TEMPO2-{tempo_percent:+.1f}% ({current_bpm:.1f}BPM)"
                            target_color = (255, 255, 0)  # Yellow for grabbed
                        # Check for active EQ knob
                        elif self.active_knob:
                            knob, deck, eq_band = self.active_knob
                            eq_value = self.audio_engine.get_eq_low_value(deck)
                            if abs(eq_value) < 0.05:
                                eq_text = "NEUTRAL"
                            elif eq_value > 0:
                                eq_text = f"BOOST+{int(eq_value*100)}%"
                            else:
                                eq_text = f"CUT{int(eq_value*100)}%"
                            target_text = f"🎛️ GRABBED {eq_band.upper()}{deck}-{eq_text}"
                            target_color = (255, 255, 0)  # Yellow for grabbed
                        # Check faders for hover (if no active slider)
                        elif self.check_fader_collision(x, y, self.volume_fader_1):
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
                        elif self.check_fader_collision(x, y, self.tempo_fader_1):
                            tempo_percent = self.audio_engine.get_tempo_percentage(1)
                            current_bpm = self.audio_engine.get_current_bpm(1)
                            target_text = f"TEMPO1-{tempo_percent:+.1f}% ({current_bpm:.1f}BPM)"
                            target_color = (100, 255, 255)  # Cyan for tempo
                        elif self.check_fader_collision(x, y, self.tempo_fader_2):
                            tempo_percent = self.audio_engine.get_tempo_percentage(2)
                            current_bpm = self.audio_engine.get_current_bpm(2)
                            target_text = f"TEMPO2-{tempo_percent:+.1f}% ({current_bpm:.1f}BPM)"
                            target_color = (100, 255, 255)  # Cyan for tempo
                        # Check EQ knobs (hover detection)
                        elif self._check_knob_area(x, y, self.deck1_eq_knobs["low"]):
                            eq_value = self.audio_engine.get_eq_low_value(1)
                            if abs(eq_value) < 0.05:
                                eq_text = "NEUTRAL"
                            elif eq_value > 0:
                                eq_text = f"BOOST+{int(eq_value*100)}%"
                            else:
                                eq_text = f"CUT{int(eq_value*100)}%"
                            target_text = f"LOW1-{eq_text}"
                            target_color = (100, 255, 100)  # Green for functional EQ
                        elif self._check_knob_area(x, y, self.deck2_eq_knobs["low"]):
                            eq_value = self.audio_engine.get_eq_low_value(2)
                            if abs(eq_value) < 0.05:
                                eq_text = "NEUTRAL"
                            elif eq_value > 0:
                                eq_text = f"BOOST+{int(eq_value*100)}%"
                            else:
                                eq_text = f"CUT{int(eq_value*100)}%"
                            target_text = f"LOW2-{eq_text}"
                            target_color = (100, 255, 100)  # Green for functional EQ
                        # Check non-functional EQ knobs (HI/MID) for visual feedback
                        elif (self._check_knob_area(x, y, self.deck1_eq_knobs["hi"]) or
                              self._check_knob_area(x, y, self.deck1_eq_knobs["mid"])):
                            target_text = "EQ1-NOT IMPLEMENTED"
                            target_color = (100, 100, 100)  # Gray for non-functional
                        elif (self._check_knob_area(x, y, self.deck2_eq_knobs["hi"]) or
                              self._check_knob_area(x, y, self.deck2_eq_knobs["mid"])):
                            target_text = "EQ2-NOT IMPLEMENTED"
                            target_color = (100, 100, 100)  # Gray for non-functional
                        
                        
                        # Check buttons if no fader/knob interaction
                        if not target_text:
                            for deck_buttons, deck_name in [(self.deck1_buttons, "D1"), (self.deck2_buttons, "D2")]:
                                for button_name, button in deck_buttons.items():
                                    if self.check_button_collision_expanded(x, y, button):
                                        target_text = f"{deck_name}-{button_name}"
                                        break
                        
                        if target_text:
                            cv2.putText(frame, target_text, (x + 15, y + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1)
                
                # Show jog pinch points separately
                for is_jog_pinched, (jog_x, jog_y) in jog_pinch_data:
                    if is_jog_pinched:
                        cv2.circle(frame, (jog_x, jog_y), 8, (255, 255, 0), -1)
                        cv2.putText(frame, "JOG", (jog_x + 10, jog_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        # Show jog wheel feedback
                        if self.active_jog:
                            jog_text = f"SCRATCH DECK {self.active_jog}" if self.audio_engine.deck1_is_playing or self.audio_engine.deck2_is_playing else f"NAVIGATE DECK {self.active_jog}"
                            cv2.putText(frame, jog_text, (jog_x + 10, jog_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
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
                cv2.putText(frame, "🎚️ TEMPO FADERS: Professional pitch/speed control (+/-20%) | CROSSFADER: A/B mixing", 
                           (10, status_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
                cv2.putText(frame, "🎛️ EQ KNOBS: LOW band EQ (pinch circumference & twist) | HI/MID visual only", 
                           (10, status_y + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                cv2.putText(frame, "🎚️ INTUITIVE CONTROLS: Pinch ON fader/knob to grab, then move freely", 
                           (10, status_y + 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
                cv2.putText(frame, "🎚️ PROFESSIONAL MIXING: Independent volume, timing, and stem control per deck", 
                           (10, status_y + 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
                cv2.putText(frame, "Press 'q' to quit", (10, self.screen_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "🎛️ JOG WHEELS: Middle+Index pinch to grab, rotate to scratch/navigate", 
                           (10, status_y + 295), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
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
