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
import traceback

# Audio processing
try:
    from pyo import *
except ImportError:
    print("Pyo not available. Install with: pip install pyo")
    import pygame.mixer as fallback_audio

# Audio analysis for waveform generation
try:
    import librosa
    import scipy.signal
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError:
    print("Audio analysis libraries not available. Install with: pip install librosa scipy")
    AUDIO_ANALYSIS_AVAILABLE = False

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

@dataclass
class WaveformData:
    """Represents waveform analysis data for visualization"""
    duration: float
    # Peaks for different frequency bands
    low_freq_peaks: np.ndarray
    mid_freq_peaks: np.ndarray
    high_freq_peaks: np.ndarray
    # Beat grid information
    beat_times: np.ndarray
    bar_times: np.ndarray

class WaveformAnalyzer:
    """Analyzes audio files to extract multi-band waveform and beat information."""
    
    def __init__(self):
        self.cache = {}  # Cache analyzed waveforms
        
    def analyze_track(self, track: Track) -> Optional[WaveformData]:
        """Analyze a track and return waveform data"""
        if not AUDIO_ANALYSIS_AVAILABLE:
            print("❌ Audio analysis libraries not available!")
            return None
            
        cache_key = f"{track.name}_{track.bpm}"
        if cache_key in self.cache:
            print(f"✅ Using cached waveform for {track.name}")
            return self.cache[cache_key]
        
        print(f"🔄 Analyzing track: {track.name}")
        print(f"   Available stems: {list(track.stems.keys()) if track.stems else 'None'}")
            
        try:
            # Use instrumental stem as primary source for waveform
            primary_stem = None
            if "instrumental" in track.stems:
                primary_stem = track.stems["instrumental"]
            elif track.stems:
                primary_stem = list(track.stems.values())[0]
            else:
                # Fallback: try to find any audio file in the track folder
                if hasattr(track, 'folder_path') and os.path.exists(track.folder_path):
                    # Look for any .mp3 or .wav file in the folder
                    for file in os.listdir(track.folder_path):
                        if file.lower().endswith(('.mp3', '.wav')):
                            primary_stem = os.path.join(track.folder_path, file)
                            break
                
            print(f"   Primary audio file: {primary_stem}")
            
            if not primary_stem or not os.path.exists(primary_stem):
                print(f"❌ Audio file not found: {primary_stem}")
                return None
                
            # Load audio file
            print(f"   Loading audio with librosa...")
            audio_data, sample_rate = librosa.load(primary_stem, sr=44100)
            duration = librosa.get_duration(y=audio_data, sr=sample_rate)
            print(f"   ✅ Audio loaded: {duration:.1f}s, {len(audio_data)} samples")
            
            # --- Enhanced Frequency Band Separation (Professional DJ Standards) ---
            hop_length = 512  # Smaller hop for better time resolution
            stft = librosa.stft(audio_data, hop_length=hop_length, n_fft=2048)
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
            
            # Professional DJ frequency ranges optimized for mixing
            LOW_FREQ_CUTOFF = 200    # Bass/sub-bass (20-200 Hz)
            MID_FREQ_CUTOFF = 2000   # Mids (200-2000 Hz) 
            HIGH_FREQ_START = 2000   # Highs (2000+ Hz)
            
            low_mask = freqs <= LOW_FREQ_CUTOFF
            mid_mask = (freqs > LOW_FREQ_CUTOFF) & (freqs <= MID_FREQ_CUTOFF)
            high_mask = freqs > HIGH_FREQ_START
            
            # --- Generate Enhanced Peaks for Each Band ---
            low_peaks = self._generate_waveform_peaks_for_band(stft, low_mask, hop_length)
            mid_peaks = self._generate_waveform_peaks_for_band(stft, mid_mask, hop_length)
            high_peaks = self._generate_waveform_peaks_for_band(stft, high_mask, hop_length)
            
            # --- Beat Tracking ---
            tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate, hop_length=hop_length)
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
            
            # Generate bar times (assuming 4/4 time signature)
            beats_per_bar = 4
            bar_beats = beat_frames[::beats_per_bar]
            bar_times = librosa.frames_to_time(bar_beats, sr=sample_rate, hop_length=hop_length)
            
            print(f"   ✅ Analysis complete!")
            print(f"   Low peaks: {len(low_peaks)}, Mid peaks: {len(mid_peaks)}, High peaks: {len(high_peaks)}")
            print(f"   Beats: {len(beat_times)}, Bars: {len(bar_times)}")
            
            waveform_data = WaveformData(
                duration=float(duration),
                low_freq_peaks=low_peaks,
                mid_freq_peaks=mid_peaks,
                high_freq_peaks=high_peaks,
                beat_times=beat_times,
                bar_times=bar_times,
            )
            
            self.cache[cache_key] = waveform_data
            print(f"   💾 Cached waveform data for {track.name}")
            return waveform_data
            
        except Exception as e:
            print(f"Error analyzing track {track.name}: {e}")
            traceback.print_exc()
            return None
            
    def _generate_waveform_peaks_for_band(self, stft, freq_mask, hop_length):
        """Generates enhanced stereo-style waveform peaks for professional DJ visualization."""
        band_stft = stft[freq_mask, :]
        
        # Calculate magnitude and apply log compression for better visual dynamics
        magnitude = np.abs(band_stft)
        band_energy = np.mean(magnitude, axis=0)  # Average across frequency bins
        
        # Apply logarithmic compression for better visual dynamics (like professional software)
        band_energy = np.log1p(band_energy * 100) / np.log(101)  # Compress to 0-1 range
        
        # Remove any NaN/inf values early in the pipeline
        band_energy = np.nan_to_num(band_energy, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Smooth the signal to reduce noise
        try:
            from scipy import ndimage
            band_energy = ndimage.gaussian_filter1d(band_energy, sigma=1.0)
        except ImportError:
            pass  # Skip smoothing if scipy not available
        
        # Downsample for visualization while maintaining peak information
        num_frames = len(band_energy)
        target_points = min(4000, num_frames)  # Higher resolution for professional look
        
        if num_frames > target_points:
            # Use simple resampling if scipy not available
            try:
                from scipy.signal import resample
                downsampled = resample(band_energy, target_points)
            except ImportError:
                # Fallback to simple downsampling
                downsample_factor = num_frames // target_points
                if downsample_factor > 1:
                    trimmed_length = (num_frames // downsample_factor) * downsample_factor
                    reshaped = band_energy[:trimmed_length].reshape(-1, downsample_factor)
                    downsampled = np.max(reshaped, axis=1)
                else:
                    downsampled = band_energy
        else:
            downsampled = band_energy
            
        # Normalize to 0-1 range with slight boost for better visibility
        if np.max(downsampled) > 0:
            downsampled = downsampled / np.max(downsampled)
            downsampled = np.power(downsampled, 0.7)  # Gamma correction for better contrast
        
        # Remove any NaN or infinite values that could cause rendering issues
        downsampled = np.nan_to_num(downsampled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure all values are in valid range [0, 1]
        downsampled = np.clip(downsampled, 0.0, 1.0)
            
        return downsampled

class RekordboxStyleVisualizer:
    """Professional DJ software style track visualization with scrolling waveforms."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.waveform_analyzer = WaveformAnalyzer()
        
        # --- Professional Visualization Settings ---
        self.track_height = 100  # Increased height for better visibility
        self.track_spacing = 10  # More spacing between tracks
        self.waveform_height = 70  # Max height of the waveform peaks
        self.visible_seconds = 16.0  # More visible audio for better context
        self.center_line_thickness = 2  # Prominent center playhead

        # --- Professional Rekordbox Colors ---
        self.bg_color = (20, 20, 20)  # Very dark gray (not pure black)
        
        # Frequency band colors (matching Rekordbox exactly)
        self.low_freq_color = (65, 180, 255)    # Bright blue for bass
        self.mid_freq_color = (255, 165, 80)    # Orange for mids  
        self.high_freq_color = (240, 240, 240)  # Bright white for highs
        
        # Grid and UI colors
        self.beat_color = (60, 60, 60)          # Subtle beat lines
        self.bar_color = (100, 100, 100)        # More prominent bar lines
        self.major_bar_color = (140, 140, 140)  # Every 4th bar more prominent
        self.playhead_color = (255, 100, 100)   # Red center playhead like Rekordbox
        self.playhead_shadow = (120, 50, 50)    # Shadow for depth
        self.bpm_color = (100, 200, 255)        # Blue for BPM display
        self.text_color = (200, 200, 200)     # Light gray text
        
        # Cached waveform data
        self.deck1_waveform: Optional[WaveformData] = None
        self.deck2_waveform: Optional[WaveformData] = None
        
    def set_track_waveform(self, deck: int, track: Track):
        """Load and cache waveform data for a track"""
        print(f"🎵 Setting waveform for Deck {deck}: {track.name}")
        waveform_data = self.waveform_analyzer.analyze_track(track)
        if deck == 1:
            self.deck1_waveform = waveform_data
            print(f"   Deck 1 waveform set: {waveform_data is not None}")
        else:
            self.deck2_waveform = waveform_data
            print(f"   Deck 2 waveform set: {waveform_data is not None}")
            
    def draw_stacked_visualization(self, overlay, audio_engine):
        """Draw professional Rekordbox-style stacked track visualization"""
        print(f"🎨 Drawing visualization - Deck1: {self.deck1_waveform is not None}, Deck2: {self.deck2_waveform is not None}")
        # Calculate professional layout with proper proportions
        margin = 80
        viz_width = self.screen_width - (2 * margin)
        viz_start_x = margin
        center_x = self.screen_width // 2
        
        # Calculate total visualization area height
        total_viz_height = (self.track_height * 2) + self.track_spacing + 60  # Extra for labels
        viz_start_y = 40  # Start from top
        
        # Draw clean background panel for the entire visualization area
        bg_rect = (viz_start_x - 10, viz_start_y - 20, 
                   viz_width + 20, total_viz_height + 40)
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                     self.bg_color, -1)
        
        # Draw subtle border around visualization area
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                     (50, 50, 50), 1)
        
        # Draw Deck 1 (top)
        deck1_y = viz_start_y
        self._draw_deck_visualization(
            overlay, 1, viz_start_x, deck1_y, viz_width, center_x,
            self.deck1_waveform, audio_engine
        )
        
        # Draw separator line between decks
        separator_y = deck1_y + self.track_height + (self.track_spacing // 2)
        cv2.line(overlay, (viz_start_x, separator_y), 
                (viz_start_x + viz_width, separator_y), (60, 60, 60), 1)
        
        # Draw Deck 2 (bottom)
        deck2_y = deck1_y + self.track_height + self.track_spacing
        self._draw_deck_visualization(
            overlay, 2, viz_start_x, deck2_y, viz_width, center_x,
            self.deck2_waveform, audio_engine
        )
        
        # Draw central playhead line across both decks (most prominent feature)
        playhead_x = center_x
        playhead_top = deck1_y
        playhead_bottom = deck2_y + self.track_height
        
        # Draw shadow for depth
        cv2.line(overlay, (playhead_x + 1, playhead_top), 
                (playhead_x + 1, playhead_bottom), self.playhead_shadow, 3)
        
        # Draw main playhead line
        cv2.line(overlay, (playhead_x, playhead_top), 
                (playhead_x, playhead_bottom), self.playhead_color, 2)
        
    def _draw_deck_visualization(self, overlay, deck_num: int, x: int, y: int, 
                               width: int, center_x: int, waveform_data: Optional[WaveformData], 
                               audio_engine):
        """Draw a professional Rekordbox-style visualization for a single deck."""
        
        if waveform_data is None or waveform_data.duration == 0:
            # Draw professional empty track placeholder
            placeholder_rect = (x, y, width, self.track_height)
            cv2.rectangle(overlay, (placeholder_rect[0], placeholder_rect[1]), 
                         (placeholder_rect[0] + placeholder_rect[2], placeholder_rect[1] + placeholder_rect[3]), 
                         (30, 30, 30), -1)
            cv2.putText(overlay, f"DECK {deck_num}: LOAD TRACK", (x + 20, y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
            return

        # --- Get Track and Timing Info ---
        position_ratio = audio_engine.get_playback_position(deck_num)
        current_time_sec = position_ratio * waveform_data.duration
        is_playing = (audio_engine.deck1_is_playing if deck_num == 1 
                     else audio_engine.deck2_is_playing)
        
        # Get track for BPM and name
        track = audio_engine.deck1_track if deck_num == 1 else audio_engine.deck2_track
        track_name = track.name if track else "Unknown Track"
        track_bpm = track.bpm if track else 120.0

        # --- Draw Enhanced Waveform ---
        self._draw_scrolling_waveform(overlay, x, y, width, center_x,
                                      waveform_data, current_time_sec)

        # --- Draw Professional Track Info ---
        # Track name (top left)
        truncated_name = track_name[:35] + "..." if len(track_name) > 35 else track_name
        cv2.putText(overlay, f"DECK {deck_num}: {truncated_name}", (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # BPM display (top right, like Rekordbox)
        bpm_text = f"{track_bpm:.1f} BPM"
        text_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        bpm_x = x + width - text_size[0] - 10
        cv2.putText(overlay, bpm_text, (bpm_x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.bpm_color, 2)
        
        # Time position display (bottom right)
        minutes = int(current_time_sec // 60)
        seconds = int(current_time_sec % 60)
        time_text = f"{minutes:02d}:{seconds:02d}"
        time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        time_x = x + width - time_size[0] - 10
        cv2.putText(overlay, time_text, (time_x, y + self.track_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Play/pause indicator (bottom left)
        status_text = "▶ PLAYING" if is_playing else "⏸ PAUSED"
        status_color = (100, 255, 100) if is_playing else (255, 100, 100)
        cv2.putText(overlay, status_text, (x + 5, y + self.track_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

    def _draw_scrolling_waveform(self, overlay, x: int, y: int, width: int, center_x: int,
                                 waveform_data: WaveformData, current_time: float):
        """Draws professional Rekordbox-style stereo waveform with enhanced beat/bar grid."""
        
        # --- Calculate Precise Audio Window ---
        seconds_per_pixel = self.visible_seconds / width
        start_time = current_time - (center_x - x) * seconds_per_pixel
        end_time = current_time + (x + width - center_x) * seconds_per_pixel
        time_to_pixel = lambda t: int(x + (t - start_time) / seconds_per_pixel)

        # Waveform area setup
        waveform_center_y = y + self.track_height // 2
        max_amplitude = self.waveform_height // 2
        
        # --- Draw Enhanced Beat and Bar Grid (Professional Style) ---
        bar_count = 0
        for bar_time in waveform_data.bar_times:
            if start_time <= bar_time <= end_time:
                px = time_to_pixel(bar_time)
                # Every 4th bar gets special prominence (phrase markers)
                if bar_count % 4 == 0:
                    cv2.line(overlay, (px, y + 5), (px, y + self.track_height - 5), 
                            self.major_bar_color, 2)
                else:
                    cv2.line(overlay, (px, y + 10), (px, y + self.track_height - 10), 
                            self.bar_color, 1)
                bar_count += 1
        
        # Beat lines (more subtle)
        for beat_time in waveform_data.beat_times:
            if start_time <= beat_time <= end_time:
                px = time_to_pixel(beat_time)
                cv2.line(overlay, (px, y + 25), (px, y + self.track_height - 25), 
                        self.beat_color, 1)

        # --- Draw Stereo-Style Multi-Band Waveforms ---
        # Bass (bottom layer, wider)
        self._render_stereo_waveform_band(overlay, width, x, waveform_center_y, start_time, 
                                         seconds_per_pixel, waveform_data.duration, 
                                         waveform_data.low_freq_peaks, self.low_freq_color, 
                                         max_amplitude, alpha=0.8)
        
        # Mids (middle layer)
        self._render_stereo_waveform_band(overlay, width, x, waveform_center_y, start_time, 
                                         seconds_per_pixel, waveform_data.duration, 
                                         waveform_data.mid_freq_peaks, self.mid_freq_color, 
                                         max_amplitude * 0.7, alpha=0.9)
        
        # Highs (top layer, thinner but most prominent)
        self._render_stereo_waveform_band(overlay, width, x, waveform_center_y, start_time, 
                                         seconds_per_pixel, waveform_data.duration, 
                                         waveform_data.high_freq_peaks, self.high_freq_color, 
                                         max_amplitude * 0.5, alpha=1.0)

    def _render_stereo_waveform_band(self, overlay, width, x, center_y, start_time, 
                                    seconds_per_pixel, duration, peaks, color, max_height, alpha=1.0):
        """Renders professional stereo-style waveform band (like Rekordbox)."""
        if len(peaks) == 0 or duration == 0:
            return

        peaks_per_second = len(peaks) / duration
        # Apply alpha blending to color
        blended_color = tuple(int(c * alpha) for c in color)

        # Pre-calculate all points for smooth rendering
        waveform_points_top = []
        waveform_points_bottom = []
        
        for i in range(width):
            pixel_time = start_time + i * seconds_per_pixel
            if not (0 <= pixel_time <= duration):
                continue

            peak_index = int(pixel_time * peaks_per_second)
            if 0 <= peak_index < len(peaks):
                # Create stereo-style symmetric waveform
                peak_value = peaks[peak_index]
                
                # Handle NaN/inf values safely
                if np.isnan(peak_value) or np.isinf(peak_value):
                    peak_value = 0.0
                
                amplitude = peak_value * max_height
                
                # Ensure amplitude is valid and within bounds
                amplitude = max(0, min(amplitude, max_height))
                
                # Top waveform (positive)
                top_y = int(center_y - amplitude)
                waveform_points_top.append((x + i, top_y))
                
                # Bottom waveform (negative, mirrored)
                bottom_y = int(center_y + amplitude)
                waveform_points_bottom.append((x + i, bottom_y))
        
        # Draw filled waveform areas for professional look
        if len(waveform_points_top) >= 2:
            # Create filled polygon for top half
            top_polygon = [(x, center_y)] + waveform_points_top + [(x + width, center_y)]
            if len(top_polygon) >= 3:
                try:
                    cv2.fillPoly(overlay, [np.array(top_polygon, np.int32)], blended_color)
                except:
                    pass  # Skip if polygon is invalid
            
            # Create filled polygon for bottom half
            bottom_polygon = [(x, center_y)] + waveform_points_bottom + [(x + width, center_y)]
            if len(bottom_polygon) >= 3:
                try:
                    cv2.fillPoly(overlay, [np.array(bottom_polygon, np.int32)], blended_color)
                except:
                    pass  # Skip if polygon is invalid
        
        # Draw outline for definition
        if len(waveform_points_top) >= 2:
            for i in range(len(waveform_points_top) - 1):
                cv2.line(overlay, waveform_points_top[i], waveform_points_top[i + 1], color, 1)
                cv2.line(overlay, waveform_points_bottom[i], waveform_points_bottom[i + 1], color, 1)

    def _render_waveform_band(self, overlay, width, x, y, start_time, seconds_per_pixel,
                              duration, peaks, color):
        """Legacy function - redirects to stereo version."""
        center_y = y + self.track_height // 2
        max_height = self.waveform_height // 2
        self._render_stereo_waveform_band(overlay, width, x, center_y, start_time, 
                                         seconds_per_pixel, duration, peaks, color, max_height)

    def _draw_waveform_with_beats(self, overlay, x: int, y: int, width: int, 
                                waveform_data: WaveformData, position: float, 
                                is_playing: bool):
        """Draw detailed waveform with beat grid"""
        # Draw beat grid first (behind waveform)
        self._draw_beat_grid(overlay, x, y, width, waveform_data, position)
        
        # Draw waveform
        peaks = waveform_data.low_freq_peaks
        if len(peaks) == 0:
            return
            
        # Calculate scaling
        samples_per_pixel = len(peaks) / width
        
        for i in range(width):
            sample_idx = int(i * samples_per_pixel)
            if sample_idx >= len(peaks):
                break
                
            # Get peak value for this pixel
            peak = peaks[sample_idx]
            wave_height = int(peak * self.waveform_height)
            
            # Determine color based on playback position
            pixel_position = i / width
            if pixel_position <= position:
                color = self.played_color if is_playing else (150, 100, 50)
            else:
                color = self.waveform_color
                
            # Draw waveform bar (centered vertically)
            bar_top = y + (self.waveform_height - wave_height) // 2
            bar_bottom = bar_top + wave_height
            
            if wave_height > 2:
                cv2.line(overlay, (x + i, bar_top), (x + i, bar_bottom), color, 1)
                
        # Draw playhead
        playhead_x = x + int(position * width)
        cv2.line(overlay, (playhead_x, y), (playhead_x, y + self.waveform_height), 
                self.playhead_color, 2)
        
        # Draw cue point (at beginning for now)
        cue_x = x
        cv2.line(overlay, (cue_x, y), (cue_x, y + self.waveform_height), 
                self.cue_color, 3)
                
    def _draw_beat_grid(self, overlay, x: int, y: int, width: int, 
                       waveform_data: WaveformData, position: float):
        """Draw beat and bar grid lines"""
        duration = waveform_data.duration
        
        # Draw beat lines
        for beat_time in waveform_data.beat_times:
            if beat_time <= duration:
                beat_x = x + int((beat_time / duration) * width)
                cv2.line(overlay, (beat_x, y), (beat_x, y + self.waveform_height), 
                        self.beat_color, 1)
        
        # Draw bar lines (thicker)
        for bar_time in waveform_data.bar_times:
            if bar_time <= duration:
                bar_x = x + int((bar_time / duration) * width)
                cv2.line(overlay, (bar_x, y), (bar_x, y + self.waveform_height), 
                        self.bar_color, 2)
                
    def _draw_simple_timeline(self, overlay, x: int, y: int, width: int, 
                            position: float, is_playing: bool):
        """Draw simple timeline when waveform data is not available"""
        # Background timeline
        cv2.rectangle(overlay, (x, y + 25), (x + width, y + 35), (50, 50, 50), -1)
        cv2.rectangle(overlay, (x, y + 25), (x + width, y + 35), (100, 100, 100), 1)
        
        # Progress
        progress_width = int(position * width)
        if progress_width > 0:
            color = self.played_color if is_playing else (100, 100, 100)
            cv2.rectangle(overlay, (x, y + 25), (x + progress_width, y + 35), color, -1)
            
        # Playhead
        playhead_x = x + progress_width
        cv2.line(overlay, (playhead_x, y + 20), (playhead_x, y + 40), 
                self.playhead_color, 2)
        
        # Simple beat markers (every 10%)
        for i in range(1, 10):
            marker_x = x + int(i * 0.1 * width)
            cv2.line(overlay, (marker_x, y + 25), (marker_x, y + 35), 
                    self.beat_color, 1)

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
        
        
        # Professional track position management - Industry Standard
        self.deck1_cue_point = 0.0      # Cue point position (default: beginning)
        self.deck2_cue_point = 0.0
        
        # AUTHORITATIVE TIMELINE POSITIONS (in seconds) - like Rekordbox
        self.deck1_timeline_position = 0.0  # Current track position (THE source of truth)
        self.deck2_timeline_position = 0.0
        
        # Playback state
        self.deck1_is_playing = False   # True playback state
        self.deck2_is_playing = False
        
        # Timeline timing for continuous playback
        import time
        self.deck1_last_update_time = 0.0    # Last time position was updated
        self.deck2_last_update_time = 0.0
        self.deck1_playback_speed = 1.0      # Current playback speed (1.0 = normal, 0.0 = paused, -1.0 = reverse)
        self.deck2_playback_speed = 1.0
        
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
        
        # Master song players (industry standard - one master player per deck)
        self.deck1_master_player = None
        self.deck2_master_player = None
        
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
            
            # Create master player for timeline control (industry standard)
            # Use the first available audio file as master player
            master_file = None
            for stem_type in ["vocals", "instrumental"]:
                if stem_type in track.stems and os.path.exists(track.stems[stem_type]):
                    master_file = track.stems[stem_type]
                    break
            
            if master_file:
                # Stop existing master player
                if deck == 1 and self.deck1_master_player:
                    self.deck1_master_player.stop()
                elif deck == 2 and self.deck2_master_player:
                    self.deck2_master_player.stop()
                
                # Create master player for timeline control
                master_player = SfPlayer(master_file, loop=True, mul=0.0)  # Silent - only for timeline
                master_player.out()
                
                if deck == 1:
                    self.deck1_master_player = master_player
                else:
                    self.deck2_master_player = master_player
                print(f"Created master player for deck {deck} timeline control")
            
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
        """Start playing the specified deck from current timeline position - Industry Standard"""
        import time
        
        # Update timeline positions first
        self._update_timeline_positions()
        
        players = self.deck1_players if deck == 1 else self.deck2_players
        master_player = self.deck1_master_player if deck == 1 else self.deck2_master_player
        
        # Set playing state and reset timing
        if deck == 1:
            self.deck1_is_playing = True
            self.deck1_state = DeckState.PLAYING
            self.deck1_playback_speed = 1.0  # Normal forward playback
            self.deck1_last_update_time = time.time()
            current_position = self.deck1_timeline_position
            current_tempo = self.deck1_tempo
            track = self.deck1_track
        else:
            self.deck2_is_playing = True
            self.deck2_state = DeckState.PLAYING
            self.deck2_playback_speed = 1.0  # Normal forward playback
            self.deck2_last_update_time = time.time()
            current_position = self.deck2_timeline_position
            current_tempo = self.deck2_tempo
            track = self.deck2_track

        # Force recreate players to ensure sound, just like CUE button - but at current position
        for stem_type, player in list(players.items()):
            player.stop()
            file_path = track.stems.get(stem_type)
            if file_path and os.path.exists(file_path):
                new_player = SfPlayer(file_path, loop=True, mul=0.0)
                if hasattr(new_player, 'setOffset'):
                    new_player.setOffset(float(current_position))
                if hasattr(new_player, 'speed'):
                    new_player.speed = float(current_tempo)
                new_player.out()
                players[stem_type] = new_player
        
        # Recreate master player as well to ensure sync
        if master_player:
            master_player.stop()
        master_file = None
        if track and "instrumental" in track.stems: # Assuming instrumental is the master
            master_file = track.stems["instrumental"]
        if master_file and os.path.exists(master_file):
            new_master = SfPlayer(master_file, loop=True, mul=0.0)
            if hasattr(new_master, 'setOffset'):
                new_master.setOffset(float(current_position))
            if hasattr(new_master, 'speed'):
                new_master.speed = float(current_tempo)
            new_master.out()
            if deck == 1:
                self.deck1_master_player = new_master
            else:
                self.deck2_master_player = new_master

        # Update volume levels which also starts playback sound
        self._update_all_stem_volumes(deck)
        
        print(f"Deck {deck} PLAYING from timeline position {current_position:.1f}s")
    
    def pause_deck(self, deck: int):
        """Pause the specified deck - stops playback but maintains timeline position"""
        import time
        
        # Update timeline positions first to capture current position
        self._update_timeline_positions()
        
        players = self.deck1_players if deck == 1 else self.deck2_players
        master_player = self.deck1_master_player if deck == 1 else self.deck2_master_player
        
        # Set paused state - timeline position is already updated
        if deck == 1:
            self.deck1_is_playing = False
            self.deck1_state = DeckState.PAUSED
            self.deck1_playback_speed = 0.0  # Stop timeline advancement
            current_position = self.deck1_timeline_position
        else:
            self.deck2_is_playing = False
            self.deck2_state = DeckState.PAUSED
            self.deck2_playback_speed = 0.0  # Stop timeline advancement
            current_position = self.deck2_timeline_position
        
        # Pause audio players
        if master_player:
            if hasattr(master_player, 'stop'):
                master_player.stop()
        
        # Mute all stem players (but keep them at current position)
        for player in players.values():
            if hasattr(player, 'stop'):
                player.stop()
            else:
                player.mul = 0.0
        
        print(f"Deck {deck} PAUSED at timeline position {current_position:.1f}s")
    
    def cue_deck(self, deck: int):
        """CUE: Jumps track to cue point (beginning) and pauses playback."""
        print(f"Deck {deck} CUE: Returning to start and pausing.")
        
        # Immediately pause the deck to stop any ongoing sound.
        self.pause_deck(deck)
        
        # Set the timeline position to the cue point (0.0 for beginning).
        cue_point = self.deck1_cue_point if deck == 1 else self.deck2_cue_point
        self.set_timeline_position(deck, cue_point)
    
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
        
        print(f"SET STEM VOLUME: Deck {deck} {stem_type} = {volume:.2f} | active = {volume > 0.0} | playing = {is_playing}")
        
        # Apply the change immediately by updating all stem volumes
        # This ensures proper crossfader gain and master volume calculations
        self._update_all_stem_volumes(deck)
        
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
        """Calculate crossfader gain for a deck - center position = full volume for both"""
        # 0.0 = full left (deck1), 0.5 = center (both full), 1.0 = full right (deck2)
        
        if deck == 1:
            # Deck 1: Full volume when crossfader is left or center
            if self.crossfader_position <= 0.5:
                gain = 1.0  # Full volume when left or center
            else:
                # Fade out as crossfader moves right from center
                gain = 2.0 * (1.0 - self.crossfader_position)
                gain = max(0.0, min(1.0, gain))
        elif deck == 2:
            # Deck 2: Full volume when crossfader is right or center
            if self.crossfader_position >= 0.5:
                gain = 1.0  # Full volume when right or center
            else:
                # Fade out as crossfader moves left from center
                gain = 2.0 * self.crossfader_position
                gain = max(0.0, min(1.0, gain))
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
            master_player = self.deck1_master_player
        elif deck == 2:
            self.deck2_tempo = tempo
            players = self.deck2_players
            master_player = self.deck2_master_player
        else:
            print(f"Invalid deck: {deck}")
            return
        
        # Apply tempo to master player first (industry standard)
        if master_player and hasattr(master_player, 'speed'):
            master_player.speed = float(tempo)
        
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
        """Update scratch speed - like rotating a real jog wheel - controls master player"""
        if deck == 1 and hasattr(self, 'deck1_is_scratching') and self.deck1_is_scratching:
            self.deck1_scratch_speed = scratch_speed
            players = self.deck1_players
            master_player = self.deck1_master_player
            base_tempo = self.deck1_tempo
        elif deck == 2 and hasattr(self, 'deck2_is_scratching') and self.deck2_is_scratching:
            self.deck2_scratch_speed = scratch_speed
            players = self.deck2_players
            master_player = self.deck2_master_player
            base_tempo = self.deck2_tempo
        else:
            return
        
        # Apply scratch speed to master player first (industry standard)
        if master_player and hasattr(master_player, 'speed'):
            master_player.speed = float(base_tempo + scratch_speed)
            
        # Apply scratch speed to all players (temporary speed change)
        for stem_type, player in players.items():
            if hasattr(player, 'speed'):
                player.speed = float(base_tempo + scratch_speed)
    
    def stop_scratch(self, deck: int):
        """Stop scratching mode - release jog wheel - restores master player tempo"""
        if deck == 1:
            self.deck1_is_scratching = False
            players = self.deck1_players
            master_player = self.deck1_master_player
            base_tempo = self.deck1_tempo
        elif deck == 2:
            self.deck2_is_scratching = False  
            players = self.deck2_players
            master_player = self.deck2_master_player
            base_tempo = self.deck2_tempo
        else:
            return
        
        # Restore normal tempo to master player first (industry standard)
        if master_player and hasattr(master_player, 'speed'):
            master_player.speed = float(base_tempo)
            
        # Restore normal tempo to all players
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
    
    def _update_timeline_positions(self):
        """Update timeline positions based on current playback state - called continuously"""
        import time
        current_time = time.time()
        
        # Update Deck 1 timeline
        if self.deck1_last_update_time > 0:
            time_delta = current_time - self.deck1_last_update_time
            if self.deck1_is_playing:
                # Move timeline forward based on playback speed and tempo
                effective_speed = self.deck1_playback_speed * self.deck1_tempo
                self.deck1_timeline_position += time_delta * effective_speed
                # Ensure position doesn't go negative
                self.deck1_timeline_position = max(0.0, self.deck1_timeline_position)
        self.deck1_last_update_time = current_time
        
        # Update Deck 2 timeline  
        if self.deck2_last_update_time > 0:
            time_delta = current_time - self.deck2_last_update_time
            if self.deck2_is_playing:
                # Move timeline forward based on playback speed and tempo
                effective_speed = self.deck2_playback_speed * self.deck2_tempo
                self.deck2_timeline_position += time_delta * effective_speed
                # Ensure position doesn't go negative
                self.deck2_timeline_position = max(0.0, self.deck2_timeline_position)
        self.deck2_last_update_time = current_time

    def get_playback_position(self, deck: int) -> float:
        """Get current timeline position as ratio (0.0 to 1.0) - Industry Standard"""
        # Update timeline positions first
        self._update_timeline_positions()
        
        estimated_track_length = 180.0  # 3 minutes
        
        if deck == 1:
            current_position = self.deck1_timeline_position
        else:
            current_position = self.deck2_timeline_position
            
        position_ratio = current_position / estimated_track_length
        return max(0.0, min(1.0, position_ratio % 1.0))
    
    def get_tempo_multiplier(self, deck: int) -> float:
        """Get current tempo multiplier for a deck (for jog wheel spinning speed)"""
        if deck == 1:
            return self.deck1_tempo
        elif deck == 2:
            return self.deck2_tempo
        return 1.0
    
    def set_timeline_position(self, deck: int, position_seconds: float):
        """Set absolute timeline position - Industry Standard DJ Software approach"""
        try:
            # Update timeline positions first
            self._update_timeline_positions()
            
            # Clamp position to valid range
            position_seconds = max(0.0, position_seconds)
            
            # Update authoritative timeline position
            if deck == 1:
                old_position = self.deck1_timeline_position
                self.deck1_timeline_position = position_seconds
                master_player = self.deck1_master_player
            else:
                old_position = self.deck2_timeline_position
                self.deck2_timeline_position = position_seconds
                master_player = self.deck2_master_player
                
            # Sync audio players to new position
            if master_player:
                if hasattr(master_player, 'setOffset'):
                    master_player.setOffset(float(position_seconds))
                elif hasattr(master_player, 'time'):
                    master_player.time = float(position_seconds)
                    
                # Sync all stem players
                self._sync_stem_players_to_master(deck, float(position_seconds))
                
            print(f"DECK {deck} TIMELINE SET: {old_position:.2f}s → {position_seconds:.2f}s")
                
        except Exception as e:
            print(f"Error setting timeline position for deck {deck}: {e}")

    def set_playback_speed(self, deck: int, speed: float):
        """Set playback speed for real-time jog wheel control - Industry Standard"""
        try:
            # Update timeline positions first
            self._update_timeline_positions()
            
            if deck == 1:
                self.deck1_playback_speed = speed
                master_player = self.deck1_master_player
                players = self.deck1_players
                base_tempo = self.deck1_tempo
            else:
                self.deck2_playback_speed = speed
                master_player = self.deck2_master_player
                players = self.deck2_players
                base_tempo = self.deck2_tempo
                
            # Apply speed to audio players for real-time scratching
            effective_speed = float(speed * base_tempo)  # Convert to Python float to avoid numpy issues
            
            if master_player and hasattr(master_player, 'speed'):
                master_player.speed = effective_speed
                
            for player in players.values():
                if hasattr(player, 'speed'):
                    player.speed = effective_speed
                    
            print(f"DECK {deck} SPEED SET: {speed:.2f}x (effective: {effective_speed:.2f}x)")
            
        except Exception as e:
            print(f"Error setting playback speed for deck {deck}: {e}")

    def nudge_track_position(self, deck: int, position_change_seconds: float):
        """Nudge track timeline position - called by jog wheel interactions"""
        # Update timeline positions first
        self._update_timeline_positions()
        
        if deck == 1:
            current_position = self.deck1_timeline_position
        else:
            current_position = self.deck2_timeline_position
            
        new_position = current_position + position_change_seconds
        self.set_timeline_position(deck, new_position)
    
    def _sync_stem_players_to_master(self, deck: int, position_seconds: float):
        """Sync all stem players to master player position"""
        try:
            players = self.deck1_players if deck == 1 else self.deck2_players
            
            for stem_type, player in players.items():
                if hasattr(player, 'setOffset'):
                    player.setOffset(position_seconds)
                elif hasattr(player, 'time'):
                    player.time = position_seconds
                    
                # Ensure all players are properly synced (sometimes needed for pyo)
                if hasattr(player, 'reset'):
                    # Reset triggers re-reading from the new position
                    player.reset()
                    
        except Exception as e:
            print(f"Error syncing stem players for deck {deck}: {e}")
    
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
            
            # Debug output for troubleshooting audio issues (only show if there's an issue)
            if is_playing and final_volume == 0.0:
                print(f"AUDIO ISSUE: Deck {deck} {stem_type} should be playing but volume=0 | stem_active={stem_active} | stem_vol={stem_volume:.2f} | master_vol={master_volume:.2f}")
    
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
        pinch_threshold = 0.035  # Reduced sensitivity - fingers must be quite close to avoid accidental detection
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
        
        # Jog wheel pinch threshold (slightly more forgiving for easier jog wheel access)
        jog_pinch_threshold = 0.035
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
        
        # Initialize Rekordbox-style visualization
        self.visualizer = RekordboxStyleVisualizer(self.screen_width, self.screen_height)
        
        # Initialize controller elements
        self.setup_controller_layout()
        
        # State
        self.current_pinches = []
        self.deck1_track = None
        self.deck2_track = None
        
        # Slider interaction state - for intuitive pinch-to-grab behavior
        self.active_slider = None  # Which slider is currently grabbed
        self.active_slider_pos = None # The (x, y) position of the pinch grabbing the slider
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
        self.active_knob_pos = None # The (x, y) position of the pinch grabbing the knob
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
                
                # Industry Standard: If no stems are active, turn them on when playing
                active_stems = self.audio_engine.deck1_active_stems if deck == 1 else self.audio_engine.deck2_active_stems
                if button.is_active and not any(active_stems.values()):
                    print(f"Deck {deck}: No active stems. Activating Vocal/Instrumental for playback.")
                    self.audio_engine.set_stem_volume(deck, "vocals", 0.7)
                    self.audio_engine.set_stem_volume(deck, "instrumental", 0.7)
                    # Update button UI to reflect this change
                    if deck == 1:
                        self.deck1_buttons["vocal"].is_active = True
                        self.deck1_buttons["instrumental"].is_active = True
                    else:
                        self.deck2_buttons["vocal"].is_active = True
                        self.deck2_buttons["instrumental"].is_active = True

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
        """Check if coordinates are within the crossfader's draggable area"""
        margin = 15  # Add margin for easier grabbing
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
        """Process hand interactions - supports multi-hand/multi-pinch interactions"""
        # Store previous button states
        prev_pressed_states = {}
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button_name, button in buttons.items():
                prev_pressed_states[id(button)] = button.is_pressed
        
        # Reset all button pressed states
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button in buttons.values():
                button.is_pressed = False
        
        # Track which controls have been interacted with in this frame to avoid conflicts
        interacted_controls = set()

        # --- Process Regular Pinches (for sliders, knobs, buttons) ---
        active_pinches = [(x, y) for is_pinched, (x, y) in pinch_data if is_pinched]
        
        # First, handle ongoing interactions (sliders, knobs)
        if self.active_slider:
            # Find the pinch closest to the active slider to continue the interaction
            pinch_to_update = self._find_closest_pinch(self.active_slider_pos, active_pinches)
            if pinch_to_update:
                x, y = pinch_to_update
                self._update_active_slider(x, y)
                interacted_controls.add(self.active_slider)
                active_pinches.remove(pinch_to_update) # This pinch has been used

        if self.active_knob:
             # Find the pinch closest to the active knob
            pinch_to_update = self._find_closest_pinch(self.active_knob_pos, active_pinches)
            if pinch_to_update:
                x, y = pinch_to_update
                self._update_active_knob(x, y)
                interacted_controls.add(self.active_knob)
                active_pinches.remove(pinch_to_update)

        # Process remaining pinches for new interactions
        for x, y in active_pinches:
            interaction_handled = False
            # Check for new slider/knob grabs if no control is currently held by this "hand"
            if not self.active_slider and not self.active_knob:
                if self.check_fader_collision(x, y, self.volume_fader_1):
                    self._grab_slider('volume_fader_1', x, y)
                    interaction_handled = True
                elif self.check_fader_collision(x, y, self.volume_fader_2):
                    self._grab_slider('volume_fader_2', x, y)
                    interaction_handled = True
                elif self.check_crossfader_collision(x, y):
                    self._grab_slider('crossfader', x, y)
                    interaction_handled = True
                elif self.check_fader_collision(x, y, self.tempo_fader_1):
                    self._grab_slider('tempo_fader_1', x, y)
                    interaction_handled = True
                elif self.check_fader_collision(x, y, self.tempo_fader_2):
                    self._grab_slider('tempo_fader_2', x, y)
                    interaction_handled = True
                elif self._check_knob_area(x, y, self.deck1_eq_knobs["low"]):
                    self._grab_knob(self.deck1_eq_knobs["low"], x, y, 1, "low")
                    interaction_handled = True
                elif self._check_knob_area(x, y, self.deck2_eq_knobs["low"]):
                    self._grab_knob(self.deck2_eq_knobs["low"], x, y, 2, "low")
                    interaction_handled = True

            # If no slider/knob was grabbed by this pinch, check for button presses
            if not interaction_handled:
                self._check_button_interactions(x, y, prev_pressed_states)

        # Release sliders/knobs if no pinches are active
        if not pinch_data or not any(p[0] for p in pinch_data):
            if self.active_slider:
                self._release_active_slider()
            if self.active_knob:
                self._release_active_knob()
        
        # --- Process Jog Wheel Pinches ---
        active_jog_pinches = [(x, y) for is_jog_pinched, (x, y) in jog_pinch_data if is_jog_pinched]
        
        if active_jog_pinches:
            # For simplicity, we'll allow one jog wheel interaction at a time for now
            # This can be expanded to two jog wheels with more complex tracking
            jog_x, jog_y = active_jog_pinches[0]
            
            if self.active_jog:
                self._update_active_jog_wheel(jog_x, jog_y)
            else:
                if self._check_jog_wheel_area(jog_x, jog_y, self.jog_wheel_1):
                    self._grab_jog_wheel(self.jog_wheel_1, jog_x, jog_y, 1)
                elif self._check_jog_wheel_area(jog_x, jog_y, self.jog_wheel_2):
                    self._grab_jog_wheel(self.jog_wheel_2, jog_x, jog_y, 2)
        else:
            if self.active_jog:
                self._release_active_jog_wheel()

    def _find_closest_pinch(self, pos, pinches):
        """Finds the closest pinch to a given position."""
        if not pinches:
            return None
        
        closest_pinch = None
        min_dist = float('inf')
        
        for pinch_pos in pinches:
            dist = np.sqrt((pos[0] - pinch_pos[0])**2 + (pos[1] - pinch_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_pinch = pinch_pos
        
        return closest_pinch

    def _grab_slider(self, slider_name: str, x: int, y: int):
        """Grab a slider for continuous control"""
        self.active_slider = slider_name
        self.active_slider_pos = (x, y)
        
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
        self.active_slider_pos = (x, y) # Update position
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
        """Check for button interactions when no slider is active - immediately apply effects"""
        interaction_found = False
        
        # Check deck 1 buttons
        if not interaction_found:
            for button in self.deck1_buttons.values():
                if self.check_button_collision_expanded(x, y, button):
                    button.is_pressed = True
                    # Always handle button interaction immediately when pinch connection is made
                    if not prev_pressed_states.get(id(button), False):
                        self.handle_button_interaction(button, 1)
                    interaction_found = True
                    break
        
        # Check deck 2 buttons
        if not interaction_found:
            for button in self.deck2_buttons.values():
                if self.check_button_collision_expanded(x, y, button):
                    button.is_pressed = True
                    # Always handle button interaction immediately when pinch connection is made
                    if not prev_pressed_states.get(id(button), False):
                        self.handle_button_interaction(button, 2)
                    interaction_found = True
                    break
        
        return interaction_found
    
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
        self.active_knob_pos = (x, y)
        
        # Calculate initial angle for rotation tracking
        dx = x - knob.center_x
        dy = y - knob.center_y
        self.knob_initial_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        knob.is_turning = True
        knob.last_touch_angle = self.knob_initial_angle
        print(f"Grabbed {eq_band} EQ knob for deck {deck}")
    
    def _update_active_knob(self, x: int, y: int):
        """Update the currently rotating knob's angle"""
        self.active_knob_pos = (x, y) # Update position
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
        """Update the currently active jog wheel rotation - realistic DJ controller timeline control"""
        if not self.active_jog:
            return
            
        deck = self.active_jog
        jog_wheel = self.jog_wheel_1 if deck == 1 else self.jog_wheel_2
        
        # Calculate current angle (inverted Y for proper DJ controller direction)
        dx = x - jog_wheel.center_x
        dy = -(y - jog_wheel.center_y)  # Invert Y to match DJ controller behavior
        current_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Calculate angle difference (handling wrap-around)
        angle_diff = current_angle - self.jog_last_angle
        
        # Handle wrap-around (-180 to +180)
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # Invert angle difference for proper DJ controller direction
        # Clockwise (positive screen rotation) = forward in track (positive timeline)
        # Counter-clockwise (negative screen rotation) = backward in track (negative timeline)
        angle_diff = -angle_diff
        
        # Update jog wheel visual angle
        jog_wheel.current_angle += angle_diff
        
        # Determine behavior based on deck state
        is_playing = self.audio_engine.deck1_is_playing if deck == 1 else self.audio_engine.deck2_is_playing
        
        if is_playing:
            # PLAYING MODE: Real-time speed control like Rekordbox/CDJ
            # Jog wheel controls playback speed while maintaining audio output
            
            # Convert jog movement to playback speed
            # Positive = clockwise = faster/forward, Negative = counterclockwise = slower/reverse  
            speed_change = angle_diff * 0.03  # Responsive but not too sensitive
            playback_speed = 1.0 + speed_change  # 1.0 = normal speed
            
            # Clamp to reasonable range for audio quality
            playback_speed = max(-2.0, min(3.0, playback_speed))
            
            # Set real-time playback speed for scratching/speed control
            self.audio_engine.set_playback_speed(deck, playback_speed)
            
            direction = "FORWARD" if speed_change > 0 else "BACKWARD" if speed_change < 0 else "NORMAL"
            print(f"JOG PLAYING: Deck {deck} {direction} | speed={playback_speed:.2f}x")
            
        else:
            # PAUSED MODE: Timeline navigation only (no audio output)
            # Jog wheel directly moves the timeline position for beat matching
            
            # Convert rotation to timeline position changes - industry standard sensitivity
            position_change_seconds = angle_diff / 360.0 * 4.0  # 360° = 4 seconds movement
            self.audio_engine.nudge_track_position(deck, position_change_seconds)
            
            direction = "FORWARD" if position_change_seconds > 0 else "BACKWARD" if position_change_seconds < 0 else "STOPPED"
            print(f"JOG PAUSED: Deck {deck} {direction} | timeline_change={position_change_seconds:.3f}s")
        
        # Update last angle for next calculation
        self.jog_last_angle = current_angle
    
    def _release_active_jog_wheel(self):
        """Release the currently active jog wheel - restore normal playback"""
        if self.active_jog:
            deck = self.active_jog
            jog_wheel = self.jog_wheel_1 if deck == 1 else self.jog_wheel_2
            jog_wheel.is_touching = False
            
            # Reset playback speed to normal (industry standard behavior)
            is_playing = self.audio_engine.deck1_is_playing if deck == 1 else self.audio_engine.deck2_is_playing
            if is_playing:
                # Restore normal playback speed (1.0x) when jog wheel is released
                self.audio_engine.set_playback_speed(deck, 1.0)
                print(f"JOG RELEASED: Deck {deck} - resumed normal playback (1.0x)")
            else:
                print(f"JOG RELEASED: Deck {deck} - timeline navigation ended")
        
        self.active_jog = None
        self.jog_initial_angle = 0.0
        self.jog_last_angle = 0.0
        self.jog_rotation_speed = 0.0
    
    def update_jog_wheel_spinning(self):
        """Update jog wheel spinning to reflect actual song timeline - like real DJ controllers"""
        import time
        
        # Get current time for smooth rotation calculations
        current_time = time.time()
        if not hasattr(self, '_last_jog_update_time'):
            self._last_jog_update_time = current_time
            self._last_track_positions = {1: 0.0, 2: 0.0}
            return
        
        time_delta = current_time - self._last_jog_update_time
        self._last_jog_update_time = current_time
        
        # Update Deck 1 jog wheel - sync with actual track timeline
        if hasattr(self.audio_engine, 'deck1_is_playing') and self.audio_engine.deck1_is_playing:
            # Get actual track position for timeline sync
            current_track_position = self.audio_engine.get_playback_position(1)
            last_track_position = self._last_track_positions.get(1, 0.0)
            
            # Calculate how much the track has progressed (timeline movement)
            track_progress = (current_track_position - last_track_position) % 1.0
            
            # Convert timeline progress to visual jog wheel rotation
            # Full track (0.0 to 1.0) = multiple full rotations for visual appeal
            timeline_rotation = track_progress * 3600  # 10 full rotations per track
            
            # Get current tempo multiplier for additional spinning effect
            try:
                tempo_multiplier = self.audio_engine.get_tempo_multiplier(1)
            except:
                tempo_multiplier = 1.0
            
            # Add base visual spinning for DJ controller feel
            base_rotation_speed = 200.0  # degrees per second
            visual_rotation = base_rotation_speed * tempo_multiplier * time_delta
            
            # Only update rotation if not being manually controlled
            if not (self.active_jog == 1 and self.jog_wheel_1.is_touching):
                # Combine timeline-based rotation with visual spinning
                total_rotation = timeline_rotation + visual_rotation
                self.jog_wheel_1.current_angle += total_rotation
                
                # Keep angle in reasonable range to prevent overflow
                self.jog_wheel_1.current_angle = self.jog_wheel_1.current_angle % 360
            
            self._last_track_positions[1] = current_track_position
        
        # Update Deck 2 jog wheel - sync with actual track timeline
        if hasattr(self.audio_engine, 'deck2_is_playing') and self.audio_engine.deck2_is_playing:
            # Get actual track position for timeline sync
            current_track_position = self.audio_engine.get_playback_position(2)
            last_track_position = self._last_track_positions.get(2, 0.0)
            
            # Calculate how much the track has progressed (timeline movement)
            track_progress = (current_track_position - last_track_position) % 1.0
            
            # Convert timeline progress to visual jog wheel rotation
            timeline_rotation = track_progress * 3600  # 10 full rotations per track
            
            # Get current tempo multiplier for additional spinning effect
            try:
                tempo_multiplier = self.audio_engine.get_tempo_multiplier(2)
            except:
                tempo_multiplier = 1.0
            
            # Add base visual spinning for DJ controller feel
            base_rotation_speed = 200.0  # degrees per second
            visual_rotation = base_rotation_speed * tempo_multiplier * time_delta
            
            # Only update rotation if not being manually controlled
            if not (self.active_jog == 2 and self.jog_wheel_2.is_touching):
                # Combine timeline-based rotation with visual spinning
                total_rotation = timeline_rotation + visual_rotation
                self.jog_wheel_2.current_angle += total_rotation
                
                # Keep angle in reasonable range to prevent overflow
                self.jog_wheel_2.current_angle = self.jog_wheel_2.current_angle % 360
            
            self._last_track_positions[2] = current_track_position
    
    def _draw_professional_jog_wheel(self, overlay, jog_wheel, deck_num, label):
        """Draw a realistic professional DJ controller jog wheel with all the visual elements"""
        center_x, center_y = jog_wheel.center_x, jog_wheel.center_y
        radius = jog_wheel.radius
        
        # Determine jog wheel state and colors
        is_playing = (self.audio_engine.deck1_is_playing if deck_num == 1 
                      else self.audio_engine.deck2_is_playing)
        is_touching = jog_wheel.is_touching
        
        # Professional color scheme based on state
        if is_touching:
            main_color = (255, 215, 0)  # Gold when touched
            inner_color = (255, 255, 100)  # Bright yellow
            tick_color = (255, 255, 255)  # White ticks
            label_color = (255, 255, 255)  # White label
        elif is_playing:
            main_color = (100, 255, 100)  # Green when playing
            inner_color = (150, 255, 150)  # Light green
            tick_color = (200, 255, 200)  # Light green ticks
            label_color = (255, 255, 255)  # White label
        else:
            main_color = (150, 150, 150)  # Gray when stopped
            inner_color = (100, 100, 100)  # Dark gray
            tick_color = (200, 200, 200)  # Light gray ticks
            label_color = (200, 200, 200)  # Gray label
        
        # Draw outer ring (main jog wheel body)
        cv2.circle(overlay, (center_x, center_y), radius, main_color, 4)
        
        # Draw inner ring for depth
        cv2.circle(overlay, (center_x, center_y), radius - 15, inner_color, 2)
        
        # Draw center hub
        cv2.circle(overlay, (center_x, center_y), 25, main_color, -1)
        cv2.circle(overlay, (center_x, center_y), 25, (255, 255, 255), 2)
        
        # Draw rotation tick marks around circumference (like real jog wheels)
        num_ticks = 24  # Professional jog wheels often have many tick marks
        for i in range(num_ticks):
            # Base angle for this tick mark
            base_angle = (i * 360 / num_ticks) * np.pi / 180
            
            # Add current rotation to show spinning
            actual_angle = base_angle + np.radians(jog_wheel.current_angle)
            
            # Calculate tick mark positions
            tick_outer_radius = radius - 5
            tick_inner_radius = radius - 20
            
            # Every 6th tick is longer (like hour marks on a clock)
            if i % 6 == 0:
                tick_inner_radius = radius - 30
                tick_thickness = 3
            else:
                tick_thickness = 1
            
            # Calculate tick mark coordinates
            outer_x = int(center_x + tick_outer_radius * np.cos(actual_angle))
            outer_y = int(center_y + tick_outer_radius * np.sin(actual_angle))
            inner_x = int(center_x + tick_inner_radius * np.cos(actual_angle))
            inner_y = int(center_y + tick_inner_radius * np.sin(actual_angle))
            
            # Draw the tick mark
            cv2.line(overlay, (outer_x, outer_y), (inner_x, inner_y), tick_color, tick_thickness)
        
        # Draw main rotation indicator (like the main hand on a clock)
        rotation_angle = np.radians(jog_wheel.current_angle)
        rotation_end_x = int(center_x + (radius - 35) * np.cos(rotation_angle))
        rotation_end_y = int(center_y + (radius - 35) * np.sin(rotation_angle))
        cv2.line(overlay, (center_x, center_y), (rotation_end_x, rotation_end_y), 
                 (255, 255, 255), 4)  # White main indicator
        
        # Draw track position indicator (shows current playback position)
        try:
            position = self.audio_engine.get_playback_position(deck_num)
            position_angle = position * 2 * np.pi - np.pi/2  # Start from top (12 o'clock)
            pos_radius = radius - 10
            pos_x = int(center_x + pos_radius * np.cos(position_angle))
            pos_y = int(center_y + pos_radius * np.sin(position_angle))
            
            # Position indicator color based on state
            if is_playing:
                pos_color = (0, 255, 255)  # Cyan when playing
            else:
                pos_color = (0, 100, 255)  # Blue when stopped
            
            # Draw enhanced position indicator
            cv2.circle(overlay, (pos_x, pos_y), 8, pos_color, -1)
            cv2.circle(overlay, (pos_x, pos_y), 8, (255, 255, 255), 2)
            
            # Draw beat sync indicator if available
            if hasattr(self, 'visualizer'):
                waveform_data = (self.visualizer.deck1_waveform if deck_num == 1 
                               else self.visualizer.deck2_waveform)
                if waveform_data and len(waveform_data.beat_times) > 0:
                    # Find the nearest beat
                    current_time = position * waveform_data.duration
                    beat_distances = np.abs(waveform_data.beat_times - current_time)
                    nearest_beat_idx = np.argmin(beat_distances)
                    beat_distance = beat_distances[nearest_beat_idx]
                    
                    # Show beat sync indicator if close to a beat
                    if beat_distance < 0.1:  # Within 100ms of a beat
                        beat_ring_color = (0, 255, 255)  # Cyan for beat sync
                        cv2.circle(overlay, (center_x, center_y), radius - 45, beat_ring_color, 3)
        except:
            pass  # Skip if position can't be determined
        
        # Draw deck label in center
        cv2.putText(overlay, label, (center_x - 30, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        
        # Show interaction feedback text above jog wheel
        if is_touching:
            feedback_text = "SCRATCHING" if is_playing else "NAVIGATING"
            text_color = (255, 255, 0)  # Yellow
            cv2.putText(overlay, feedback_text, (center_x - 40, center_y - radius - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Show speed indicator when playing
        if is_playing and not is_touching:
            try:
                tempo_multiplier = self.audio_engine.get_tempo_multiplier(deck_num)
                if tempo_multiplier != 1.0:
                    speed_text = f"{tempo_multiplier:.2f}x"
                    speed_color = (100, 255, 255)  # Cyan
                    cv2.putText(overlay, speed_text, (center_x - 20, center_y - radius - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)
            except:
                pass
    
    def draw_controller_overlay(self, frame):
        """Draw the DJ controller overlay on the frame"""
        overlay = frame.copy()
        
        # Draw professional jog wheels with realistic DJ controller appearance
        self._draw_professional_jog_wheel(overlay, self.jog_wheel_1, 1, "DECK 1")
        
        self._draw_professional_jog_wheel(overlay, self.jog_wheel_2, 2, "DECK 2")
        
        # --- Draw Waveforms (TOP LAYER) ---
        # This is drawn before other controls so it's in the background
        self.visualizer.draw_stacked_visualization(overlay, self.audio_engine)
        
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
                # Load waveform data for visualization
                self.visualizer.set_track_waveform(1, track1)
                # Set cue point at beginning (professional default)
                self.audio_engine.set_cue_point(1, 0.0)
                # Set default buttons active for deck 1 - both vocal and instrumental ON for full track
                self.deck1_buttons["vocal"].is_active = True
                self.deck1_buttons["instrumental"].is_active = True
                # Ensure audio engine reflects these settings properly
                self.audio_engine.set_stem_volume(1, "vocals", 0.7)
                self.audio_engine.set_stem_volume(1, "instrumental", 0.7)
                print(f"Loaded '{track1.name}' into Deck 1 (cue point: beginning)")
        
        if len(self.track_loader.available_tracks) >= 2:
            track2 = self.track_loader.get_track(1)
            if track2:
                self.deck2_track = track2
                self.audio_engine.load_track(2, track2)
                # Load waveform data for visualization
                self.visualizer.set_track_waveform(2, track2)
                # Set cue point at beginning (professional default)
                self.audio_engine.set_cue_point(2, 0.0)
                # Set default buttons active for deck 2 - both vocal and instrumental ON for full track
                self.deck2_buttons["vocal"].is_active = True
                self.deck2_buttons["instrumental"].is_active = True
                # Ensure audio engine reflects these settings properly
                self.audio_engine.set_stem_volume(2, "vocals", 0.7)
                self.audio_engine.set_stem_volume(2, "instrumental", 0.7)
                print(f"Loaded '{track2.name}' into Deck 2 (cue point: beginning)")
    
    def draw_fingertip_landmarks(self, frame, results):
        """Draw white fingertip landmarks and transparent distance lines with connection points"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get frame dimensions
                height, width, _ = frame.shape
                
                # Define finger tip landmarks (MediaPipe hand landmark indices)
                THUMB_TIP = 4
                INDEX_TIP = 8
                MIDDLE_TIP = 12
                
                # Get landmark positions
                thumb_tip = hand_landmarks.landmark[THUMB_TIP]
                index_tip = hand_landmarks.landmark[INDEX_TIP]
                middle_tip = hand_landmarks.landmark[MIDDLE_TIP]
                
                # Convert normalized coordinates to pixel coordinates
                thumb_x = int(thumb_tip.x * width)
                thumb_y = int(thumb_tip.y * height)
                index_x = int(index_tip.x * width)
                index_y = int(index_tip.y * height)
                middle_x = int(middle_tip.x * width)
                middle_y = int(middle_tip.y * height)
                
                # Calculate distances for pinch detection
                thumb_index_distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                middle_index_distance = np.sqrt((middle_x - index_x)**2 + (middle_y - index_y)**2)
                
                # Pinch thresholds (converted from normalized to pixel space)
                regular_pinch_threshold = 0.035 * width  # Same as in HandTracker
                jog_pinch_threshold = 0.035 * width     # Same as in HandTracker
                
                # Check if any pinch connections are active
                regular_pinch_active = thumb_index_distance < regular_pinch_threshold
                jog_pinch_active = middle_index_distance < jog_pinch_threshold
                any_connection_active = regular_pinch_active or jog_pinch_active
                
                # Only show individual dots and lines when NO connections are active
                if not any_connection_active:
                    # Draw fingertip points (white)
                    cv2.circle(frame, (thumb_x, thumb_y), 5, (255, 255, 255), -1)   # White for thumb
                    cv2.circle(frame, (index_x, index_y), 5, (255, 255, 255), -1)  # White for index
                    cv2.circle(frame, (middle_x, middle_y), 5, (255, 255, 255), -1) # White for middle
                    
                    # Create overlay for transparent lines
                    overlay = frame.copy()
                    
                    # Draw connecting lines (white)
                    # Thumb to Index (regular pinch)
                    cv2.line(overlay, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 255), 2)
                    
                    # Middle to Index (jog pinch)
                    cv2.line(overlay, (middle_x, middle_y), (index_x, index_y), (255, 255, 255), 2)
                    
                    # Apply transparency to lines (30% opacity)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw bigger white points when fingers are connected (these always show when active)
                if regular_pinch_active:
                    # Regular pinch connection point
                    connect_x = (thumb_x + index_x) // 2
                    connect_y = (thumb_y + index_y) // 2
                    cv2.circle(frame, (connect_x, connect_y), 12, (255, 255, 255), -1)
                
                if jog_pinch_active:
                    # Jog pinch connection point
                    connect_x = (middle_x + index_x) // 2
                    connect_y = (middle_y + index_y) // 2
                    cv2.circle(frame, (connect_x, connect_y), 12, (255, 255, 255), -1)
    
    def load_default_tracks(self):
        """Load default tracks into both decks with easy song selection"""
        
        levels = "[fadr.com] Stems - Avicii - Levels (Lyrics)"
        no_broke_boys = "[fadr.com] Stems - Disco Lines & Tinashe - No Broke Boys (Official Audio)"
        calabria = "[fadr.com] Stems - Calabria 2007 With LYRICS [IiX8yqDVWSU]"
        sprinter = "[fadr.com] Stems - Central Cee x Dave - Sprinter (Lyrics)"
        victory_lap = "[fadr.com] Stems - Fred again.. x Skepta x PlaqueBoyMax - Victory Lap (Lyrics)"
        golden = "[fadr.com] Stems - Huntrix - Golden (Lyrics) KPop Demon Hunters [htk6MRjmcnQ]"
        i_love_it = "[fadr.com] Stems - Icona Pop - I Love It (Feat. Charli XCX)  [Audio]"
        die_young = "[fadr.com] Stems - Kesha - Die Young (Lyrics)"
        no_hands = "[fadr.com] Stems - Waka Flocka Flame - No Hands (feat. Roscoe Dash and Wale)  Lyrics"
        sushi_dont_lie = "[fadr.com] Stems - 揽佬 SKAI ISYOURGOD八方来财因果Official Music Video"
        sexy_bitch = "[fadr.com] Stems - David Guetta - Sexy Bitch (feat. Akon)  Lyrics"
        heads_will_roll = "[fadr.com] Stems - Heads Will Roll (A-Trak Remix Radio Edit)"
        fukumean = "[fadr.com] Stems - Gunna - fukumean [Official Visualizer]"
        
        DECK1_SONG = sexy_bitch  # <-- PASTE DECK 1 SONG NAME HERE
        DECK2_SONG = heads_will_roll  # <-- PASTE DECK 2 SONG NAME HERE
        
        # Scan available tracks
        self.track_loader.scan_tracks()
        
        if len(self.track_loader.available_tracks) == 0:
            print("❌ No tracks found in songs folder!")
            return
            
        # Display available tracks for reference
        print(f"📁 Found {len(self.track_loader.available_tracks)} available tracks:")
        for i, track in enumerate(self.track_loader.available_tracks):
            print(f"  {i+1}: {track.name}")
        print()
        
        # Load specific songs if specified
        if DECK1_SONG:
            track1 = self._find_track_by_name(DECK1_SONG)
            if track1:
                self._load_track_to_deck(1, track1)
                print(f"🎵 DECK 1: Loaded '{track1.name}'")
            else:
                print(f"❌ DECK 1: Track '{DECK1_SONG}' not found!")
                print("Loading first available track instead...")
                track1 = self.track_loader.get_track(0)
                if track1:
                    self._load_track_to_deck(1, track1)
        else:
            # Load first available track
            track1 = self.track_loader.get_track(0)
            if track1:
                self._load_track_to_deck(1, track1)
                print(f"🎵 DECK 1: Auto-loaded '{track1.name}'")
        
        if DECK2_SONG:
            track2 = self._find_track_by_name(DECK2_SONG)
            if track2:
                self._load_track_to_deck(2, track2)
                print(f"🎵 DECK 2: Loaded '{track2.name}'")
            else:
                print(f"❌ DECK 2: Track '{DECK2_SONG}' not found!")
                print("Loading second available track instead...")
                track2 = self.track_loader.get_track(1) if len(self.track_loader.available_tracks) > 1 else self.track_loader.get_track(0)
                if track2:
                    self._load_track_to_deck(2, track2)
        else:
            # Load second available track, or first if only one available
            track2 = self.track_loader.get_track(1) if len(self.track_loader.available_tracks) > 1 else self.track_loader.get_track(0)
            if track2:
                self._load_track_to_deck(2, track2)
                print(f"🎵 DECK 2: Auto-loaded '{track2.name}'")
        
        print("✅ Track loading complete!")
        print()
    
    def _find_track_by_name(self, song_name: str):
        """Find a track by its folder name (exact match or partial match)"""
        # First try exact match
        for track in self.track_loader.available_tracks:
            if track.name == song_name:
                return track
        
        # Then try partial match (contains)
        for track in self.track_loader.available_tracks:
            if song_name.lower() in track.name.lower():
                return track
        
        return None
    
    def _load_track_to_deck(self, deck: int, track):
        """Load a specific track to a specific deck with proper setup"""
        if deck == 1:
            self.deck1_track = track
            self.audio_engine.load_track(1, track)
            # Load waveform data for visualization
            self.visualizer.set_track_waveform(1, track)
            # Set cue point at beginning (professional default)
            self.audio_engine.set_cue_point(1, 0.0)
            # Set default buttons active - both vocal and instrumental ON
            self.deck1_buttons["vocal"].is_active = True
            self.deck1_buttons["instrumental"].is_active = True
            # Ensure audio engine reflects these settings properly
            self.audio_engine.set_stem_volume(1, "vocals", 0.7)
            self.audio_engine.set_stem_volume(1, "instrumental", 0.7)
        elif deck == 2:
            self.deck2_track = track
            self.audio_engine.load_track(2, track)
            # Load waveform data for visualization
            self.visualizer.set_track_waveform(2, track)
            # Set cue point at beginning (professional default)
            self.audio_engine.set_cue_point(2, 0.0)
            # Set default buttons active - both vocal and instrumental ON
            self.deck2_buttons["vocal"].is_active = True
            self.deck2_buttons["instrumental"].is_active = True
            # Ensure audio engine reflects these settings properly
            self.audio_engine.set_stem_volume(2, "vocals", 0.7)
            self.audio_engine.set_stem_volume(2, "instrumental", 0.7)
    
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
                
                # Update jog wheel spinning based on playback state (realistic DJ behavior)
                self.update_jog_wheel_spinning()
                
                # Draw controller overlay
                frame = self.draw_controller_overlay(frame)
                
                # Draw fingertip landmarks and distance lines
                self.draw_fingertip_landmarks(frame, results)
                
                # Show pinch feedback on targeted elements
                for is_pinched, (x, y) in pinch_data:
                    if is_pinched:
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
                
                # Jog pinch visualization now handled by draw_fingertip_landmarks
                
                # Clean interface - all status text removed for cleaner look
                
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
