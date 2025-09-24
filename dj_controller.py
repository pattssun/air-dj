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

    # RotaryKnob class removed - no longer needed without EQ knobs

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
    album_artwork: Optional[str] = None  # Path to album artwork PNG

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
        self.visible_seconds = 8.0  # Zoomed-in view for precise beat analysis
        self.center_line_thickness = 2  # Prominent center playhead

        # --- Professional Rekordbox Colors ---
        self.bg_color = (20, 20, 20)  # Very dark gray (not pure black)
        
        # Enhanced frequency band colors for better distinction
        self.low_freq_color = (50, 150, 255)     # Deeper blue for bass - more contrast
        self.mid_freq_color = (255, 140, 60)     # Richer orange for mids - more distinct
        self.high_freq_color = (255, 255, 255)   # Pure white for highs - maximum contrast
        
        # Grid and UI colors - PRIORITY: Make beat/bar lines highly visible
        self.beat_color = (120, 120, 120)       # Very bright beat lines - PRIORITY
        self.bar_color = (200, 200, 200)        # ULTRA-bright individual bar lines - PRIORITY  
        self.major_bar_color = (255, 255, 255)  # Pure white phrase markers - MAXIMUM PRIORITY
        self.playhead_color = (255, 80, 80)     # Bright red center playhead
        self.playhead_shadow = (150, 40, 40)    # Stronger shadow for depth
        self.bpm_color = (100, 200, 255)        # Blue for BPM display
        self.text_color = (220, 220, 220)       # Brighter text
        
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
        # Calculate compact layout with minimal dead space
        margin = 10  # Minimal margins for compact view
        viz_width = self.screen_width - (2 * margin)
        viz_start_x = margin
        center_x = self.screen_width // 2
        
        # Calculate minimal visualization area height
        total_viz_height = (self.track_height * 2) + self.track_spacing + 20  # Minimal space for tracks
        viz_start_y = 10  # Minimal top margin
        
        # Draw compact background panel
        bg_rect = (viz_start_x - 5, viz_start_y - 5, 
                   viz_width + 10, total_viz_height + 10)
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                     self.bg_color, -1)
        
        # Draw subtle border around visualization area
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                     (50, 50, 50), 1)
        
        # Draw Deck 1 (top) with container border for height verification
        deck1_y = viz_start_y
        
        # Draw track container border for Deck 1 (visual height confirmation)
        cv2.rectangle(overlay, (viz_start_x, deck1_y), 
                     (viz_start_x + viz_width, deck1_y + self.track_height), 
                     (40, 40, 40), 1)
        
        self._draw_deck_visualization(
            overlay, 1, viz_start_x, deck1_y, viz_width, center_x,
            self.deck1_waveform, audio_engine
        )
        
        # Draw separator line between decks
        separator_y = deck1_y + self.track_height + (self.track_spacing // 2)
        cv2.line(overlay, (viz_start_x, separator_y), 
                (viz_start_x + viz_width, separator_y), (60, 60, 60), 1)
        
        # Draw Deck 2 (bottom) with container border for height verification
        deck2_y = deck1_y + self.track_height + self.track_spacing
        
        # Draw track container border for Deck 2 (visual height confirmation)
        cv2.rectangle(overlay, (viz_start_x, deck2_y), 
                     (viz_start_x + viz_width, deck2_y + self.track_height), 
                     (40, 40, 40), 1)
        
        self._draw_deck_visualization(
            overlay, 2, viz_start_x, deck2_y, viz_width, center_x,
            self.deck2_waveform, audio_engine
        )
        
        # Draw central playhead line across both decks (most prominent feature)
        playhead_x = center_x
        playhead_top = deck1_y
        playhead_bottom = deck2_y + self.track_height
        
        # Draw wider shadow for more depth
        cv2.line(overlay, (playhead_x + 2, playhead_top), 
                (playhead_x + 2, playhead_bottom), self.playhead_shadow, 4)
        
        # Draw main playhead line - thicker and more prominent
        cv2.line(overlay, (playhead_x, playhead_top), 
                (playhead_x, playhead_bottom), self.playhead_color, 3)
        
        # Add bright center highlight
        cv2.line(overlay, (playhead_x, playhead_top), 
                (playhead_x, playhead_bottom), (255, 200, 200), 1)
        
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
        
        # Get tempo multiplier for BPM sync visualization
        tempo_multiplier = audio_engine.get_tempo_multiplier(deck_num)
        
        # Calculate current time accounting for BPM sync
        # When tempo is faster (>1.0), we're further in the original track
        # When tempo is slower (<1.0), we're behind in the original track
        current_time_sec = position_ratio * waveform_data.duration
        
        is_playing = (audio_engine.deck1_is_playing if deck_num == 1 
                     else audio_engine.deck2_is_playing)
        
        # Get track for BPM and name
        track = audio_engine.deck1_track if deck_num == 1 else audio_engine.deck2_track
        track_name = track.name if track else "Unknown Track"
        track_bpm = track.bpm if track else 120.0

        # --- Draw Enhanced Waveform ---
        self._draw_scrolling_waveform(overlay, x, y, width, center_x,
                                      waveform_data, current_time_sec, tempo_multiplier)

        # --- Draw Professional Track Info ---
        # Track name (top left) - positioned inside track container
        truncated_name = track_name[:35] + "..." if len(track_name) > 35 else track_name
        cv2.putText(overlay, f"DECK {deck_num}: {truncated_name}", (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # BPM display removed for cleaner visualization
        
        # Time position display (bottom right) - positioned inside track container
        minutes = int(current_time_sec // 60)
        seconds = int(current_time_sec % 60)
        time_text = f"{minutes:02d}:{seconds:02d}"
        time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        time_x = x + width - time_size[0] - 10
        cv2.putText(overlay, time_text, (time_x, y + self.track_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Play/pause indicator removed for cleaner visualization

    def _draw_scrolling_waveform(self, overlay, x: int, y: int, width: int, center_x: int,
                                 waveform_data: WaveformData, current_time: float, tempo_multiplier: float = 1.0):
        """Draws professional Rekordbox-style stereo waveform with enhanced beat/bar grid."""
        
        # --- Calculate Precise Audio Window ---
        seconds_per_pixel = self.visible_seconds / width
        start_time = current_time - (center_x - x) * seconds_per_pixel
        end_time = current_time + (x + width - center_x) * seconds_per_pixel
        time_to_pixel = lambda t: int(x + (t - start_time) / seconds_per_pixel)

        # Waveform area setup
        waveform_center_y = y + self.track_height // 2
        max_amplitude = self.waveform_height // 2
        
        # --- PRIORITY: Draw Beat and Bar Grid Lines (ALWAYS VISIBLE) ---
        # Debug: Show timeline coverage for 8-second view
        timeline_info = f"Timeline: {start_time:.1f}s to {end_time:.1f}s ({end_time - start_time:.1f}s visible)"
        
        # Draw beat lines first (behind bars) - MAKE THESE ALWAYS SHOW
        # Scale beat times by tempo multiplier for BPM sync visualization
        beat_lines_drawn = 0
        for beat_time in waveform_data.beat_times:
            # Adjust beat time for BPM sync - slower tempo spreads beats out, faster tempo compresses them
            # If tempo_multiplier < 1.0 (slower), beats appear later (spread out)
            # If tempo_multiplier > 1.0 (faster), beats appear earlier (compressed)
            synced_beat_time = beat_time * tempo_multiplier
            if start_time <= synced_beat_time <= end_time:
                px = time_to_pixel(synced_beat_time)
                # PRIORITY: Ultra-prominent beat lines extending full height
                cv2.line(overlay, (px, y + 5), (px, y + self.track_height - 5), 
                        self.beat_color, 3)  # Extra thick beat lines - PRIORITY
                beat_lines_drawn += 1
        
        # Draw EVERY INDIVIDUAL bar line - MAKE EACH BAR VISIBLE
        bar_count = 0
        bar_lines_drawn = 0
        for bar_time in waveform_data.bar_times:
            # Adjust bar time for BPM sync - same scaling as beats
            synced_bar_time = bar_time * tempo_multiplier
            if start_time <= synced_bar_time <= end_time:
                px = time_to_pixel(synced_bar_time)
                # Every 4th bar gets ultra prominence (phrase markers)
                if bar_count % 4 == 0:
                    # PRIORITY: Major phrase markers - maximum visibility
                    cv2.line(overlay, (px, y - 2), (px, y + self.track_height + 2), 
                            self.major_bar_color, 4)  # Thickest lines extending beyond track
                else:
                    # PRIORITY: EVERY individual bar marker - highly visible
                    cv2.line(overlay, (px, y), (px, y + self.track_height), 
                            self.bar_color, 3)  # Thick bar lines full height for EVERY bar
                bar_count += 1
                bar_lines_drawn += 1
        
        # PRIORITY: Always show grid status (for debugging visibility)
        if beat_lines_drawn == 0 and bar_lines_drawn == 0:
            # If no lines drawn, draw emergency grid markers every second
            for i in range(int(start_time), int(end_time) + 1):
                px = time_to_pixel(float(i))
                if x <= px <= x + width:
                    cv2.line(overlay, (px, y + 20), (px, y + self.track_height - 20), 
                            (255, 0, 0), 2)  # Red emergency grid
        
        # Display timeline info for 8-second confirmation
        if hasattr(self, '_timeline_debug_counter'):
            self._timeline_debug_counter += 1
        else:
            self._timeline_debug_counter = 1
        
        # Show timeline info every 2 seconds to confirm 8-second view
        if self._timeline_debug_counter % 120 == 0:  # Every 2 seconds
            total_visible = end_time - start_time
            print(f"📊 8-Second View: Showing {total_visible:.1f}s | Beats: {beat_lines_drawn} | Bars: {bar_lines_drawn}")

        # --- Draw Stereo-Style Multi-Band Waveforms with Better Separation ---
        # Adjust waveform timing for BPM sync - same scaling as beat grid
        synced_start_time = start_time / tempo_multiplier
        synced_seconds_per_pixel = seconds_per_pixel / tempo_multiplier
        synced_duration = waveform_data.duration / tempo_multiplier
        
        # Bass (bottom layer, full amplitude for strong visual impact)
        self._render_stereo_waveform_band(overlay, width, x, waveform_center_y, synced_start_time, 
                                         synced_seconds_per_pixel, synced_duration, 
                                         waveform_data.low_freq_peaks, self.low_freq_color, 
                                         max_amplitude * 0.9, alpha=0.85)
        
        # Mids (middle layer, distinct sizing)
        self._render_stereo_waveform_band(overlay, width, x, waveform_center_y, synced_start_time, 
                                         synced_seconds_per_pixel, synced_duration, 
                                         waveform_data.mid_freq_peaks, self.mid_freq_color, 
                                         max_amplitude * 0.65, alpha=0.95)
        
        # Highs (top layer, smaller but very bright for clarity)
        self._render_stereo_waveform_band(overlay, width, x, waveform_center_y, synced_start_time, 
                                         synced_seconds_per_pixel, synced_duration, 
                                         waveform_data.high_freq_peaks, self.high_freq_color, 
                                         max_amplitude * 0.4, alpha=1.0)

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
        
        # EQ controls removed for cleaner interface
        
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
                    buffersize=256,  # Reduced from 512 for ultra-low latency
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
                    buffersize=256,  # Reduced from 512 for ultra-low latency
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
            pass  # Silently try basic configuration
            
            # Very basic fallback
            try:
                self.server = Server(sr=44100, nchnls=2)
                self.server.boot()
                self.server.start()
                pass  # Silently start basic audio
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
                pass  # Silently create master player
            
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
                        pass  # Silently load stem
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
            pass  # Silently initialize volumes
            
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
    
    # EQ methods removed for cleaner interface
    
    
    
    
    
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
            min_detection_confidence=0.5,  # Reduced for faster detection
            min_tracking_confidence=0.3   # Reduced for faster tracking
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_width = 1920
        self.frame_height = 1080
        self.pinch_history = []  # Minimal smoothing for instant response
        
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
        
        # Optimized pinch threshold for ultra-responsive DJ control
        pinch_threshold = 0.04  # Balanced for instant response while avoiding false triggers
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
        
        # Jog wheel pinch threshold optimized for instant scratching response
        jog_pinch_threshold = 0.04  # Increased for faster jog wheel response
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
        
        pass  # Silently scan tracks
    
    def _parse_stem_folder(self, folder_name: str, folder_path: str) -> Optional[Track]:
        """Parse a stem folder and create a Track object"""
        # Look for stem files and album artwork
        stems = {}
        bpm = 120  # Default BPM
        key = "C"  # Default key
        album_artwork = None
        
        # Expected stem types
        stem_types = ["vocals", "instrumental", "drums", "bass", "other"]
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
                
            # Check for album artwork (PNG files)
            if file.lower().endswith('.png'):
                if album_artwork is None:  # Use first PNG found
                    album_artwork = file_path
                    print(f"🎨 Found album artwork for '{folder_name}': {file}")
            
            # Check for stem files
            elif file.lower().endswith(('.mp3', '.wav')):
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
                stems=stems,
                album_artwork=album_artwork
            )
        return None
    
    def get_track(self, index: int) -> Optional[Track]:
        """Get track by index"""
        if 0 <= index < len(self.available_tracks):
            return self.available_tracks[index]
        return None

class DJController:
    """Main DJ Controller class with transparent overlay"""
    
    def __init__(self, enable_bpm_sync=True, selected_songs=None):
        # BPM sync configuration
        self.enable_bpm_sync = enable_bpm_sync
        
        # Store selected songs
        self.selected_songs = selected_songs
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.audio_engine = AudioEngine()
        self.track_loader = TrackLoader()
        
        # Controller layout configuration
        self.screen_width = 1920
        self.screen_height = 1080
        self.overlay_alpha = 0.8
        
        # Initialize Rekordbox-style visualization
        self.visualizer = RekordboxStyleVisualizer(self.screen_width, self.screen_height)
        
        # Initialize controller elements
        self.setup_controller_layout()
        
        # Load UI images
        self.cue_image = cv2.imread('ui/Cue.png', cv2.IMREAD_UNCHANGED)
        self.play_pause_image = cv2.imread('ui/Play:Pause.png', cv2.IMREAD_UNCHANGED)
        self.jogwheel_image = cv2.imread('ui/jogwheel2.png', cv2.IMREAD_UNCHANGED)
        
        # Resize jogwheel to 500px diameter if loaded successfully
        if self.jogwheel_image is not None:
            self.jogwheel_image = cv2.resize(self.jogwheel_image, (500, 500))
        
        # Verify images loaded successfully
        if self.cue_image is None:
            print("Warning: Failed to load ui/Cue.png")
        if self.play_pause_image is None:
            print("Warning: Failed to load ui/Play:Pause.png")
        if self.jogwheel_image is None:
            print("Warning: Failed to load ui/jogwheel2.png")
        
        # State
        self.current_pinches = []
        self.deck1_track = None
        self.deck2_track = None
        
        # Jog wheel rotation states
        self.deck1_jog_rotation = 0.0  # Current rotation angle in degrees
        self.deck2_jog_rotation = 0.0  # Current rotation angle in degrees
        
        # Album artwork rotation states (separate from jog wheel rotation)
        self.deck1_artwork_rotation = 0.0  # Album artwork rotation in degrees
        self.deck2_artwork_rotation = 0.0  # Album artwork rotation in degrees
        
        # Album artwork images (loaded when tracks are loaded)
        self.deck1_artwork_image = None
        self.deck2_artwork_image = None
        
        # Slider interaction state - for intuitive pinch-to-grab behavior
        self.active_slider = None  # Which slider is currently grabbed
        self.active_slider_pos = None # The (x, y) position of the pinch grabbing the slider
        self.slider_grab_offset = 0  # Offset from slider position when grabbed
        
        # Jog wheel interaction state - for scratching and track navigation
        self.active_jog = None  # Which jog wheel is currently grabbed (1 or 2)
        self.jog_initial_angle = 0.0  # Initial touch angle for rotation tracking
        self.jog_last_angle = 0.0  # Last touch angle for calculating rotation
        self.jog_rotation_speed = 0.0  # Current rotation speed for scratching
        
        # Video capture with DJ-optimized camera wrapper
        pass  # Silently set up camera
        try:
            from iphone_camera_integration import create_dj_camera
            self.dj_camera = create_dj_camera()
            # For compatibility with existing code that expects self.cap
            self.cap = self.dj_camera.cap
        except ImportError:
            print("⚠️  DJ Camera module not available, using basic setup")
            self.cap = cv2.VideoCapture(0)
            self.dj_camera = None
            
            # Basic camera configuration if wrapper not available
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    def setup_controller_layout(self):
        """Setup the DJ controller layout matching the screenshot"""
        # Center the layout - calculate positions relative to screen center
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Jog wheels - 500px diameter (250px radius), positioned at top corners with 100px border
        # Position jog wheels at top corners: 100px from edges + 250px radius = 350px from edge centers
        jog_wheel_center_x_left = 100 + 250  # 350px from left edge
        jog_wheel_center_x_right = self.screen_width - 100 - 250  # 350px from right edge  
        jog_wheel_center_y = 100 + 250  # 350px from top edge
        
        self.jog_wheel_1 = JogWheel("jog1", jog_wheel_center_x_left, jog_wheel_center_y, 250)
        self.jog_wheel_2 = JogWheel("jog2", jog_wheel_center_x_right, jog_wheel_center_y, 250)
        
        # SWAPPED LAYOUT: Pads closer to edges, cue/play buttons closer to center
        
        # Calculate positions for 2x2 pad grids - 150x150px pads with 20px separation
        # Pads now positioned closer to screen edges
        pad_size = 150
        pad_separation = 20
        
        # Grid dimensions: 2x2 with 150px pads and 20px separation
        grid_width = pad_size + pad_separation + pad_size  # 320px
        grid_height = pad_size + pad_separation + pad_size  # 320px
        
        # Bottom row y-position (100px from bottom)
        bottom_pad_y = self.screen_height - 100 - pad_size
        # Top row y-position
        top_pad_y = bottom_pad_y - pad_separation - pad_size
        
        # Adjusted positions: cue/play buttons 100px towards edges, pads 20px towards center
        
        # Left deck pads - moved 20px towards center from previous position
        left_grid_x = 125 + 20  # 145px from left edge (20px towards center)
        left_col1_x = left_grid_x
        left_col2_x = left_grid_x + pad_size + pad_separation  # 145 + 150 + 20 = 315px
        
        # Right deck pads - moved 20px towards center from previous position
        right_grid_end_x = self.screen_width - 125 - 20  # screen_width - 145px (20px towards center)
        right_col2_x = right_grid_end_x - pad_size  # Rightmost pad
        right_col1_x = right_col2_x - pad_separation - pad_size  # Left pad
        
        # Calculate circular button positions - fine-tuned: 100px towards edges, then 35px back towards center
        play_pause_center_x_left = 635 - 100 + 35  # 570px from left edge (100px towards edge, then 35px back towards center)
        play_pause_center_x_right = self.screen_width - 635 + 100 - 35  # 1350px from left edge (100px towards edge, then 35px back towards center)
        play_pause_center_y = self.screen_height - 100 - 75  # 175px from bottom edge
        
        # Cue buttons 20px above play/pause (center-to-center distance = 20 + 75 + 75 = 170px)
        cue_center_y = play_pause_center_y - 170
        
        # Buttons for Deck 1 (left side) - circular buttons + 2x2 pad grid
        self.deck1_buttons = {
            "cue": ControllerButton("Cue", play_pause_center_x_left - 75, cue_center_y - 75, 150, 150, button_type="momentary"),
            "play_pause": ControllerButton("Play/Pause", play_pause_center_x_left - 75, play_pause_center_y - 75, 150, 150, button_type="toggle"),
            # Top row (clickable) - SWAPPED: vocal/instrumental now on top
            "vocal": ControllerButton("Vocal", left_col1_x, top_pad_y, pad_size, pad_size, button_type="toggle"),
            "instrumental": ControllerButton("Instrumental", left_col2_x, top_pad_y, pad_size, pad_size, button_type="toggle"),
            # Bottom row (non-clickable, just for show) - SWAPPED: display pads now on bottom
            "pad_bottom_left": ControllerButton("PAD 1", left_col1_x, bottom_pad_y, pad_size, pad_size, button_type="display"),
            "pad_bottom_right": ControllerButton("PAD 2", left_col2_x, bottom_pad_y, pad_size, pad_size, button_type="display")
        }
        
        # Buttons for Deck 2 (right side) - circular buttons + 2x2 pad grid
        self.deck2_buttons = {
            "cue": ControllerButton("Cue", play_pause_center_x_right - 75, cue_center_y - 75, 150, 150, button_type="momentary"),
            "play_pause": ControllerButton("Play/Pause", play_pause_center_x_right - 75, play_pause_center_y - 75, 150, 150, button_type="toggle"),
            # Top row (clickable) - SWAPPED: vocal/instrumental now on top, matching left side
            "vocal": ControllerButton("Vocal", right_col1_x, top_pad_y, pad_size, pad_size, button_type="toggle"),
            "instrumental": ControllerButton("Instrumental", right_col2_x, top_pad_y, pad_size, pad_size, button_type="toggle"),
            # Bottom row (non-clickable, just for show) - SWAPPED: display pads now on bottom
            "pad_bottom_left": ControllerButton("PAD 3", right_col1_x, bottom_pad_y, pad_size, pad_size, button_type="display"),
            "pad_bottom_right": ControllerButton("PAD 4", right_col2_x, bottom_pad_y, pad_size, pad_size, button_type="display")
        }
        
        # Center controls (crossfader only, no effects)
        self.center_buttons = {}
        
        # Calculate slider positions: aligned with inner edges of performance pads for better symmetry
        # Left deck: align to right edge of right column (instrumental pad)
        # Right deck: align to left edge of right column (vocal pad) minus slider width
        tempo_slider_x_left = left_col2_x + pad_size  # Right edge of left deck's right column
        tempo_slider_x_right = right_col2_x - 30  # Left edge of right deck's right column minus slider width
        tempo_y = 100  # At top 100px border
        volume_y = tempo_y + 280 + 100  # 100px below tempo sliders (tempo_y + tempo_height + gap)
        
        # Tempo controls - positioned at top border, aligned with inner pad edges
        self.tempo_fader_1 = Fader("Tempo1", tempo_slider_x_left, tempo_y, 30, 280, value=0.5)
        self.tempo_fader_2 = Fader("Tempo2", tempo_slider_x_right, tempo_y, 30, 280, value=0.5)
        
        # Volume faders - middle point aligned with middle gap between pads, height matches pad grid
        volume_fader_x_left = jog_wheel_center_x_left + 250 + 100  # 100px to the right of left jog wheel
        volume_fader_x_right = jog_wheel_center_x_right - 250 - 100 - 30  # 100px to the left of right jog wheel (minus slider width)
        
        # Calculate middle of gap between top and bottom pads
        gap_middle_y = bottom_pad_y - pad_separation // 2  # Middle of 20px gap
        # Height covers both pads + gap = 150 + 20 + 150 = 320px
        volume_fader_height = pad_size + pad_separation + pad_size  # 320px
        volume_fader_y = gap_middle_y - volume_fader_height // 2  # Center slider on gap middle
        
        self.volume_fader_1 = Fader("Vol1", volume_fader_x_left, volume_fader_y, 30, volume_fader_height, value=1.0)
        self.volume_fader_2 = Fader("Vol2", volume_fader_x_right, volume_fader_y, 30, volume_fader_height, value=1.0)
        
        # Crossfader - 2/3 original width (267px), centered horizontally, aligned with pad gap
        crossfader_width = int(400 * 2 / 3)  # 267px (2/3 of original)
        crossfader_y = gap_middle_y  # Align with middle of gap between pads
        self.crossfader = Fader("Crossfader", center_x - crossfader_width // 2, crossfader_y, crossfader_width, 30, value=0.5)
        
        # EQ and effects knobs removed - cleaner DJ controller layout
    
    def handle_button_interaction(self, button: ControllerButton, deck: int = 0):
        """Handle button press interactions"""
        if button.name == "Cue":
            if deck == 1:
                self.audio_engine.cue_deck(1)
                button.is_active = True
                # Reset jog wheel rotation when cued
                self.deck1_jog_rotation = 0.0
                self.jog_wheel_1.current_angle = 0.0
                # Reset album artwork rotation to default position (0 degrees)
                self.deck1_artwork_rotation = 0.0
            elif deck == 2:
                self.audio_engine.cue_deck(2)
                button.is_active = True
                # Reset jog wheel rotation when cued
                self.deck2_jog_rotation = 0.0
                self.jog_wheel_2.current_angle = 0.0
                # Reset album artwork rotation to default position (0 degrees)
                self.deck2_artwork_rotation = 0.0
        
        elif button.name == "Play/Pause":
            if button.button_type == "toggle":
                button.is_active = not button.is_active
                
                # Industry Standard: If no stems are active, turn them on when playing
                active_stems = self.audio_engine.deck1_active_stems if deck == 1 else self.audio_engine.deck2_active_stems
                if button.is_active and not any(active_stems.values()):
                    print(f"Deck {deck}: No active stems. Activating Vocal/Instrumental for playback.")
                    self.audio_engine.set_stem_volume(deck, "vocals", 1.0)
                    self.audio_engine.set_stem_volume(deck, "instrumental", 1.0)
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
                volume = 1.0 if button.is_active else 0.0
                self.audio_engine.set_stem_volume(deck, "vocals", volume)
                print(f"Deck {deck} vocals: {'ON' if button.is_active else 'OFF'}")
        
        elif button.name == "Instrumental":
            if button.button_type == "toggle":
                button.is_active = not button.is_active
                # Toggle instrumental stem volume
                volume = 1.0 if button.is_active else 0.0
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
        """Check if coordinates collide with button with enhanced hit area for better reliability"""
        # Skip display-only buttons (non-clickable)
        if button.button_type == "display":
            return False
        
        # Determine if we're in an edge area for better tolerance  
        edge_margin = 200
        in_edge_area = (x < edge_margin or x > self.screen_width - edge_margin or 
                       y < edge_margin or y > self.screen_height - edge_margin)
            
        if button.name in ["Cue", "Play/Pause"]:
            # Enhanced circular collision detection for cue and play/pause buttons
            center_x = button.x + 75  # radius = 75px for 150px diameter
            center_y = button.y + 75
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            
            # Expanded radius with edge area compensation
            if in_edge_area:
                expanded_radius = 100  # 75 + 25px extra in edge areas
            else:
                expanded_radius = 85   # 75 + 10px in center areas
                
            return distance <= expanded_radius
        else:
            # Enhanced rectangular collision detection with margins
            if in_edge_area:
                margin = 25  # Large margin in edge areas
            else:
                margin = 15  # Moderate margin in center areas
                
            return (button.x - margin <= x <= button.x + button.width + margin and 
                    button.y - margin <= y <= button.y + button.height + margin)
    
    def check_button_collision_expanded(self, x: int, y: int, button: ControllerButton) -> bool:
        """Check if coordinates collide with button using greatly expanded hit area for maximum reliability"""
        # Skip display-only buttons (non-clickable)
        if button.button_type == "display":
            return False
        
        # Determine if we're in an edge/corner area where tracking is less reliable
        edge_margin = 200  # Consider 200px from edges as "difficult tracking area"
        in_edge_area = (x < edge_margin or x > self.screen_width - edge_margin or 
                       y < edge_margin or y > self.screen_height - edge_margin)
        
        if button.name in ["Cue", "Play/Pause"]:
            # Greatly expanded circular collision detection for cue and play/pause buttons
            center_x = button.x + 75  # radius = 75px for 150px diameter
            center_y = button.y + 75
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            
            # Much larger hit areas - especially in edge areas
            if in_edge_area:
                expanded_radius = 110  # 75 + 35px extra margin in edge areas
            else:
                expanded_radius = 95   # 75 + 20px margin in center areas
            
            return distance <= expanded_radius
        else:
            # Greatly expanded rectangular collision detection for pads
            if in_edge_area:
                margin = 30  # Large margin in edge areas for pads
            else:
                margin = 20  # Standard large margin for pads
                
            return (button.x - margin <= x <= button.x + button.width + margin and 
                    button.y - margin <= y <= button.y + button.height + margin)
    
    def check_fader_collision(self, x: int, y: int, fader: Fader) -> bool:
        """Check if coordinates collide with fader (greatly expanded area for maximum reliability)"""
        # Determine if we're in an edge area for extra tolerance
        edge_margin = 200
        in_edge_area = (x < edge_margin or x > self.screen_width - edge_margin or 
                       y < edge_margin or y > self.screen_height - edge_margin)
        
        # Much larger margins for better interaction, especially at edges
        if in_edge_area:
            margin = 35  # Extra large margin in edge areas
        else:
            margin = 25  # Large margin in center areas
            
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
        """Check if coordinates are within the crossfader's draggable area (greatly expanded for reliability)"""
        # Determine if we're in an edge area for extra tolerance
        edge_margin = 200
        in_edge_area = (x < edge_margin or x > self.screen_width - edge_margin or 
                       y < edge_margin or y > self.screen_height - edge_margin)
        
        # Much larger margins for better interaction, especially at edges
        if in_edge_area:
            margin = 35  # Extra large margin in edge areas
        else:
            margin = 25  # Large margin in center areas
            
        return (self.crossfader.x - margin <= x <= self.crossfader.x + self.crossfader.width + margin and
                self.crossfader.y - margin <= y <= self.crossfader.y + self.crossfader.height + margin)
    
    def create_track_visualization_window(self):
        """Create separate window for track visualization with same width as webcam window"""
        # Create compact image with minimal height for visualization
        viz_height = 240  # Compact height matching actual content needs
        viz_frame = np.zeros((viz_height, self.screen_width, 3), dtype=np.uint8)
        
        # Draw the visualization on this frame
        self.visualizer.draw_stacked_visualization(viz_frame, self.audio_engine)
        
        return viz_frame
    
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
        """Enhanced multi-touch system - supports simultaneous interactions with multiple controls"""
        # Store previous button states
        prev_pressed_states = {}
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button_name, button in buttons.items():
                prev_pressed_states[id(button)] = button.is_pressed
        
        # Reset all button pressed states
        for buttons in [self.deck1_buttons, self.deck2_buttons, self.center_buttons]:
            for button in buttons.values():
                button.is_pressed = False
        
        # --- Enhanced Multi-Touch System ---
        active_pinches = [(x, y) for is_pinched, (x, y) in pinch_data if is_pinched]
        used_pinches = set()  # Track which pinches have been assigned
        
        # Initialize active controls tracking if not exists
        if not hasattr(self, 'active_controls'):
            self.active_controls = {}  # Maps control_id -> (control_type, control_object, position)
        
        # Step 1: Update all existing active controls
        controls_to_remove = []
        for control_id, (control_type, control_obj, last_pos) in list(self.active_controls.items()):
            # Find the closest available pinch to continue this interaction
            available_pinches = [p for p in active_pinches if p not in used_pinches]
            if available_pinches:
                closest_pinch = self._find_closest_pinch(last_pos, available_pinches)
                if closest_pinch and self._distance(closest_pinch, last_pos) < 100:  # Within reasonable range
                    x, y = closest_pinch
                    used_pinches.add(closest_pinch)
                    
                    # Update the control
                    if control_type == 'slider':
                        self._update_slider_by_name(control_obj, x, y)
                        self.active_controls[control_id] = (control_type, control_obj, (x, y))
                    elif control_type == 'knob':
                        self._update_knob_by_object(control_obj, x, y)
                        self.active_controls[control_id] = (control_type, control_obj, (x, y))
                else:
                    # No close pinch found, release this control
                    controls_to_remove.append(control_id)
            else:
                # No available pinches, release this control
                controls_to_remove.append(control_id)
        
        # Remove controls that no longer have pinches
        for control_id in controls_to_remove:
            del self.active_controls[control_id]

        # Step 2: Handle new interactions with remaining pinches
        remaining_pinches = [p for p in active_pinches if p not in used_pinches]
        
        for x, y in remaining_pinches:
            interaction_handled = False
            
            # Check for new slider interactions (multiple sliders can be active simultaneously)
            if self.check_fader_collision(x, y, self.volume_fader_1):
                control_id = f"volume_fader_1_{x}_{y}"
                self.active_controls[control_id] = ('slider', 'volume_fader_1', (x, y))
                self._update_slider_by_name('volume_fader_1', x, y)
                interaction_handled = True
                used_pinches.add((x, y))
            elif self.check_fader_collision(x, y, self.volume_fader_2):
                control_id = f"volume_fader_2_{x}_{y}"
                self.active_controls[control_id] = ('slider', 'volume_fader_2', (x, y))
                self._update_slider_by_name('volume_fader_2', x, y)
                interaction_handled = True
                used_pinches.add((x, y))
            elif self.check_crossfader_collision(x, y):
                control_id = f"crossfader_{x}_{y}"
                self.active_controls[control_id] = ('slider', 'crossfader', (x, y))
                self._update_slider_by_name('crossfader', x, y)
                interaction_handled = True
                used_pinches.add((x, y))
            
            # DISABLED: Check for tempo sliders - interaction removed but code preserved
            # if not interaction_handled and self.check_fader_collision(x, y, self.tempo_fader_1):
            #     control_id = f"tempo_fader_1_{x}_{y}"
            #     self.active_controls[control_id] = ('slider', 'tempo_fader_1', (x, y))
            #     self._update_slider_by_name('tempo_fader_1', x, y)
            #     interaction_handled = True
            #     used_pinches.add((x, y))
            # elif not interaction_handled and self.check_fader_collision(x, y, self.tempo_fader_2):
            #     control_id = f"tempo_fader_2_{x}_{y}"
            #     self.active_controls[control_id] = ('slider', 'tempo_fader_2', (x, y))
            #     self._update_slider_by_name('tempo_fader_2', x, y)
            #     interaction_handled = True
            #     used_pinches.add((x, y))
            
            # EQ knob interactions removed for cleaner interface
            
            # Check for button presses if nothing else handled
            if not interaction_handled:
                # Check all button groups for simultaneous button presses
                for buttons, deck_name in [(self.deck1_buttons, "Deck 1"), (self.deck2_buttons, "Deck 2"), (self.center_buttons, "Center")]:
                    for button in buttons.values():
                        if self.check_button_collision_expanded(x, y, button):
                            button.is_pressed = True
                            # Trigger button action if this is a new press
                            if not prev_pressed_states.get(id(button), False):
                                deck = 1 if deck_name == "Deck 1" else (2 if deck_name == "Deck 2" else 0)
                                self.handle_button_interaction(button, deck)
                            interaction_handled = True
                            used_pinches.add((x, y))
                            break
                    if interaction_handled:
                        break
        
        # Step 3: Handle jog wheel interactions (separate pinch type)
        self._process_jog_wheel_interactions(jog_pinch_data)
        
        # Step 4: Clean up - release all controls if no pinches are active
        if not active_pinches:
            # Clear all active controls
            for control_id, (control_type, control_obj, _) in self.active_controls.items():
                if control_type == 'slider':
                    # Find the actual fader object and stop dragging
                    if isinstance(control_obj, str):
                        if control_obj == 'volume_fader_1':
                            self.volume_fader_1.is_dragging = False
                        elif control_obj == 'volume_fader_2':
                            self.volume_fader_2.is_dragging = False
                        elif control_obj == 'crossfader':
                            self.crossfader.is_dragging = False
                        # DISABLED: Tempo fader cleanup - interaction removed but code preserved  
                        # elif control_obj == 'tempo_fader_1':
                        #     self.tempo_fader_1.is_dragging = False
                        # elif control_obj == 'tempo_fader_2':
                        #     self.tempo_fader_2.is_dragging = False
                elif control_type == 'knob':
                    control_obj.is_dragging = False
            
            self.active_controls.clear()
    
    def _process_jog_wheel_interactions(self, jog_pinch_data):
        """Handle jog wheel interactions with separate pinch gesture"""
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
    
    def _distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    def _update_slider_by_name(self, slider_name, x, y):
        """Update a slider by its name - supports multiple simultaneous sliders"""
        if slider_name == 'volume_fader_1':
            fader = self.volume_fader_1
            relative_y = (y - fader.y) / fader.height
            fader_value = max(0.0, min(1.0, 1.0 - relative_y))
            fader.value = fader_value
            fader.is_dragging = True
            self.audio_engine.set_master_volume(1, fader_value)
            
        elif slider_name == 'volume_fader_2':
            fader = self.volume_fader_2
            relative_y = (y - fader.y) / fader.height
            fader_value = max(0.0, min(1.0, 1.0 - relative_y))
            fader.value = fader_value
            fader.is_dragging = True
            self.audio_engine.set_master_volume(2, fader_value)
            
        elif slider_name == 'crossfader':
            fader = self.crossfader
            relative_x = (x - fader.x) / fader.width
            crossfader_value = max(0.0, min(1.0, relative_x))
            fader.value = crossfader_value
            fader.is_dragging = True
            self.audio_engine.set_crossfader_position(crossfader_value)
            
        # DISABLED: Tempo fader processing - interaction removed but code preserved
        # elif slider_name == 'tempo_fader_1':
        #     fader = self.tempo_fader_1
        #     relative_y = (y - fader.y) / fader.height
        #     fader_value = max(0.0, min(1.0, 1.0 - relative_y))
        #     fader.value = fader_value
        #     fader.is_dragging = True
        #     self.audio_engine.set_tempo(1, fader_value)
        #     
        # elif slider_name == 'tempo_fader_2':
        #     fader = self.tempo_fader_2
        #     relative_y = (y - fader.y) / fader.height
        #     fader_value = max(0.0, min(1.0, 1.0 - relative_y))
        #     fader.value = fader_value
        #     fader.is_dragging = True
        #     self.audio_engine.set_tempo(2, fader_value)
    

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
        # DISABLED: Tempo fader grab processing - interaction removed but code preserved
        # elif slider_name == 'tempo_fader_1':
        #     fader = self.tempo_fader_1
        #     current_pos = fader.y + (1.0 - fader.value) * fader.height
        #     self.slider_grab_offset = y - current_pos
        # elif slider_name == 'tempo_fader_2':
        #     fader = self.tempo_fader_2
        #     current_pos = fader.y + (1.0 - fader.value) * fader.height
        #     self.slider_grab_offset = y - current_pos
        
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
            
        # DISABLED: Tempo fader active slider processing - interaction removed but code preserved
        # elif self.active_slider == 'tempo_fader_1':
        #     fader = self.tempo_fader_1
        #     adjusted_y = y - self.slider_grab_offset
        #     relative_y = (adjusted_y - fader.y) / fader.height
        #     fader_value = max(0.0, min(1.0, 1.0 - relative_y))
        #     fader.value = fader_value
        #     fader.is_dragging = True
        #     self.audio_engine.set_tempo(1, fader_value)
        #     
        # elif self.active_slider == 'tempo_fader_2':
        #     fader = self.tempo_fader_2
        #     adjusted_y = y - self.slider_grab_offset
        #     relative_y = (adjusted_y - fader.y) / fader.height
        #     fader_value = max(0.0, min(1.0, 1.0 - relative_y))
        #     fader.value = fader_value
        #     fader.is_dragging = True
        #     self.audio_engine.set_tempo(2, fader_value)
    
    def _release_active_slider(self):
        """Release the currently active slider"""
        if self.active_slider == 'volume_fader_1':
            self.volume_fader_1.is_dragging = False
        elif self.active_slider == 'volume_fader_2':
            self.volume_fader_2.is_dragging = False
        elif self.active_slider == 'crossfader':
            self.crossfader.is_dragging = False
        # DISABLED: Tempo fader release processing - interaction removed but code preserved
        # elif self.active_slider == 'tempo_fader_1':
        #     self.tempo_fader_1.is_dragging = False
        # elif self.active_slider == 'tempo_fader_2':
        #     self.tempo_fader_2.is_dragging = False
            
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
        
        # Update our separate rotation tracking for image rotation
        if deck == 1:
            self.deck1_jog_rotation += angle_diff
        else:
            self.deck2_jog_rotation += angle_diff
        
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
        
        # NOTE: Keep deck jog rotation at current position when released
        # Only reset rotation when cue button is pressed
    
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
     
    def _rotate_image(self, image, angle, center):
        """Rotate an image around its center point"""
        if image is None:
            return None
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return rotated
    
    def _draw_professional_jog_wheel(self, overlay, jog_wheel, deck_num, label):
        """Draw jog wheel using UI image with rotation based on interaction"""
        center_x, center_y = jog_wheel.center_x, jog_wheel.center_y
        radius = jog_wheel.radius
        
        # Determine jog wheel state
        is_playing = (self.audio_engine.deck1_is_playing if deck_num == 1 
                      else self.audio_engine.deck2_is_playing)
        is_touching = jog_wheel.is_touching
        
        if self.jogwheel_image is not None:
            # Get current rotation angle (only rotate when manually interacting, not during playback)
            current_rotation = self.deck1_jog_rotation if deck_num == 1 else self.deck2_jog_rotation
            
        # Only rotate the wheel image when being touched (like real DJ controllers)
        if is_touching:
            rotated_image = self._rotate_image(self.jogwheel_image, -current_rotation, (250, 250))
        else:
            rotated_image = self.jogwheel_image
        
        # Draw the jog wheel image
        if rotated_image is not None:
            top_left_x = center_x - 250
            top_left_y = center_y - 250
            self._draw_jog_wheel_image(overlay, rotated_image, top_left_x, top_left_y)
        
        # Draw album artwork in center of jog wheel (210px diameter, rotating like real DJ controllers)
        self._draw_album_artwork_on_jog_wheel(overlay, deck_num, center_x, center_y)
        
        # Draw position indicator line (like real DJ controllers - shows playback position)
        if is_playing:
            try:
                position = self.audio_engine.get_playback_position(deck_num)
                position_angle = position * 2 * np.pi - np.pi/2  # Start from top (12 o'clock)
                
                # Draw line from center to edge showing current position
                line_end_x = int(center_x + 200 * np.cos(position_angle))
                line_end_y = int(center_y + 200 * np.sin(position_angle))
                
                # Position line color
                pos_color = (255, 255, 255)  # White line
                cv2.line(overlay, (center_x, center_y), (line_end_x, line_end_y), pos_color, 3)
                
                # Draw small dot at the end
                cv2.circle(overlay, (line_end_x, line_end_y), 5, pos_color, -1)
                
            except:
                pass  # Skip if position can't be determined
        
        # Draw interaction feedback
        if is_touching:
            # Draw ring around wheel to show interaction
            ring_color = (255, 215, 0)  # Gold when touched
            cv2.circle(overlay, (center_x, center_y), radius + 5, ring_color, 4)
            
            feedback_text = "SCRATCHING" if is_playing else "NAVIGATING"
            text_color = (255, 255, 0)  # Yellow
            cv2.putText(overlay, feedback_text, (center_x - 40, center_y - radius - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    def _draw_album_artwork_on_jog_wheel(self, overlay, deck_num, center_x, center_y):
        """Draw rotating album artwork in the center of the jog wheel (210px diameter)"""
        # Get the appropriate album artwork image and rotation angle
        if deck_num == 1:
            artwork_image = self.deck1_artwork_image
            rotation_angle = self.deck1_artwork_rotation
        elif deck_num == 2:
            artwork_image = self.deck2_artwork_image
            rotation_angle = self.deck2_artwork_rotation
        else:
            return  # Invalid deck number
        
        # If no album artwork is available, keep the center blank
        if artwork_image is None:
            return
        
        # Rotate the album artwork based on current rotation angle
        rotated_artwork = self._rotate_image(artwork_image, rotation_angle, (105, 105))  # Center at 105,105 for 210x210 image
        
        if rotated_artwork is not None:
            # Draw the rotated album artwork in the center of the jog wheel
            # Position it so it's centered on the jog wheel center
            artwork_top_left_x = center_x - 105  # Half of 210px
            artwork_top_left_y = center_y - 105  # Half of 210px
            
            # Apply circular mask to make it look like a real vinyl record
            self._draw_circular_image(overlay, rotated_artwork, artwork_top_left_x, artwork_top_left_y, 105)
    
    def _draw_circular_image(self, overlay, image, x, y, radius):
        """Draw an image with circular mask (like a vinyl record)"""
        if image is None:
            return
        
        h, w = image.shape[:2]
        
        # Ensure we don't draw outside overlay bounds
        if x < 0 or y < 0 or x + w > overlay.shape[1] or y + h > overlay.shape[0]:
            return
        
        # Create circular mask
        center = (radius, radius)  # Center of the image
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
        
        # Extract BGR channels
        if len(image.shape) == 4:
            bgr = image[:, :, :3]
        else:
            bgr = image
        
        # Apply circular mask to only draw pixels within the circle
        overlay_region = overlay[y:y+h, x:x+w]
        
        for c in range(3):
            overlay_region[:, :, c] = np.where(mask, bgr[:, :, c], overlay_region[:, :, c])
    
    def _draw_jog_wheel_image(self, overlay, image, x, y):
        """Draw jog wheel image with circular mask to remove black background"""
        if image is None:
            return
        
        h, w = image.shape[:2]
        
        # Ensure we don't go out of bounds
        if x < 0 or y < 0 or x + w > overlay.shape[1] or y + h > overlay.shape[0]:
            return
        
        # Create circular mask for jog wheel (250px radius for 500px diameter)
        center = (250, 250)  # Center of the 500x500 image
        radius = 250
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
        
        # Extract BGR channels (ignore alpha since we're masking)
        if len(image.shape) == 4:
            bgr = image[:, :, :3]
        else:
            bgr = image
        
        # Apply circular mask to only draw pixels within the circle
        overlay_region = overlay[y:y+h, x:x+w]
        
        for c in range(3):
            overlay_region[:, :, c] = np.where(mask, bgr[:, :, c], overlay_region[:, :, c])
    
    def _draw_image_with_alpha(self, overlay, image, x, y):
        """Draw an image with circular mask onto overlay at specified position"""
        if image is None:
            return
        
        h, w = image.shape[:2]
        
        # Ensure we don't go out of bounds
        if x < 0 or y < 0 or x + w > overlay.shape[1] or y + h > overlay.shape[0]:
            return
        
        # Create circular mask (75px radius for 150px diameter)
        center = (75, 75)  # Center of the 150x150 image
        radius = 75
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
        
        # Extract BGR channels (ignore alpha since we're masking)
        if len(image.shape) == 4:
            bgr = image[:, :, :3]
        else:
            bgr = image
        
        # Apply circular mask to only draw pixels within the circle
        overlay_region = overlay[y:y+h, x:x+w]
        
        for c in range(3):
            overlay_region[:, :, c] = np.where(mask, bgr[:, :, c], overlay_region[:, :, c])
    
    def _draw_button_ring(self, overlay, center_x, center_y, radius, color, thickness=4):
        """Draw a colored ring around a circular button"""
        cv2.circle(overlay, (center_x, center_y), radius, color, thickness)
    
    def _draw_rounded_rectangle(self, overlay, x, y, width, height, radius, color, thickness=-1):
        """Draw a rounded rectangle"""
        # Main rectangles
        cv2.rectangle(overlay, (x + radius, y), (x + width - radius, y + height), color, thickness)
        cv2.rectangle(overlay, (x, y + radius), (x + width, y + height - radius), color, thickness)
        
        # Corner circles
        cv2.circle(overlay, (x + radius, y + radius), radius, color, thickness)
        cv2.circle(overlay, (x + width - radius, y + radius), radius, color, thickness)
        cv2.circle(overlay, (x + radius, y + height - radius), radius, color, thickness)
        cv2.circle(overlay, (x + width - radius, y + height - radius), radius, color, thickness)
    
    def _draw_rounded_border_only(self, overlay, x, y, width, height, radius, color, thickness=2):
        """Draw only the external border outline of a rounded rectangle"""
        # Draw the four edge lines (avoiding corners)
        # Top edge
        cv2.line(overlay, (x + radius, y), (x + width - radius, y), color, thickness)
        # Bottom edge  
        cv2.line(overlay, (x + radius, y + height), (x + width - radius, y + height), color, thickness)
        # Left edge
        cv2.line(overlay, (x, y + radius), (x, y + height - radius), color, thickness)
        # Right edge
        cv2.line(overlay, (x + width, y + radius), (x + width, y + height - radius), color, thickness)
        
        # Draw corner arcs (quarter circles)
        # Top-left corner
        cv2.ellipse(overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        # Top-right corner
        cv2.ellipse(overlay, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        # Bottom-right corner
        cv2.ellipse(overlay, (x + width - radius, y + height - radius), (radius, radius), 0, 0, 90, color, thickness)
        # Bottom-left corner
        cv2.ellipse(overlay, (x + radius, y + height - radius), (radius, radius), 90, 0, 90, color, thickness)
    
    def _draw_expanded_hit_area_feedback(self, overlay):
        """Draw subtle visual feedback showing expanded hit areas for better user understanding"""
        # Only show feedback with very low opacity so it's not distracting
        feedback_alpha = 0.15  # Very subtle
        
        # Determine edge areas where extra large hit areas are used
        edge_margin = 200
        
        # Draw expanded hit areas for all interactive buttons
        for deck_buttons in [self.deck1_buttons, self.deck2_buttons]:
            for button in deck_buttons.values():
                if button.button_type == "display":  # Skip non-clickable buttons
                    continue
                    
                # Check if button is in edge area
                button_center_x = button.x + button.width // 2
                button_center_y = button.y + button.height // 2
                in_edge_area = (button_center_x < edge_margin or button_center_x > self.screen_width - edge_margin or 
                               button_center_y < edge_margin or button_center_y > self.screen_height - edge_margin)
                
                if button.name in ["Cue", "Play/Pause"]:
                    # Draw expanded circular hit area
                    center_x = button.x + 75
                    center_y = button.y + 75
                    
                    if in_edge_area:
                        expanded_radius = 110  # Matches collision detection
                        feedback_color = (0, 255, 255)  # Cyan for edge areas
                    else:
                        expanded_radius = 95   # Matches collision detection
                        feedback_color = (0, 200, 255)  # Light blue for center areas
                    
                    # Draw very subtle circle outline showing expanded hit area
                    cv2.circle(overlay, (center_x, center_y), expanded_radius, feedback_color, 2)
                    
                    # Draw original button area in different color for comparison
                    cv2.circle(overlay, (center_x, center_y), 75, (255, 255, 255), 1)
                    
                else:
                    # Draw expanded rectangular hit area for pads
                    if in_edge_area:
                        margin = 30  # Matches collision detection
                        feedback_color = (255, 255, 0)  # Yellow for edge areas
                    else:
                        margin = 20  # Matches collision detection
                        feedback_color = (255, 200, 0)  # Orange for center areas
                    
                    # Draw expanded hit area rectangle
                    expanded_x = button.x - margin
                    expanded_y = button.y - margin
                    expanded_width = button.width + (margin * 2)
                    expanded_height = button.height + (margin * 2)
                    
                    # Draw very subtle rectangle outline showing expanded hit area
                    cv2.rectangle(overlay, (expanded_x, expanded_y), 
                                 (expanded_x + expanded_width, expanded_y + expanded_height), 
                                 feedback_color, 2)
                    
                    # Draw original button area in different color for comparison
                    cv2.rectangle(overlay, (button.x, button.y), 
                                 (button.x + button.width, button.y + button.height), 
                                 (255, 255, 255), 1)
        
        # Draw expanded hit areas for faders with similar feedback
        for fader in [self.volume_fader_1, self.volume_fader_2]:
            # Check if fader is in edge area
            fader_center_x = fader.x + fader.width // 2
            fader_center_y = fader.y + fader.height // 2
            in_edge_area = (fader_center_x < edge_margin or fader_center_x > self.screen_width - edge_margin or 
                           fader_center_y < edge_margin or fader_center_y > self.screen_height - edge_margin)
            
            if in_edge_area:
                margin = 35  # Matches collision detection
                feedback_color = (0, 255, 0)  # Green for edge areas
            else:
                margin = 25  # Matches collision detection
                feedback_color = (0, 200, 0)  # Light green for center areas
            
            # Draw expanded hit area rectangle for faders
            expanded_x = fader.x - margin
            expanded_y = fader.y - margin
            expanded_width = fader.width + (margin * 2)
            expanded_height = fader.height + (margin * 2)
            
            # Draw very subtle rectangle outline
            cv2.rectangle(overlay, (expanded_x, expanded_y), 
                         (expanded_x + expanded_width, expanded_y + expanded_height), 
                         feedback_color, 1)
        
        # Draw expanded hit area for crossfader
        crossfader_center_x = self.crossfader.x + self.crossfader.width // 2
        crossfader_center_y = self.crossfader.y + self.crossfader.height // 2
        in_edge_area = (crossfader_center_x < edge_margin or crossfader_center_x > self.screen_width - edge_margin or 
                       crossfader_center_y < edge_margin or crossfader_center_y > self.screen_height - edge_margin)
        
        if in_edge_area:
            margin = 35
            feedback_color = (255, 0, 255)  # Magenta for edge areas
        else:
            margin = 25
            feedback_color = (200, 0, 255)  # Purple for center areas
        
        # Draw expanded hit area for crossfader
        expanded_x = self.crossfader.x - margin
        expanded_y = self.crossfader.y - margin
        expanded_width = self.crossfader.width + (margin * 2)
        expanded_height = self.crossfader.height + (margin * 2)
        
        cv2.rectangle(overlay, (expanded_x, expanded_y), 
                     (expanded_x + expanded_width, expanded_y + expanded_height), 
                     feedback_color, 1)
    
    def draw_controller_overlay(self, frame):
        """Draw the DJ controller overlay on the frame"""
        overlay = frame.copy()
        
        # Draw professional jog wheels with realistic DJ controller appearance
        self._draw_professional_jog_wheel(overlay, self.jog_wheel_1, 1, "DECK 1")
        
        self._draw_professional_jog_wheel(overlay, self.jog_wheel_2, 2, "DECK 2")
        
        # --- Track visualization removed from overlay, now in separate window ---
        
        # Draw buttons - circular for cue/play-pause, rectangular for others
        for deck_buttons, deck_name in [(self.deck1_buttons, "Deck 1"), (self.deck2_buttons, "Deck 2")]:
            for button in deck_buttons.values():
                color = button.active_color if button.is_active else button.color
                if button.is_pressed:
                    color = (255, 255, 100)  # Highlight when pressed
                
                if button.name in ["Cue", "Play/Pause"]:
                    # Draw image-based buttons for cue and play/pause
                    center_x = button.x + 75  # radius = 75px for 150px diameter
                    center_y = button.y + 75
                    radius = 75
                    
                    # Select appropriate image
                    if button.name == "Cue":
                        image = self.cue_image
                    else:  # Play/Pause
                        image = self.play_pause_image
                    
                    # Draw the button image (images are 150x150, button.x/y is top-left)
                    self._draw_image_with_alpha(overlay, image, button.x, button.y)
                    
                    # Draw colored ring around button instead of changing whole button color
                    ring_color = color
                    if button.is_pressed:
                        ring_color = (255, 255, 100)  # Highlight when pressed
                    elif button.is_active:
                        ring_color = button.active_color
                    else:
                        ring_color = button.color
                    
                    # Only draw ring if button has a state (not default gray)
                    if button.is_active or button.is_pressed:
                        self._draw_button_ring(overlay, center_x, center_y, radius, ring_color, 4)
                else:
                    # Draw clean rounded pads - 25px corner radius like real DJ controllers
                    # Use lighter grey when pad is pressed/active for better visual feedback
                    if button.is_active or button.is_pressed:
                        pad_color = (60, 60, 60)  # Lighter grey when active/pressed
                    else:
                        pad_color = (34, 34, 34)  # Original hex #222222 in BGR when inactive
                    corner_radius = 25
                     
                    # Draw filled rounded rectangle - clean solid pad with rounded corners
                    self._draw_rounded_rectangle(overlay, button.x, button.y, 
                                               button.width, button.height, corner_radius, pad_color, -1)
                    
                    # Draw light grey border around ALL pads for better definition (same thickness as active border)
                    light_grey_border = (80, 80, 80)  # Light grey border for all pads
                    border_thickness = 4  # Same thickness as active border for consistency
                    border_offset = border_thickness // 2
                    self._draw_rounded_border_only(overlay, button.x - border_offset, button.y - border_offset, 
                                                 button.width + border_thickness, button.height + border_thickness, 
                                                 corner_radius, light_grey_border, border_thickness)
                    
                    # Draw colored border light around button when active/pressed (external rounded border)
                    if button.is_active or button.is_pressed:
                        ring_color = color
                        if button.is_pressed:
                            ring_color = (255, 255, 100)  # Highlight when pressed
                        elif button.is_active:
                            ring_color = button.active_color
                        
                        # Draw external border outline only (clean border around the outside)
                        border_thickness = 4
                        border_offset = border_thickness // 2
                        self._draw_rounded_border_only(overlay, button.x - border_offset, button.y - border_offset, 
                                                     button.width + border_thickness, button.height + border_thickness, 
                                                     corner_radius, ring_color, border_thickness)
        
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
            # Consistent track width (12px wide)
            track_x = fader.x + 9
            track_width = 12
            
            # Fader track - #222222 with light grey contour
            cv2.rectangle(overlay, (track_x, fader.y), 
                         (track_x + track_width, fader.y + fader.height), (34, 34, 34), -1)  # #222222
            cv2.rectangle(overlay, (track_x, fader.y), 
                         (track_x + track_width, fader.y + fader.height), (180, 180, 180), 1)  # Light grey contour
            
            # Middle indicator line
            middle_y = fader.y + fader.height // 2
            cv2.line(overlay, (track_x - 3, middle_y), (track_x + track_width + 3, middle_y), (180, 180, 180), 2)
            
            # Longer, thicker handle
            handle_y = int(fader.y + fader.height * (1 - fader.value))
            handle_x = fader.x - 8
            handle_width = 46
            handle_height = 30  # Made longer
            
            # White handle (no color changes, no text)
            cv2.rectangle(overlay, (handle_x, handle_y - handle_height//2), 
                         (handle_x + handle_width, handle_y + handle_height//2), (255, 255, 255), -1)
            cv2.rectangle(overlay, (handle_x, handle_y - handle_height//2), 
                         (handle_x + handle_width, handle_y + handle_height//2), (150, 150, 150), 2)
        
        # Crossfader - clean minimal style
        cf_rect = (self.crossfader.x, self.crossfader.y, self.crossfader.width, self.crossfader.height)
        cf_pos = self.crossfader.value
        
        # Consistent track thickness (12px tall)
        track_y = cf_rect[1] + 9
        track_height = 12
        
        # Crossfader track - #222222 with light grey contour
        cv2.rectangle(overlay, (cf_rect[0], track_y), 
                     (cf_rect[0] + cf_rect[2], track_y + track_height), (34, 34, 34), -1)  # #222222
        cv2.rectangle(overlay, (cf_rect[0], track_y), 
                     (cf_rect[0] + cf_rect[2], track_y + track_height), (180, 180, 180), 1)  # Light grey contour
        
        # Middle indicator line
        middle_x = cf_rect[0] + cf_rect[2] // 2
        cv2.line(overlay, (middle_x, track_y - 3), (middle_x, track_y + track_height + 3), (180, 180, 180), 2)
        
        # Longer, thicker handle
        handle_x = int(cf_rect[0] + cf_rect[2] * cf_pos)
        handle_width = 35  # Made longer
        handle_height = 50  # Made longer
        
        # White handle (no color changes, no text)
        cv2.rectangle(overlay, (handle_x - handle_width//2, cf_rect[1] - 10), 
                     (handle_x + handle_width//2, cf_rect[1] + cf_rect[3] + 10), (255, 255, 255), -1)
        cv2.rectangle(overlay, (handle_x - handle_width//2, cf_rect[1] - 10), 
                     (handle_x + handle_width//2, cf_rect[1] + cf_rect[3] + 10), (150, 150, 150), 2)
        
        # Tempo controls - Hidden but functionality maintained (might bring back later)
        # for fader, side in [(self.tempo_fader_1, "left"), (self.tempo_fader_2, "right")]:
        #     deck_num = 1 if side == "left" else 2
        #     
        #     # Tempo fader background
        #     fader_color = (60, 60, 60)
        #     if fader.is_dragging:
        #         fader_color = (80, 80, 60)  # Slightly yellow when active
        #     
        #     cv2.rectangle(overlay, (fader.x, fader.y), 
        #                  (fader.x + fader.width, fader.y + fader.height), fader_color, -1)
        #     
        #     # Tempo fader border
        #     cv2.rectangle(overlay, (fader.x, fader.y), 
        #                  (fader.x + fader.width, fader.y + fader.height), (150, 150, 150), 2)
        #     
        #     # Center line (normal tempo)
        #     center_y = fader.y + fader.height // 2
        #     cv2.line(overlay, (fader.x, center_y), (fader.x + fader.width, center_y), 
        #             (200, 200, 200), 1)
        #     
        #     # Tempo handle
        #     handle_y = int(fader.y + fader.height * (1 - fader.value))
        #     handle_color = (255, 255, 255) if not fader.is_dragging else (255, 255, 0)
        #     cv2.rectangle(overlay, (fader.x - 3, handle_y - 8), 
        #                  (fader.x + fader.width + 3, handle_y + 8), handle_color, -1)
        #     
        #     # Tempo value and BPM display
        #     tempo_percent = self.audio_engine.get_tempo_percentage(deck_num)
        #     current_bpm = self.audio_engine.get_current_bpm(deck_num)
        #     
        #     # Tempo percentage
        #     tempo_text = f"{tempo_percent:+.1f}%"
        #     if abs(tempo_percent) < 0.1:
        #         tempo_text = "0.0%"
        #         tempo_color = (0, 255, 0)  # Green for normal speed
        #     elif tempo_percent > 0:
        #         tempo_color = (0, 100, 255)  # Blue for faster
        #     else:
        #         tempo_color = (255, 100, 0)  # Orange for slower
        #     
        #     cv2.putText(overlay, tempo_text, (fader.x - 25, fader.y - 25), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, tempo_color, 1)
        #     
        #     # Current BPM
        #     bmp_text = f"{current_bpm:.1f}"
        #     cv2.putText(overlay, bmp_text, (fader.x - 15, fader.y - 10), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        #     
        #     # Label
        #     cv2.putText(overlay, f"TEMPO{deck_num}", (fader.x - 30, fader.y + fader.height + 15), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # EQ knobs removed for cleaner interface
        
        # Visual feedback for expanded hit areas (disabled for now)
        # self._draw_expanded_hit_area_feedback(overlay)
        
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
                self.audio_engine.set_stem_volume(1, "vocals", 1.0)
                self.audio_engine.set_stem_volume(1, "instrumental", 1.0)
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
                self.audio_engine.set_stem_volume(2, "vocals", 1.0)
                self.audio_engine.set_stem_volume(2, "instrumental", 1.0)
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
                regular_pinch_threshold = 0.04 * width  # Optimized for instant response
                jog_pinch_threshold = 0.04 * width     # Optimized for instant scratching
                
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
    
    def load_default_tracks(self, selected_songs=None):
        """Load tracks into both decks"""
        
        # Scan available tracks first
        self.track_loader.scan_tracks()
        
        if len(self.track_loader.available_tracks) == 0:
            print("❌ No tracks found in songs folder!")
            return
        
        # Get song selection
        if selected_songs:
            DECK1_SONG, DECK2_SONG = selected_songs
            print(f"🎵 DECK 1: {DECK1_SONG}")
            print(f"🎵 DECK 2: {DECK2_SONG}")
        else:
            # Use default hardcoded songs
            DECK1_SONG = "Kesha - Die Young"
            DECK2_SONG = "Avicii - Levels"
            print(f"🎵 Using default songs: {DECK1_SONG} & {DECK2_SONG}")
        
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
        
        # Auto BPM Sync - Calculate average BPM and sync both decks
        self._auto_bpm_sync()
        
        # Display final BPM status
        self.print_bpm_status()
        
        print("✅ Ready to DJ!")
    
    def _auto_bpm_sync(self):
        """Automatically sync both decks to the average BPM of the loaded tracks"""
        if not self.enable_bpm_sync:
            print("🔄 BPM Sync: DISABLED - Tracks will play at original BPM")
            print("   📊 To enable sync, restart without 'unsync' command")
            return
            
        if not self.deck1_track or not self.deck2_track:
            print("⚠️ BPM Sync: Need both tracks loaded")
            return
        
        # Get original BPMs
        deck1_bpm = self.deck1_track.bpm
        deck2_bpm = self.deck2_track.bpm
        
        # Calculate average BPM (no decimals)
        average_bpm = int((deck1_bpm + deck2_bpm) / 2)
        
        # Calculate tempo adjustment needed for each deck
        deck1_tempo_ratio = average_bpm / deck1_bpm
        deck2_tempo_ratio = average_bpm / deck2_bpm
        
        # Check if BPM sync is possible within tempo range (0.8x to 1.2x)
        if deck1_tempo_ratio < 0.8 or deck1_tempo_ratio > 1.2:
            print(f"⚠️ Warning: Deck 1 needs {deck1_tempo_ratio:.3f}x tempo (outside 0.8-1.2 range)")
        if deck2_tempo_ratio < 0.8 or deck2_tempo_ratio > 1.2:
            print(f"⚠️ Warning: Deck 2 needs {deck2_tempo_ratio:.3f}x tempo (outside 0.8-1.2 range)")
        
        # Convert ratio to fader value (0.5 = normal tempo)
        # Actual tempo range: 0.8 = 80% speed, 1.0 = 100% speed, 1.2 = 120% speed
        # Fader range: 0.0 = 80% speed, 0.5 = 100% speed, 1.0 = 120% speed
        deck1_fader_value = (deck1_tempo_ratio - 0.8) / 0.4
        deck2_fader_value = (deck2_tempo_ratio - 0.8) / 0.4
        
        # Clamp to valid range [0.0, 1.0] and warn if clamping occurs
        original_deck1_fader = deck1_fader_value
        original_deck2_fader = deck2_fader_value
        deck1_fader_value = max(0.0, min(1.0, deck1_fader_value))
        deck2_fader_value = max(0.0, min(1.0, deck2_fader_value))
        
        if abs(original_deck1_fader - deck1_fader_value) > 0.001:
            print(f"⚠️ Deck 1 fader clamped from {original_deck1_fader:.3f} to {deck1_fader_value:.3f}")
        if abs(original_deck2_fader - deck2_fader_value) > 0.001:
            print(f"⚠️ Deck 2 fader clamped from {original_deck2_fader:.3f} to {deck2_fader_value:.3f}")
        
        # Apply tempo sync to both decks
        self.tempo_fader_1.value = deck1_fader_value
        self.tempo_fader_2.value = deck2_fader_value
        self.audio_engine.set_tempo(1, deck1_fader_value)
        self.audio_engine.set_tempo(2, deck2_fader_value)
        
        # Calculate actual resulting BPMs for verification
        actual_deck1_bpm = deck1_bpm * deck1_tempo_ratio
        actual_deck2_bpm = deck2_bpm * deck2_tempo_ratio
        
        # Print comprehensive sync information
        print("=" * 60)
        print("🎵 AUTO BPM SYNC APPLIED")
        print("=" * 60)
        print(f"📊 SYNC TARGET: {average_bpm} BPM (average of both tracks)")
        print()
        print("🎚️ DECK 1 BPM DETAILS:")
        print(f"   📀 Track: {self.deck1_track.name}")
        print(f"   🎵 Original BPM: {deck1_bpm}")
        print(f"   ⚡ Tempo Multiplier: {deck1_tempo_ratio:.3f}x ({((deck1_tempo_ratio-1)*100):+.1f}%)")
        print(f"   🎯 Synced BPM: {actual_deck1_bpm:.1f}")
        print()
        print("🎚️ DECK 2 BPM DETAILS:")
        print(f"   📀 Track: {self.deck2_track.name}")
        print(f"   🎵 Original BPM: {deck2_bpm}")
        print(f"   ⚡ Tempo Multiplier: {deck2_tempo_ratio:.3f}x ({((deck2_tempo_ratio-1)*100):+.1f}%)")
        print(f"   🎯 Synced BPM: {actual_deck2_bpm:.1f}")
        print()
        print(f"✅ RESULT: Both tracks now playing at {average_bpm} BPM")
        print("=" * 60)
        print()
    
    def print_bpm_status(self):
        """Print current BPM status for both decks"""
        print()
        print("=" * 50)
        print("🎵 CURRENT BPM STATUS")
        print("=" * 50)
        
        if self.deck1_track:
            current_tempo_1 = self.audio_engine.get_tempo_multiplier(1)
            original_bpm_1 = self.deck1_track.bpm
            current_bpm_1 = original_bpm_1 * current_tempo_1
            print(f"🎚️ DECK 1: {self.deck1_track.name}")
            print(f"   🎵 Original BPM: {original_bpm_1}")
            print(f"   ⚡ Current Tempo: {current_tempo_1:.3f}x ({((current_tempo_1-1)*100):+.1f}%)")
            print(f"   🎯 Current BPM: {current_bpm_1:.1f}")
            print(f"   ▶️ Playing: {'YES' if self.audio_engine.deck1_is_playing else 'NO'}")
        else:
            print("🎚️ DECK 1: No track loaded")
        
        print()
        
        if self.deck2_track:
            current_tempo_2 = self.audio_engine.get_tempo_multiplier(2)
            original_bpm_2 = self.deck2_track.bpm
            current_bpm_2 = original_bpm_2 * current_tempo_2
            print(f"🎚️ DECK 2: {self.deck2_track.name}")
            print(f"   🎵 Original BPM: {original_bpm_2}")
            print(f"   ⚡ Current Tempo: {current_tempo_2:.3f}x ({((current_tempo_2-1)*100):+.1f}%)")
            print(f"   🎯 Current BPM: {current_bpm_2:.1f}")
            print(f"   ▶️ Playing: {'YES' if self.audio_engine.deck2_is_playing else 'NO'}")
        else:
            print("🎚️ DECK 2: No track loaded")
        
        print()
        
        if self.deck1_track and self.deck2_track:
            current_bpm_1 = self.deck1_track.bpm * self.audio_engine.get_tempo_multiplier(1)
            current_bpm_2 = self.deck2_track.bpm * self.audio_engine.get_tempo_multiplier(2)
            bpm_difference = abs(current_bpm_1 - current_bpm_2)
            
            if not self.enable_bpm_sync:
                print(f"🔄 SYNC STATUS: DISABLED - Tracks at original BPM")
            else:
                sync_status = "SYNCED" if bpm_difference < 1.0 else f"DIFF: {bpm_difference:.1f} BPM"
                print(f"🔄 SYNC STATUS: {sync_status}")
        
        print("=" * 50)
        print()

    def _find_track_by_name(self, song_name: str):
        """Find a track by its folder name (exact match or partial match)"""
        import unicodedata
        print(f"🔍 Looking for: '{song_name}'")
        
        # Normalize Unicode strings to handle composed vs decomposed forms
        normalized_song_name = unicodedata.normalize('NFC', song_name)
        
        # First try exact match with Unicode normalization
        for track in self.track_loader.available_tracks:
            normalized_track_name = unicodedata.normalize('NFC', track.name)
            if normalized_track_name == normalized_song_name:
                print(f"✅ Exact match found: '{track.name}'")
                return track
        
        # Then try partial match (contains) with Unicode normalization
        for track in self.track_loader.available_tracks:
            normalized_track_name = unicodedata.normalize('NFC', track.name)
            if normalized_song_name.lower() in normalized_track_name.lower():
                print(f"✅ Partial match found: '{track.name}'")
                return track
        
        # Special handling for complex names with Unicode normalization
        if 'newjeans' in normalized_song_name.lower():
            for track in self.track_loader.available_tracks:
                normalized_track_name = unicodedata.normalize('NFC', track.name)
                track_lower = normalized_track_name.lower()
                if any(keyword in track_lower for keyword in ['newjeans', 'new jeans', '뉴진스']):
                    print(f"✅ NewJeans match found: '{track.name}'")
                    return track
        
        # If still no match, print available tracks for debugging
        print(f"❌ No match found for '{song_name}'")
        print("📂 Available tracks:")
        for i, track in enumerate(self.track_loader.available_tracks[:5]):  # Show first 5
            print(f"   {i+1}: {track.name}")
        if len(self.track_loader.available_tracks) > 5:
            print(f"   ... and {len(self.track_loader.available_tracks) - 5} more")
        
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
            # Load album artwork for jog wheel
            self._load_album_artwork(1, track)
            # Set default buttons active - both vocal and instrumental ON
            self.deck1_buttons["vocal"].is_active = True
            self.deck1_buttons["instrumental"].is_active = True
            # Ensure audio engine reflects these settings properly
            self.audio_engine.set_stem_volume(1, "vocals", 1.0)
            self.audio_engine.set_stem_volume(1, "instrumental", 1.0)
        elif deck == 2:
            self.deck2_track = track
            self.audio_engine.load_track(2, track)
            # Load waveform data for visualization
            self.visualizer.set_track_waveform(2, track)
            # Set cue point at beginning (professional default)
            self.audio_engine.set_cue_point(2, 0.0)
            # Load album artwork for jog wheel
            self._load_album_artwork(2, track)
            # Set default buttons active - both vocal and instrumental ON
            self.deck2_buttons["vocal"].is_active = True
            self.deck2_buttons["instrumental"].is_active = True
            # Ensure audio engine reflects these settings properly
            self.audio_engine.set_stem_volume(2, "vocals", 1.0)
            self.audio_engine.set_stem_volume(2, "instrumental", 1.0)
    
    def _load_album_artwork(self, deck: int, track):
        """Load and prepare album artwork for jog wheel display"""
        if track.album_artwork and os.path.exists(track.album_artwork):
            try:
                # Load the image
                artwork = cv2.imread(track.album_artwork, cv2.IMREAD_COLOR)
                if artwork is not None:
                    # Resize to 210px diameter (as requested)
                    artwork_resized = cv2.resize(artwork, (210, 210))
                    
                    # Store in appropriate deck variable
                    if deck == 1:
                        self.deck1_artwork_image = artwork_resized
                        self.deck1_artwork_rotation = 0.0  # Reset to default position
                        print(f"🎨 Loaded album artwork for Deck 1: {os.path.basename(track.album_artwork)}")
                    elif deck == 2:
                        self.deck2_artwork_image = artwork_resized
                        self.deck2_artwork_rotation = 0.0  # Reset to default position
                        print(f"🎨 Loaded album artwork for Deck 2: {os.path.basename(track.album_artwork)}")
                else:
                    print(f"⚠️ Failed to load album artwork: {track.album_artwork}")
                    if deck == 1:
                        self.deck1_artwork_image = None
                    elif deck == 2:
                        self.deck2_artwork_image = None
            except Exception as e:
                print(f"❌ Error loading album artwork: {e}")
                if deck == 1:
                    self.deck1_artwork_image = None
                elif deck == 2:
                    self.deck2_artwork_image = None
        else:
            # No album artwork available
            if deck == 1:
                self.deck1_artwork_image = None
                self.deck1_artwork_rotation = 0.0
            elif deck == 2:
                self.deck2_artwork_image = None  
                self.deck2_artwork_rotation = 0.0
            print(f"📀 No album artwork found for Deck {deck} - jog wheel will remain blank")
    
    def update_album_artwork_rotation(self):
        """Update album artwork rotation based on playback state - like real DJ controllers"""
        # Deck 1 album artwork rotation
        if self.audio_engine.deck1_is_playing:
            # Rotate CLOCKWISE at DOUBLE speed when playing (like real DJ controllers)
            # About 66.6 RPM like a fast spinning record - negative values for clockwise rotation
            rotation_speed = -4.0  # degrees per frame (negative = clockwise, doubled from -2.0)
            self.deck1_artwork_rotation += rotation_speed
            if self.deck1_artwork_rotation <= -360.0:
                self.deck1_artwork_rotation += 360.0
        # When paused/stopped, rotation stays at current position
        
        # Deck 2 album artwork rotation
        if self.audio_engine.deck2_is_playing:
            rotation_speed = -4.0  # degrees per frame (negative = clockwise, doubled from -2.0)
            self.deck2_artwork_rotation += rotation_speed
            if self.deck2_artwork_rotation <= -360.0:
                self.deck2_artwork_rotation += 360.0
        # When paused/stopped, rotation stays at current position
    
    def run(self):
        """Main loop for the DJ controller"""
        pass  # Silently start interface
        
        # Load default tracks with selected songs
        self.load_default_tracks(self.selected_songs)
        
        # Frame rate synchronization for smooth 60fps operation
        self.target_fps = 60.0
        self.frame_time = 1.0 / self.target_fps  # 16.67ms per frame at 60fps
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.actual_fps = 60.0
        
        # Animation controller will be initialized after deck setup
        
        try:
            while True:
                # Use DJ camera wrapper if available, otherwise fallback to basic capture
                if hasattr(self, 'dj_camera') and self.dj_camera:
                    ret, frame = self.dj_camera.read_frame()
                else:
                    ret, frame = self.cap.read()
                    # Apply manual horizontal flip if using basic capture
                    if ret and frame is not None:
                        frame = cv2.flip(frame, 1)
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Ensure frame is exactly the right size for UI layout
                if frame.shape[:2] != (self.screen_height, self.screen_width):
                    frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                
                # Frame is already flipped by DJ camera wrapper or manual flip above
                # No additional flip needed
                
                # Process hand tracking - get both pinch types
                pinch_data, jog_pinch_data, results = self.hand_tracker.process_frame(frame)
                
                # Process interactions with both pinch types
                self.process_hand_interactions(pinch_data, jog_pinch_data)
                
                # Update jog wheel spinning based on playback state (realistic DJ behavior)
                self.update_jog_wheel_spinning()
                
                # Update album artwork rotation based on playback state (like real DJ controllers)
                self.update_album_artwork_rotation()
                
                # Update smooth animations for 60fps synchronization
                if hasattr(self, 'animation_controller') and self.animation_controller:
                    deck_states = {
                        'deck1_playing': hasattr(self, 'deck1_playing') and self.deck1_playing,
                        'deck2_playing': hasattr(self, 'deck2_playing') and self.deck2_playing,
                        'deck1_position': getattr(self, 'deck1_position', 0.0),
                        'deck2_position': getattr(self, 'deck2_position', 0.0),
                        'deck1_bpm': 128,  # Will be updated with actual BPM
                        'deck2_bpm': 126,  # Will be updated with actual BPM
                        'deck1_volume': getattr(self, 'deck1_master_volume', 0.0),
                        'deck2_volume': getattr(self, 'deck2_master_volume', 0.0)
                    }
                    self.animation_controller.update(deck_states)
                
                # Record performance metrics
                frame_start_time = current_time if 'current_time' in locals() else time.time()
                
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
                        # Active EQ knob tracking removed
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
                        # EQ knob hover detection removed for cleaner interface
                        
                        
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
                
                # Display main DJ controller frame
                cv2.imshow('Air DJ Controller', frame)
                
                # Display separate track visualization window
                viz_frame = self.create_track_visualization_window()
                cv2.imshow('Track Visualization', viz_frame)
                
                # Frame rate synchronization - maintain exact 60fps
                frame_end_time = time.time()
                frame_processing_time = frame_end_time - frame_start_time
                
                elapsed_time = frame_end_time - self.last_frame_time
                
                # Record performance data
                if hasattr(self, 'performance_monitor') and self.performance_monitor:
                    self.performance_monitor.record_frame_time(elapsed_time)
                
                # Calculate sleep time to maintain 60fps
                sleep_time = self.frame_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Update frame timing
                self.last_frame_time = time.time()
                self.frame_count += 1
                
                # Calculate and report actual FPS
                if self.frame_count % 60 == 0:  # Every 60 frames (1 second at 60fps)
                    fps_elapsed = frame_end_time - self.fps_start_time
                    if fps_elapsed > 0:
                        self.actual_fps = 60.0 / fps_elapsed
                        self.fps_start_time = frame_end_time
                
                # Report performance stats periodically
                if hasattr(self, 'performance_monitor') and self.performance_monitor and self.performance_monitor.should_report():
                    stats = self.performance_monitor.get_performance_stats()
                    print(f"🎬 Performance: {stats['avg_fps']:.1f}fps avg, "
                          f"{stats['frame_time_ms']:.1f}ms frame time, "
                          f"Processing: {frame_processing_time*1000:.1f}ms")
                
                # Handle key presses with frame-synchronized timing
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):  # Press 'f' to show FPS debug info
                    if hasattr(self, 'performance_monitor') and self.performance_monitor:
                        stats = self.performance_monitor.get_performance_stats()
                        print(f"🎬 FPS Debug: Target={self.target_fps:.1f}, "
                              f"Actual={self.actual_fps:.1f}, "
                              f"Avg={stats['avg_fps']:.1f}, "
                              f"Range={stats['min_fps']:.1f}-{stats['max_fps']:.1f}")
                    else:
                        print(f"🎬 FPS Debug: Target={self.target_fps:.1f}, Actual={self.actual_fps:.1f}")
                elif key == ord('s'):  # Press 's' to show smooth animation status
                    print(f"✨ Animation Status: Jog wheels, album art, and UI synchronized to {self.target_fps}fps")
                    
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
