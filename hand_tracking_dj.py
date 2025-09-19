import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import json
import os
import logging
from pyo import *
import threading
import queue
import random

# Import specific pyo modules for better clarity
from pyo import Server, SndTable, TableRead, Sine, SfPlayer, Harmonizer, STRev, Mix, SigTo, Sig, PeakAmp

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Force TensorFlow to use CPU only and disable AVX/AVX2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization

class HandDJ:
    def __init__(self, audio_file=None):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Reduced for faster detection
            min_tracking_confidence=0.3   # Reduced for faster tracking
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load calibration data if available
        self.calibration = self.load_calibration()
        
        # Audio initialization
        self.server = Server().boot()
        self.server.start()
        
        # Default audio if none provided
        if audio_file:
            self.audio_path = audio_file
            # Try 3 different methods to play the audio
            if not self.try_load_audio():
                # If all loading methods fail, fall back to sine wave
                self.use_sine_wave()
        else:
            # Use a sine wave as default sound source
            self.use_sine_wave()
            
        # Parameters for audio manipulation
        self.speed = 1.0        # Default speed (1.0 = normal)
        self.pitch = 0          # Default pitch shift (0 = no shift)
        self.volume = 5.0       # Default volume (5.0 = normal on 0-10 scale)
        
        # Global variable for PYO to control SfPlayer
        self.g_speed = SigTo(1.0, time=0.1)
        
        # Minimal smoothing for ultra-low latency DJ response
        self.speed_history = [1.0] * 2   # Reduced from 5 to 2 for instant response
        self.pitch_history = [0] * 2     # Reduced from 5 to 2 for instant response
        self.volume_history = [5.0] * 2  # Reduced from 5 to 2 for instant response
        
        # Video capture setup
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera is opened correctly
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
        
        # For audio analysis and spectrum visualization
        self.spectrum_data = np.zeros(64)
        self.amplitude_data = 0.5  # Default amplitude
        self.last_amplitude_update = time.time()
        self.peaks = [0.0] * 64  # For peak holding in visualization
        self.peak_decay = 0.05   # How fast peaks fall
        
        # For waveform visualization
        self.waveform_buffer = []
        self.waveform_buffer_size = 50   # Reduced from 100 for lower latency
        self.waveform_amplitude = 0.0
        self.waveform_frequency = 0.0
        self.waveform_update_time = time.time()
        self.waveform_update_interval = 0.02  # Increased to 50 updates per second for smoother response
        
        # For frame counting (debug and timing)
        self.frame_count = 0
        
        # For gesture recognition
        self.twist_history = {
            'left': {'angles': [], 'timestamps': [], 'triggered': False},
            'right': {'angles': [], 'timestamps': [], 'triggered': False}
        }
        self.twist_threshold = 15  # Degrees from horizontal
        self.twist_cooldown = 0.3  # Reduced from 1.0s for faster response
        self.twist_memory = 3     # Reduced from 5 for faster gesture detection
        
        # For playlist functionality
        self.playlist = []
        self.current_track_index = 0
        self.track_change_time = 0
        self.track_change_animation = 0
        
        if audio_file:
            self.playlist.append(audio_file)
            self.scan_for_additional_tracks(audio_file)
    
    def use_sine_wave(self):
        """Switch to sine wave audio source"""
        print("Using sine wave as audio source with harmonics")
        # Create a richer sine wave with harmonics for better quality
        
        # Create control signals for smoother transitions
        self.freq_sig = SigTo(440, time=0.05, init=440)
        self.amp_sig = SigTo(0.3, time=0.05, init=0.3)
        
        # Create a richer sine wave with harmonics for better quality
        self.sine = Sine(freq=self.freq_sig, mul=self.amp_sig)
        self.harmonic1 = Sine(freq=self.freq_sig*2, mul=self.amp_sig*0.5)  # First harmonic
        self.harmonic2 = Sine(freq=self.freq_sig*3, mul=self.amp_sig*0.27)  # Second harmonic
        self.mixer = Mix([self.sine, self.harmonic1, self.harmonic2], voices=3)
        self.output = self.mixer.out()
        
        # Add reverb for better sound
        self.reverb = STRev(self.output, revtime=0.8, cutoff=8000, bal=0.1).out()
        self.audio_path = None
        
        # Set up peak detector for amplitude visualization
        try:
            self.peak_detector = PeakAmp(self.mixer)
        except Exception as e:
            print(f"Could not create peak detector: {e}")
        
        print("Sine wave synthesizer initialized with speed and pitch controls")
    
    def try_load_audio(self):
        """Try multiple methods to load audio file"""
        try:
            # Method 1: Try SfPlayer with high quality settings (good for MP3)
            print(f"Method 1: Trying SfPlayer with {self.audio_path}")
            try:
                # Create a global speed control variable with smoother transition
                self.g_speed = SigTo(1.0, time=0.05, init=1.0)
                print("Created global speed control with SigTo")
                
                # Use the global speed control for SfPlayer
                print("Initializing SfPlayer with speed control...")
                # For SfPlayer, we need to explicitly set the speed parameter
                # SfPlayer treats speed differently than our 0.1-2.0 range
                # We'll create a direct reference for better control
                self.player = SfPlayer(self.audio_path, loop=True, mul=0.8, interp=4)
                
                # Create a Harmonizer for pitch shifting
                self.pitch_shifter = Harmonizer(self.player, transpo=0, mul=0.8)
                # Add a high quality reverb for better sound
                self.reverb = STRev(self.pitch_shifter, revtime=1.0, cutoff=10000, bal=0.1).out()
                self.output = self.reverb
                
                # Set up peak detector for amplitude visualization
                try:
                    self.peak_detector = PeakAmp(self.pitch_shifter)
                except Exception as e:
                    print(f"Could not create peak detector: {e}")
                
                print("Success: Using SfPlayer with enhanced quality")
                return True
            except Exception as e:
                print(f"SfPlayer failed: {e}")
            
            # Method 2: Try TableRead with SndTable with high quality settings
            print(f"Method 2: Trying SndTable with {self.audio_path}")
            try:
                # Load audio into table
                self.table = SndTable(self.audio_path)
                
                # Store the base rate for future speed calculations
                self.base_rate = self.table.getRate()
                print(f"DEBUG - Loaded audio with base rate: {self.base_rate} Hz")
                
                # Create a rate control variable with smoother transition
                self.g_rate = SigTo(self.base_rate, time=0.05, init=self.base_rate)
                print("Created global rate control with SigTo")
                
                # Create TableRead with better interpolation
                self.player = TableRead(
                    table=self.table, 
                    freq=self.g_rate,  # Use rate control object instead of fixed rate
                    loop=True,
                    interp=4,  # Higher quality interpolation
                    mul=0.8
                )
                
                # Create a Harmonizer for pitch shifting
                self.pitch_shifter = Harmonizer(self.player, transpo=0, mul=0.8)
                
                # Add a high quality reverb for better sound
                self.reverb = STRev(self.pitch_shifter, revtime=1.0, cutoff=10000, bal=0.1).out()
                self.output = self.reverb
                
                # Set up peak detector for amplitude visualization
                try:
                    self.peak_detector = PeakAmp(self.pitch_shifter)
                except Exception as e:
                    print(f"Could not create peak detector: {e}")
                
                print("Success: Using SndTable with enhanced quality and speed control")
                return True
            except Exception as e:
                print(f"SndTable failed: {e}")
            
            # Method 3: Try to load as a raw audio file
            print(f"Method 3: Trying alternative method")
            return False  # All methods failed
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def load_calibration(self):
        """Load calibration data from file if it exists"""
        calibration = {
            "pinch_min": 0.0,  # Set to exactly 0 for accurate min mapping
            "pinch_max": 0.400,  # Set to exactly 0.400 as specified
            "distance_min": 0.1,
            "distance_max": 0.7
        }
        
        try:
            if os.path.exists('calibration.json'):
                with open('calibration.json', 'r') as f:
                    loaded_data = json.load(f)
                    calibration.update(loaded_data)
                print("Loaded calibration data from calibration.json")
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            print("Using default calibration values")
            
        # Force pinch min and max to requested values regardless of loaded data
        calibration["pinch_min"] = 0.0  # Ensure exact 0 for pinch min
        calibration["pinch_max"] = 0.400  # Ensure exact 0.400 for pinch max
        
        return calibration
    
    def update_audio_params(self):
        """Update audio parameters based on hand movements"""
        try:
            if self.audio_path is None:
                # For sine wave with harmonics
                # Calculate frequency using exponential formula for more natural pitch changes
                # Map pitch to frequency range 20-600Hz
                base_freq = 20  # Minimum frequency (changed from 60Hz to 20Hz)
                max_freq = 600  # Maximum frequency
                normalized_pitch = (self.pitch + 12) / 24.0  # Normalize pitch to 0-1 range
                new_freq = base_freq + normalized_pitch * (max_freq - base_freq)
                
                # Apply speed factor to the frequency
                # Speed affects the perceived pitch in sine wave synthesis
                speed_adjusted_freq = new_freq * self.speed
                
                # Update sine wave and harmonics with smooth transitions
                if hasattr(self, 'freq_sig'):
                    self.freq_sig.value = speed_adjusted_freq
                else:
                    # Fallback direct control
                    self.sine.freq = speed_adjusted_freq
                if hasattr(self, 'harmonic1'):
                    self.harmonic1.freq = speed_adjusted_freq * 2  # First harmonic (octave up)
                if hasattr(self, 'harmonic2'):
                    self.harmonic2.freq = speed_adjusted_freq * 3  # Second harmonic

                # Update volume with proper typecasting
                # Convert 0-10 scale to 0-1 for audio processing
                vol = float(self.volume) / 10.0
                if hasattr(self, 'amp_sig'):
                    self.amp_sig.value = vol * 0.6
                else:
                    # Fallback direct control
                    self.sine.mul = vol * 0.6
                    if hasattr(self, 'harmonic1'):
                        self.harmonic1.mul = vol * 0.3
                    if hasattr(self, 'harmonic2'):
                        self.harmonic2.mul = vol * 0.15
                
                # Force audio processing to update
                self.server.process()
            else:
                # For audio file
                # Convert all parameters to Python floats to avoid numpy type issues
                speed = float(self.speed)
                pitch = float(self.pitch)
                # Convert 0-10 scale to 0-1 for audio processing
                volume = float(self.volume) / 10.0
                
                # For SfPlayer (MP3 files)
                if hasattr(self, 'player') and isinstance(self.player, SfPlayer):
                    # Update the speed ratio using direct methods
                    try:
                        success = False
                        
                        # Method 1: SfPlayer has a direct setSpeed method we can use to control playback speed
                        # This is the most reliable way to affect playback speed
                        try:
                            # Call the direct method to control playback speed
                            self.player.setSpeed(speed)
                            success = True
                        except Exception as e:
                            # Alternative method: try manipulating the internal _base_objs
                            try:
                                # Directly update the playback speed through base objects
                                if hasattr(self.player, '_base_objs'):
                                    for obj in self.player._base_objs:
                                        if hasattr(obj, 'setSpeed'):
                                            obj.setSpeed(speed)
                                    success = True
                            except Exception as e:
                                pass
                        
                        # Final fallback: recreate the player with new speed if critical
                        if not success and abs(speed - 1.0) > 0.1:
                            try:
                                # Get current position if possible
                                current_pos = 0
                                if hasattr(self.player, 'pos'):
                                    try:
                                        current_pos = self.player.pos
                                    except:
                                        pass
                                
                                # Temporarily store and disconnect harmonizer and reverb
                                if hasattr(self, 'pitch_shifter'):
                                    self.pitch_shifter.stop()
                                if hasattr(self, 'reverb'):
                                    self.reverb.stop()
                                
                                # Create new player with explicit speed param
                                old_player = self.player
                                self.player = SfPlayer(self.audio_path, speed=speed, loop=True, 
                                                      mul=volume, interp=4)
                                
                                # Try to set position if we got one
                                if current_pos > 0:
                                    try:
                                        self.player.pos = current_pos
                                    except:
                                        pass
                                
                                # Reconnect the signal chain
                                if hasattr(self, 'pitch_shifter'):
                                    self.pitch_shifter = Harmonizer(self.player, transpo=self.pitch_shifter.transpo, 
                                                                   mul=volume)
                                    self.reverb = STRev(self.pitch_shifter, revtime=self.reverb.revtime, 
                                                      cutoff=self.reverb.cutoff, bal=0.1).out()
                                
                                # Stop the old player
                                old_player.stop()
                                success = True
                            except Exception as e:
                                pass
                        
                        # Force audio processing to update
                        self.server.process()
                        
                        self.player.mul = volume
                        if hasattr(self, 'pitch_shifter'):
                            # Map pitch to frequency range 20-600Hz
                            normalized_pitch = (pitch + 12) / 24.0  # Normalize to 0-1
                            transpo_value = (normalized_pitch * 2 - 1) * 12  # Map to semitone range
                            self.pitch_shifter.transpo = transpo_value
                    except Exception as e:
                        print(f"SfPlayer update error: {e}")
                
                # For TableRead with SndTable (WAV and other formats)
                if hasattr(self, 'player') and hasattr(self, 'table'):
                    try:
                        # Speed affects the playback rate
                        if hasattr(self, 'base_rate'):
                            base_rate = self.base_rate
                        else:
                            base_rate = self.table.getRate()
                            
                        new_rate = base_rate * speed
                        
                        # Update rate control variable
                        if hasattr(self, 'g_rate'):
                            self.g_rate.value = new_rate
                        else:
                            # Fallback for direct frequency setting
                            self.player.freq = new_rate
                        
                        # Force audio processing to update
                        self.server.process()
                            
                        # Make sure these changes are reflected in the output
                        if hasattr(self, 'pitch_shifter'):
                            # Map pitch to frequency range 20-600Hz
                            normalized_pitch = (pitch + 12) / 24.0  # Normalize to 0-1
                            transpo_value = (normalized_pitch * 2 - 1) * 12  # Map to semitone range
                            self.pitch_shifter.transpo = transpo_value
                            self.pitch_shifter.mul = volume
                    except Exception as e:
                        print(f"TableRead update error: {e}")
                
                # Apply dynamic effects based on parameters
                if hasattr(self, 'reverb') and isinstance(self.reverb, STRev):
                    # Adjust reverb time based on speed (slower = more reverb)
                    rev_time = 1.0 + (1.0 - min(1.0, speed)) * 2.0
                    self.reverb.revtime = rev_time
                    
                    # Adjust filter cutoff based on pitch
                    pitch_norm = (pitch + 12) / 24.0  # Normalize pitch to 0-1
                    cutoff = 2000 + pitch_norm * 8000  # Map to frequency range
                    self.reverb.cutoff = cutoff
            
            # Update waveform visualization data
            self.update_waveform()
            
            # Update visualization data
            self.update_visualization_data()
            
        except Exception as e:
            print(f"Error updating audio: {e}")
    
    def update_waveform(self):
        """Update waveform visualization data based on current audio parameters"""
        # Only update at specific intervals for performance
        current_time = time.time()
        if current_time - self.waveform_update_time < self.waveform_update_interval:
            return
        
        self.waveform_update_time = current_time
        
        # Calculate frequency value from pitch for waveform shape
        base_freq = 20
        max_freq = 600
        normalized_pitch = (self.pitch + 12) / 24.0
        self.waveform_frequency = normalized_pitch  # 0-1 value representing frequency
        
        # Determine amplitude based on volume and speed
        self.waveform_amplitude = (self.volume / 10.0) * min(1.5, self.speed)
        
        # Generate new waveform point
        # Use sine wave with variable frequency and amplitude for visualization
        t = current_time * 10  # Time factor for animation speed
        
        # Add frequency variation based on pitch
        freq_factor = 1 + self.waveform_frequency * 4  # Range 1-5x
        
        # Create a value that oscillates based on time, frequency and contains some randomness
        wave_value = np.sin(t * freq_factor) * self.waveform_amplitude
        
        # Add some harmonics for more interesting visuals
        wave_value += np.sin(t * freq_factor * 2) * (self.waveform_amplitude * 0.3)
        wave_value += np.sin(t * freq_factor * 3) * (self.waveform_amplitude * 0.15)
        
        # Add slight randomness for organic feel
        wave_value += (np.random.random() - 0.5) * self.waveform_amplitude * 0.2
        
        # Add to buffer
        self.waveform_buffer.append(wave_value)
        
        # Keep buffer at fixed size
        while len(self.waveform_buffer) > self.waveform_buffer_size:
            self.waveform_buffer.pop(0)
    
    def reset_parameters(self):
        """Reset all audio parameters to their default values"""
        # Store previous values for logging
        prev_speed = self.speed
        prev_pitch = self.pitch
        prev_vol = self.volume
        
        # Reset to defaults
        self.speed = 1.0
        self.pitch = 0.0
        self.volume = 5.0
        
        # Reset the smoothing history
        self.speed_history = [1.0] * len(self.speed_history)
        self.pitch_history = [0.0] * len(self.pitch_history)
        self.volume_history = [5.0] * len(self.volume_history)
        
        # Update audio immediately
        self.update_audio_params()
        
        # Calculate consistent default frequency based on pitch=0
        base_freq = 20
        max_freq = 600
        normalized_pitch = (0 + 12) / 24.0
        default_frequency = int(base_freq + normalized_pitch * (max_freq - base_freq))
        
        # Log the reset with consistent frequency calculation
        print(f"✓ RESET: Speed: 1.0x | Pitch: {default_frequency}Hz | Volume: 5.0")
        
        # Print current parameters to maintain consistent format
        self.log_parameters()
    
    def log_parameters(self):
        """Log current parameter values in a single line format"""
        # Calculate frequency from pitch
        base_freq = 20
        max_freq = 600
        normalized_pitch = (self.pitch + 12) / 24.0
        frequency = int(base_freq + normalized_pitch * (max_freq - base_freq))
        
        # Format output with all parameters on one line
        print(f"LEVELS | Speed: {self.speed:.1f}x | Pitch: {frequency}Hz | Volume: {self.volume:.1f}")
    
    def log_pinch_debug(self, hand, pinch_dist, mapped_value):
        """Log detailed debug information about pinch distances and mappings"""
        control_type = "Speed" if hand == "left" else "Pitch"
        value_str = f"{mapped_value:.1f}x" if hand == "left" else f"{mapped_value}Hz"
        normalized = pinch_dist / self.calibration["pinch_max"]
        
        print(f"DEBUG | {hand.upper()} pinch: {pinch_dist:.3f} | Normalized: {normalized:.2f} | {control_type}: {value_str}")
    
    def smooth_value(self, new_value, history_list):
        """Apply smoothing to reduce jitter"""
        history_list.pop(0)
        history_list.append(float(new_value))  # Ensure it's a Python float
        return float(sum(history_list) / len(history_list))  # Return Python float
    
    def scan_for_additional_tracks(self, first_track):
        """Look for additional audio tracks in the same directory"""
        try:
            # Get the directory of the first track
            directory = os.path.dirname(first_track)
            if not directory:
                directory = '.'
                
            # Get the file extension of the first track
            ext = os.path.splitext(first_track)[1].lower()
            
            # Find all files with the same extension
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if file_path != first_track and file_path.lower().endswith(ext):
                    # Add to playlist if it's a different audio file with same extension
                    self.playlist.append(file_path)
            
            if len(self.playlist) > 1:
                print(f"Found {len(self.playlist)} tracks in playlist:")
                for i, track in enumerate(self.playlist):
                    print(f"  {i+1}. {os.path.basename(track)}")
        except Exception as e:
            print(f"Error scanning for additional tracks: {e}")
    
    def next_track(self):
        """Switch to the next track in the playlist"""
        if len(self.playlist) <= 1:
            print("No more tracks in playlist")
            return False
            
        # Move to next track
        self.current_track_index = (self.current_track_index + 1) % len(self.playlist)
        next_file = self.playlist[self.current_track_index]
        print(f"▶️ Next track: {os.path.basename(next_file)}")
        
        # Stop current audio
        self.server.stop()
        
        # Restart server with new audio file
        self.server = Server().boot()
        self.server.start()
        
        # Set new audio path
        self.audio_path = next_file
        
        # Try to load the new audio file
        if not self.try_load_audio():
            # If loading fails, fall back to sine wave
            self.use_sine_wave()
            
        # Update audio params to maintain current settings
        self.update_audio_params()
        
        # Set track change animation
        self.track_change_time = time.time()
        self.track_change_animation = 1  # 1 = next track
        
        return True
        
    def prev_track(self):
        """Switch to the previous track in the playlist"""
        if len(self.playlist) <= 1:
            print("No more tracks in playlist")
            return False
            
        # Move to previous track
        self.current_track_index = (self.current_track_index - 1) % len(self.playlist)
        prev_file = self.playlist[self.current_track_index]
        print(f"◀️ Previous track: {os.path.basename(prev_file)}")
        
        # Stop current audio
        self.server.stop()
        
        # Restart server with new audio file
        self.server = Server().boot()
        self.server.start()
        
        # Set new audio path
        self.audio_path = prev_file
        
        # Try to load the new audio file
        if not self.try_load_audio():
            # If loading fails, fall back to sine wave
            self.use_sine_wave()
            
        # Update audio params to maintain current settings
        self.update_audio_params()
        
        # Set track change animation
        self.track_change_time = time.time()
        self.track_change_animation = -1  # -1 = previous track
        
        return True
    
    def detect_horizontal_twist(self, thumb_pos, index_pos, handedness):
        """Detect if the pinch line is being held horizontally (twisted)"""
        if thumb_pos is None or index_pos is None:
            return False
            
        # Calculate the angle of the pinch line
        dx = index_pos[0] - thumb_pos[0]
        dy = index_pos[1] - thumb_pos[1]
        
        # Calculate angle in degrees (0 = horizontal, 90 = vertical)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get the history for this hand
        history_key = 'left' if handedness == 'Left' else 'right'
        history = self.twist_history[history_key]
        
        # Add current angle to history with timestamp
        current_time = time.time()
        history['angles'].append(angle)
        history['timestamps'].append(current_time)
        
        # Keep history at manageable size
        if len(history['angles']) > self.twist_memory:
            history['angles'].pop(0)
            history['timestamps'].pop(0)
            
        # Debug the angle occasionally
        if hasattr(self, 'frame_count') and self.frame_count % 60 == 0:
            print(f"DEBUG: {handedness} hand angle: {angle:.1f} degrees")
            
        # Check for horizontal twist gesture with direction
        if handedness == 'Left':
            # For left hand, check if twisted to the left (angle near 180 or -180 degrees)
            # When twisted left, angle will be close to 180 or -180 degrees
            is_twisted = abs(abs(angle) - 180) < self.twist_threshold
        else:
            # For right hand, just check if horizontal (angle near 0)
            is_twisted = abs(angle) < self.twist_threshold
        
        # Only trigger once when twist is detected and not recently triggered
        if is_twisted and not history['triggered']:
            # Check cooldown
            if len(history['timestamps']) > 0:
                last_trigger_time = history.get('last_trigger_time', 0)
                if current_time - last_trigger_time > self.twist_cooldown:
                    history['triggered'] = True
                    history['last_trigger_time'] = current_time
                    print(f"DEBUG: {handedness} twist detected, angle = {angle:.1f}")
                    return True
        elif not is_twisted:
            # Reset trigger state when no longer in twisted position
            history['triggered'] = False
            
        return False
    
    def process_hands(self, left_hand_landmarks, right_hand_landmarks):
        # Track if any parameter has changed significantly
        params_changed = False
        
        # Store midpoints of pinch lines for volume calculation
        left_pinch_midpoint = None
        right_pinch_midpoint = None
        
        # Track thumb and index positions for gesture detection
        left_thumb_pos = None
        left_index_pos = None
        right_thumb_pos = None
        right_index_pos = None
        
        # Process left hand for speed control (thumb-index pinch)
        if left_hand_landmarks:
            # Get screen coordinates for gesture detection
            h, w, c = self.image_shape if hasattr(self, 'image_shape') else (1080, 1920, 3)
            
            left_thumb_tip = left_hand_landmarks.landmark[4]
            left_index_tip = left_hand_landmarks.landmark[8]
            left_thumb_pos = (int(left_thumb_tip.x * w), int(left_thumb_tip.y * h))
            left_index_pos = (int(left_index_tip.x * w), int(left_index_tip.y * h))
            
            left_thumb = np.array([left_thumb_tip.x, left_thumb_tip.y])
            left_index = np.array([left_index_tip.x, left_index_tip.y])
            
            # Calculate midpoint of left pinch line
            left_pinch_midpoint = (left_thumb + left_index) / 2
            
            # Calculate pinch distance
            left_pinch_dist = np.linalg.norm(left_thumb - left_index)
            
            # Direct linear mapping from pinch distance to speed
            # Pinch 0 -> 0.1x speed, Pinch 0.400 -> 2.0x speed
            pinch_max = self.calibration["pinch_max"]  # Should be 0.400
            
            # Clamp pinch distance to valid range
            clamped_pinch = max(0, min(pinch_max, left_pinch_dist))
            
            # Linear mapping from pinch to speed 
            normalized_pinch = clamped_pinch / pinch_max  # 0 to 1 range
            raw_speed = 0.1 + normalized_pinch * 1.9  # 0.1 to 2.0 range
            
            # Debug logging - every 30 frames
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 0
            
            if self.frame_count % 30 == 0:
                self.log_pinch_debug("left", left_pinch_dist, raw_speed)
            
            # Apply smoothing
            old_speed = self.speed
            self.speed = self.smooth_value(raw_speed, self.speed_history)
            
            # Check if speed changed significantly
            if abs(self.speed - old_speed) > 0.05:
                params_changed = True
            
            # Check for horizontal twist gesture with left hand
            if self.detect_horizontal_twist(left_thumb_pos, left_index_pos, 'Left'):
                # Left hand horizontal twist - previous track
                self.prev_track()
        
        # Process right hand for pitch/frequency control
        if right_hand_landmarks:
            # Get screen coordinates for gesture detection
            h, w, c = self.image_shape if hasattr(self, 'image_shape') else (1080, 1920, 3)
            
            right_thumb_tip = right_hand_landmarks.landmark[4]
            right_index_tip = right_hand_landmarks.landmark[8]
            right_thumb_pos = (int(right_thumb_tip.x * w), int(right_thumb_tip.y * h))
            right_index_pos = (int(right_index_tip.x * w), int(right_index_tip.y * h))
            
            right_thumb = np.array([right_thumb_tip.x, right_thumb_tip.y])
            right_index = np.array([right_index_tip.x, right_index_tip.y])
            
            # Calculate midpoint of right pinch line
            right_pinch_midpoint = (right_thumb + right_index) / 2
            
            # Calculate pinch distance
            right_pinch_dist = np.linalg.norm(right_thumb - right_index)
            
            # Direct linear mapping from pinch distance to frequency
            # Pinch 0 -> 20Hz, Pinch 0.400 -> 600Hz
            pinch_max = self.calibration["pinch_max"]  # Should be 0.400
            
            # Clamp pinch distance to valid range
            clamped_pinch = max(0, min(pinch_max, right_pinch_dist))
            
            # Linear mapping from pinch to frequency
            normalized_pinch = clamped_pinch / pinch_max  # 0 to 1 range
            
            # Direct mapping from pinch to frequency (20-600Hz)
            frequency = 20 + normalized_pinch * 580  # 20 to 600 Hz (changed from 60Hz to 20Hz)
            
            # Convert to pitch value (-12 to 12 semitones)
            # We need to convert from frequency to pitch for internal processing
            # The formula is: normalized_pitch = (pitch + 12) / 24
            # So: pitch = normalized_pitch * 24 - 12
            normalized_freq = normalized_pinch  # 0 to 1 for 20Hz to 600Hz
            raw_pitch = normalized_freq * 24 - 12  # -12 to 12 semitones
            
            # Debug logging - every 30 frames
            if self.frame_count % 30 == 0:
                self.log_pinch_debug("right", right_pinch_dist, int(frequency))
            
            # Apply smoothing and convert to Python float
            old_pitch = self.pitch
            self.pitch = self.smooth_value(raw_pitch, self.pitch_history)
        
            # Check if pitch changed significantly
            if abs(self.pitch - old_pitch) > 0.5:
                params_changed = True
            
            # Check for horizontal twist gesture with right hand
            if self.detect_horizontal_twist(right_thumb_pos, right_index_pos, 'Right'):
                # Right hand horizontal twist - next track
                self.next_track()
        
        # Calculate volume based on distance between pinch midpoints
        if left_pinch_midpoint is not None and right_pinch_midpoint is not None:
            # Calculate distance between pinch midpoints
            pinch_midpoint_distance = np.linalg.norm(left_pinch_midpoint - right_pinch_midpoint)
            
            # Map distance to volume using calibration data
            dist_min = self.calibration["distance_min"]
            dist_max = self.calibration["distance_max"]
            dist_range = max(0.001, dist_max - dist_min)  # Avoid division by zero
            
            # Normalize and map to volume range (0.0 to 10.0)
            normalized_dist = (pinch_midpoint_distance - dist_min) / dist_range
            raw_volume = max(0.0, min(1.0, normalized_dist)) * 10.0
            
            # Apply smoothing and convert to Python float
            old_volume = self.volume
            self.volume = self.smooth_value(raw_volume, self.volume_history)
            
            # Check if volume changed significantly
            if abs(self.volume - old_volume) > 0.5:
                params_changed = True
        
        # Log all parameters on one line if any of them changed significantly
        if params_changed:
            self.log_parameters()
    
    def draw_track_change_animation(self, image):
        """Draw a track change animation when switching tracks"""
        if self.track_change_animation == 0 or time.time() - self.track_change_time > 1.5:
            return
        
        h, w, c = image.shape
        
        # Calculate animation progress (0.0 to 1.0)
        progress = min(1.0, (time.time() - self.track_change_time) / 1.0)
        
        # Create animation based on direction
        if self.track_change_animation > 0:  # Next track
            # Simplified text
            arrow_text = "NEXT TRACK"
            arrow_color = (255, 255, 255)  # Changed to white
            
            # Arrow start position moves from left to right
            start_x = int(w * 0.1 + progress * w * 0.8)
            
        else:  # Previous track
            # Simplified text
            arrow_text = "PREV TRACK"
            arrow_color = (255, 255, 255)  # Changed to white
            
            # Arrow start position moves from right to left
            start_x = int(w * 0.9 - progress * w * 0.8)
        
        # Draw text with fading effect
        alpha = int(255 * (1.0 - progress))
        
        # Create overlay for text with fade effect
        overlay = image.copy()
        cv2.putText(overlay, arrow_text, (start_x - 80, h // 2 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, arrow_color, 2)
        
        # Blend based on alpha for fade-out effect
        cv2.addWeighted(overlay, alpha/255, image, 1 - alpha/255, 0, image)
        
        # Reset animation when complete
        if progress >= 1.0:
            self.track_change_animation = 0
    
    def draw_waveform(self, image, start_point, end_point, line_color=(255, 255, 255)):
        """Draw an audio waveform along the volume line"""
        if len(self.waveform_buffer) < 2:
            return
            
        h, w, c = image.shape
        
        # Calculate line parameters
        line_length = np.linalg.norm(np.array(end_point) - np.array(start_point))
        line_angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        
        # Create wave visualization points along the line
        num_points = min(len(self.waveform_buffer), int(line_length / 4))
        
        if num_points < 2:
            return
            
        # Sample the buffer to match the number of points
        indices = np.linspace(0, len(self.waveform_buffer)-1, num_points).astype(int)
        sampled_buffer = [self.waveform_buffer[i] for i in indices]
        
        # Create points along the line with perpendicular offsets based on waveform values
        points = []
        segment_length = line_length / (num_points - 1)
        
        for i, wave_val in enumerate(sampled_buffer):
            # Position along the line
            ratio = i / (num_points - 1)
            x = int(start_point[0] + (end_point[0] - start_point[0]) * ratio)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * ratio)
            
            # Calculate perpendicular offset
            # Wave values are centered at 0, with positive/negative values
            perpendicular_angle = line_angle + np.pi/2
            amplitude_scale = 20  # Maximum pixel offset
            
            # Scale amplitude based on current volume and frequency
            # Dynamic amplitude makes waveform react to audio parameters
            offset_distance = wave_val * amplitude_scale
            
            offset_x = int(offset_distance * np.cos(perpendicular_angle))
            offset_y = int(offset_distance * np.sin(perpendicular_angle))
            
            # Add point with offset
            points.append((x + offset_x, y + offset_y))
        
        # Draw the waveform
        # Color based on frequency (pitch) - blue for low, green for mid, red for high
        freq_value = self.waveform_frequency  # 0-1 normalized frequency
        
        # Create dynamic color based on frequency
        if freq_value < 0.33:
            # Blue to cyan
            r = int(freq_value * 3 * 255)
            g = int(freq_value * 3 * 255)
            b = 255
        elif freq_value < 0.66:
            # Cyan to yellow
            r = int((freq_value - 0.33) * 3 * 255)
            g = 255
            b = int((0.66 - freq_value) * 3 * 255)
        else:
            # Yellow to red
            r = 255
            g = int((1.0 - freq_value) * 3 * 255)
            b = 0
        
        # Adjust color based on volume (amplitude)
        amp_factor = min(1.0, self.waveform_amplitude * 1.2)
        # Brighten colors for higher amplitude
        r = min(255, int(r * (0.6 + amp_factor * 0.4)))
        g = min(255, int(g * (0.6 + amp_factor * 0.4)))
        b = min(255, int(b * (0.6 + amp_factor * 0.4)))
        
        waveform_color = (b, g, r)  # BGR format for OpenCV
        
        # Draw a smoother curve using polylines
        if len(points) > 1:
            # Draw filled area for more impact
            # Create a closed polygon by adding points below the line
            fill_points = points.copy()
            
            # Add points along the main line in reverse order
            for i in range(num_points-1, -1, -1):
                ratio = i / (num_points - 1)
                x = int(start_point[0] + (end_point[0] - start_point[0]) * ratio)
                y = int(start_point[1] + (end_point[1] - start_point[1]) * ratio)
                fill_points.append((x, y))
            
            # Create numpy array for the polygon
            fill_points_array = np.array(fill_points, dtype=np.int32)
            
            # Draw semi-transparent filled area
            overlay = image.copy()
            cv2.fillPoly(overlay, [fill_points_array], waveform_color)
            
            # Apply transparency (alpha blending)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # Draw the waveform line with the same color but solid
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(image, [points_array], False, waveform_color, 2, cv2.LINE_AA)
            
            # Add subtle glow effect for more visual impact
            if self.waveform_amplitude > 0.3:
                # Create blurred version of the line for glow effect
                glow_img = np.zeros_like(image)
                cv2.polylines(glow_img, [points_array], False, waveform_color, 4, cv2.LINE_AA)
                
                # Apply blur to create glow
                glow_amount = int(5 * self.waveform_amplitude)
                if glow_amount > 0:
                    glow_img = cv2.GaussianBlur(glow_img, (glow_amount*2+1, glow_amount*2+1), 0)
                    
                    # Blend the glow with the original image
                    glow_alpha = min(0.7, self.waveform_amplitude * 0.5)
                    cv2.addWeighted(glow_img, glow_alpha, image, 1.0, 0, image)
    
    def update_visualization_data(self):
        """Update visualization data based on current audio parameters and amplitude"""
        # Update amplitude data if possible
        try:
            if hasattr(self, 'peak_detector') and self.peak_detector is not None:
                # Only update periodically to avoid too many calls
                current_time = time.time()
                if current_time - self.last_amplitude_update > 0.05:  # 20 times per second
                    self.last_amplitude_update = current_time
                    amp = self.peak_detector.get()
                    if amp is not None and isinstance(amp, (int, float)):
                        # Scale amplitude by volume
                        self.amplitude_data = amp * (self.volume / 5.0)
                    else:
                        # Fallback to volume-based amplitude if detection fails
                        self.amplitude_data = self.volume / 10.0
        except Exception as e:
            # Fallback to volume-based amplitude
            self.amplitude_data = self.volume / 10.0
        
        # Generate spectrum data based on pitch and amplitude
        # This creates a simulated spectrum that changes with audio parameters
        try:
            # Create a new spectrum array
            new_spectrum = np.zeros(64)
            
            # Calculate frequency emphasis based on pitch
            # Higher pitch = emphasis on higher frequencies
            pitch_normalized = (self.pitch + 12) / 24.0  # 0 to 1
            
            # Create frequency distribution curve
            if pitch_normalized < 0.5:
                # Bass heavy (low frequencies emphasized)
                peak_freq = int(16 * pitch_normalized)  # 0-8
            else:
                # Treble heavy (high frequencies emphasized)
                peak_freq = int(16 + (pitch_normalized - 0.5) * 32)  # 8-24
            
            # Create a spectrum with a peak at the calculated frequency
            for i in range(64):
                # Distance from peak frequency (with wrapping)
                dist = min(abs(i - peak_freq), abs(i - peak_freq - 64))
                
                # Create a falloff from the peak
                if dist == 0:
                    falloff = 1.0
                else:
                    falloff = 1.0 / (dist * 0.5) ** 1.2
                
                # Set the amplitude value with some randomness for liveliness
                rand_factor = 0.7 + 0.3 * random.random()
                
                # Speed affects the spectral spread (faster = more spread)
                speed_factor = max(0.5, min(1.5, self.speed))
                
                # More energy in the spectrum for higher speeds
                energy = self.amplitude_data * rand_factor * speed_factor
                
                # Set the spectrum value
                new_spectrum[i] = min(1.0, falloff * energy)
            
            # Apply peak holding for smoother visualization
            for i in range(64):
                # Decay existing peaks
                self.peaks[i] = max(0, self.peaks[i] - self.peak_decay)
                
                # Update peaks with new data
                if new_spectrum[i] > self.peaks[i]:
                    self.peaks[i] = new_spectrum[i]
                
                # Use peaks for actual spectrum display
                self.spectrum_data[i] = self.peaks[i]
            
        except Exception as e:
            print(f"Error generating spectrum: {e}")
    
    def draw_spectrum_histogram(self, image, start_point, end_point):
        """Draw an audio spectrum histogram along the volume line"""
        h, w, c = image.shape
        
        # Calculate line parameters
        line_vector = np.array(end_point) - np.array(start_point)
        line_length = np.linalg.norm(line_vector)
        line_angle = np.arctan2(line_vector[1], line_vector[0])
        
        # Skip if line is too short
        if line_length < 20:
            return
        
        # Number of histogram bars
        num_bars = min(64, int(line_length / 8))
        
        # Calculate bar width and spacing
        bar_width = max(2, int(line_length / (num_bars * 1.5)))
        spacing = max(1, int((line_length - bar_width * num_bars) / num_bars))
        
        # Resample spectrum data to match number of bars
        if len(self.spectrum_data) != num_bars:
            resampled = np.zeros(num_bars)
            for i in range(num_bars):
                start_idx = int(i * len(self.spectrum_data) / num_bars)
                end_idx = int((i + 1) * len(self.spectrum_data) / num_bars)
                if start_idx < len(self.spectrum_data) and end_idx <= len(self.spectrum_data):
                    resampled[i] = np.max(self.spectrum_data[start_idx:end_idx])
            histogram_data = resampled
        else:
            histogram_data = self.spectrum_data
        
        # Scale histogram data based on current audio parameters
        max_height = 80.0  # Maximum bar height in pixels
        
        # Make bars more dynamic with volume and speed
        volume_boost = self.volume / 5.0  # Normalize around 1.0
        speed_impact = self.speed / 1.0  # Normalize around 1.0
        
        # Combined scaling factor
        scale_factor = volume_boost * (0.8 + speed_impact * 0.4)
        
        # Create bars
        for i in range(num_bars):
            # Position along the line
            ratio = i / (num_bars - 1)
            
            # Calculate bar position (center)
            center_x = int(start_point[0] + line_vector[0] * ratio)
            center_y = int(start_point[1] + line_vector[1] * ratio)
            
            # Get bar height from spectrum data (scaled)
            value = min(1.0, histogram_data[i] * scale_factor)
            bar_height = int(max_height * value)
            
            # Calculate bar endpoints (perpendicular to line)
            perp_angle = line_angle + np.pi/2
            
            # Bar grows symmetrically from center line
            bar_top_x = int(center_x + bar_height * np.cos(perp_angle))
            bar_top_y = int(center_y + bar_height * np.sin(perp_angle))
            bar_bottom_x = int(center_x - bar_height * np.cos(perp_angle))
            bar_bottom_y = int(center_y - bar_height * np.sin(perp_angle))
            
            # Create color based on frequency and amplitude
            # Higher frequencies shift toward red, lower toward blue
            freq_ratio = i / num_bars
            amp_ratio = value
            
            # Create dynamic color based on frequency
            if freq_ratio < 0.33:
                # Blue to cyan
                r = int(freq_ratio * 3 * 255)
                g = int(freq_ratio * 3 * 255)
                b = 255
            elif freq_ratio < 0.66:
                # Cyan to yellow
                r = int((freq_ratio - 0.33) * 3 * 255)
                g = 255
                b = int((0.66 - freq_ratio) * 3 * 255)
            else:
                # Yellow to red
                r = 255
                g = int((1.0 - freq_ratio) * 3 * 255)
                b = 0
            
            # Adjust color intensity based on amplitude
            intensity = 0.4 + amp_ratio * 0.6
            r = min(255, int(r * intensity))
            g = min(255, int(g * intensity))
            b = min(255, int(b * intensity))
            
            # Draw the bar
            bar_color = (b, g, r)  # BGR for OpenCV
            
            # Draw filled rectangle for each bar
            points = np.array([
                [center_x - bar_width//2, center_y],
                [bar_top_x - bar_width//2, bar_top_y],
                [bar_top_x + bar_width//2, bar_top_y],
                [center_x + bar_width//2, center_y]
            ], np.int32)
            
            # Draw filled top half
            cv2.fillPoly(image, [points], bar_color)
            
            # Draw bottom half
            points = np.array([
                [center_x - bar_width//2, center_y],
                [bar_bottom_x - bar_width//2, bar_bottom_y],
                [bar_bottom_x + bar_width//2, bar_bottom_y],
                [center_x + bar_width//2, center_y]
            ], np.int32)
            
            cv2.fillPoly(image, [points], bar_color)
            
            # Add glow effect for higher amplitudes
            if amp_ratio > 0.7:
                glow_img = np.zeros_like(image)
                glow_pts = np.array([
                    [center_x - bar_width, center_y],
                    [bar_top_x - bar_width, bar_top_y],
                    [bar_top_x + bar_width, bar_top_y],
                    [center_x + bar_width, center_y],
                    [bar_bottom_x + bar_width, bar_bottom_y],
                    [bar_bottom_x - bar_width, bar_bottom_y]
                ], np.int32)
                
                cv2.fillPoly(glow_img, [glow_pts], bar_color)
                
                # Apply blur for glow effect
                glow_size = int(5 * amp_ratio)
                if glow_size > 0:
                    glow_img = cv2.GaussianBlur(glow_img, (glow_size*2+1, glow_size*2+1), 0)
                    
                    # Blend with original image
                    glow_alpha = min(0.4, amp_ratio * 0.3)
                    cv2.addWeighted(glow_img, glow_alpha, image, 1.0, 0, image)
    
    def run(self):
        try:
            print("\nHand DJ started!")
            print("Controls:")
            print("  - Left hand pinch: Speed control (0.1x to 2.0x)")
            print("  - Right hand pinch: Pitch control (20Hz to 600Hz)")
            print("  - Distance between hands: Volume (0-10)")
            print("  - Right hand horizontal twist: Next track")
            print("  - Left hand twist to the left: Previous track")
            print("  - Press 'q' to quit")
            print("  - Press 'r' to reset all parameters to default")
            print("  - Press 'n' for next track, 'p' for previous track")
            
            if len(self.playlist) > 1:
                print(f"\nPlaylist loaded with {len(self.playlist)} tracks")
                print(f"Current track: {os.path.basename(self.playlist[self.current_track_index])}")
            
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Failed to capture video frame.")
                    break
                
                # Store image shape for gesture detection
                self.image_shape = image.shape
                
                # Flip the image horizontally for a mirror effect
                image = cv2.flip(image, 1)
                
                # Convert the BGR image to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect hands
                results = self.hands.process(rgb_image)
                
                # Draw hand landmarks on the image
                if results.multi_hand_landmarks:
                    left_hand_landmarks = None
                    right_hand_landmarks = None
                    
                    # Identify left and right hands
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        handedness = results.multi_handedness[hand_idx].classification[0].label
                        
                        # Store landmarks for specific hand
                        if handedness == 'Left':  # Camera is mirrored, so this is the actual LEFT hand
                            left_hand_landmarks = hand_landmarks
                        elif handedness == 'Right':  # Camera is mirrored, so this is the actual RIGHT hand
                            right_hand_landmarks = hand_landmarks
                        
                        # Don't draw all hand landmarks, just the important points for controls
                        
                        # Draw circle at thumb and index tips
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        
                        h, w, c = image.shape
                        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                        
                        # White color for all elements
                        color = (255, 255, 255)
                        
                        # Calculate pinch distance
                        pinch_dist = np.linalg.norm(
                            np.array([thumb_tip.x, thumb_tip.y]) - 
                            np.array([index_tip.x, index_tip.y])
                        )
                        
                        # Calculate normalized value for visual feedback
                        normalized = min(1.0, pinch_dist / self.calibration["pinch_max"])
                        
                        # Determine if we're at "normal" levels
                        if handedness == 'Left':
                            # Speed control
                            speed_val = 0.1 + normalized * 1.9
                            # Check if close to default (1.0)
                            is_default = abs(speed_val - 1.0) < 0.1
                            
                            # Visual feedback based on distance from default
                            if is_default:
                                # At or near default - use a special indicator
                                inner_color = (100, 255, 100)  # Light green for "normal"
                                outer_size = 14
                                cv2.circle(image, thumb_pos, outer_size, inner_color, 2)
                                cv2.circle(image, index_pos, outer_size, inner_color, 2)
                            else:
                                # Show how far from default with color intensity
                                deviation = abs(speed_val - 1.0) / 0.9  # Normalized deviation
                                
                                # Blend between white and gold based on deviation
                                if speed_val < 1.0:
                                    # Slower than normal - blue hue
                                    color_intensity = int(200 * deviation) + 55
                                    inner_color = (color_intensity, 200, 100)  # Bluish
                                else:
                                    # Faster than normal - orange hue
                                    color_intensity = int(200 * deviation) + 55
                                    inner_color = (50, 150, color_intensity)  # Orangish
                            
                            # Updated display format - title over value
                            label_text = "SPEED"
                            value_text = f"{speed_val:.1f}x"
                        else:
                            # Pitch control
                            freq_val = 20 + normalized * 580
                            # Base frequency when pitch is 0
                            default_freq = 310  # The middle frequency
                            
                            # Check if close to default
                            is_default = abs(freq_val - default_freq) < 40
                            
                            # Visual feedback based on distance from default
                            if is_default:
                                # At or near default - use a special indicator
                                inner_color = (100, 255, 100)  # Light green for "normal"
                                outer_size = 14
                                cv2.circle(image, thumb_pos, outer_size, inner_color, 2)
                                cv2.circle(image, index_pos, outer_size, inner_color, 2)
                            else:
                                # Show how far from default with color intensity
                                deviation = min(1.0, abs(freq_val - default_freq) / 290)  # Normalized deviation
                                
                                # Blend between white and color based on deviation
                                if freq_val < default_freq:
                                    # Lower than normal - purple hue
                                    color_intensity = int(200 * deviation) + 55
                                    inner_color = (color_intensity, 100, color_intensity)  # Purplish
                                else:
                                    # Higher than normal - yellowish hue
                                    color_intensity = int(200 * deviation) + 55
                                    inner_color = (50, color_intensity, color_intensity)  # Yellowish
                            
                            # Updated display format - title over value
                            label_text = "PITCH"
                            value_text = f"{int(freq_val)}Hz"
                        
                        # Draw circles on finger tips with visual feedback color
                        cv2.circle(image, thumb_pos, 10, inner_color, -1)
                        cv2.circle(image, index_pos, 10, inner_color, -1)
                        
                        # Draw pinch line with feedback color
                        cv2.line(image, thumb_pos, index_pos, inner_color, 2)
                        
                        # New display with label on top, value underneath
                        label_pos = (thumb_pos[0] - 70, thumb_pos[1] + 30)
                        value_pos = (thumb_pos[0] - 70, thumb_pos[1] + 70)  # Increased to 40 unit spacing
                        
                        # Draw the label and value
                        cv2.putText(image, label_text, label_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(image, value_text, value_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    # Process hand gestures to update audio parameters
                    self.process_hands(left_hand_landmarks, right_hand_landmarks)
                    self.update_audio_params()
                    
                    # If both hands detected, draw a line between them for volume
                    if left_hand_landmarks and right_hand_landmarks:
                        # Get thumb and index positions for both hands
                        left_thumb_tip = left_hand_landmarks.landmark[4]
                        left_index_tip = left_hand_landmarks.landmark[8]
                        right_thumb_tip = right_hand_landmarks.landmark[4]
                        right_index_tip = right_hand_landmarks.landmark[8]
                        
                        h, w, c = image.shape
                        
                        # Calculate midpoints of pinch lines
                        left_midpoint = (
                            int((left_thumb_tip.x + left_index_tip.x) * w / 2),
                            int((left_thumb_tip.y + left_index_tip.y) * h / 2)
                        )
                        
                        right_midpoint = (
                            int((right_thumb_tip.x + right_index_tip.x) * w / 2),
                            int((right_thumb_tip.y + right_index_tip.y) * h / 2)
                        )
                        
                        # Draw midpoints with white color
                        cv2.circle(image, left_midpoint, 5, (255, 255, 255), -1)
                        cv2.circle(image, right_midpoint, 5, (255, 255, 255), -1)
                        
                        # Draw audio spectrum histogram instead of waveform
                        self.draw_spectrum_histogram(image, left_midpoint, right_midpoint)
                        
                        # Calculate midpoint for volume label
                        mid_x = (left_midpoint[0] + right_midpoint[0]) // 2
                        mid_y = (left_midpoint[1] + right_midpoint[1]) // 2
                        
                        # Show midpoint distance for volume
                        midpoint_distance = np.linalg.norm(
                            np.array([left_midpoint[0], left_midpoint[1]]) / np.array([w, h]) - 
                            np.array([right_midpoint[0], right_midpoint[1]]) / np.array([w, h])
                        )
                        
                        # Display volume value based on distance with white color
                        normalized = min(1.0, max(0.0, (midpoint_distance - self.calibration["distance_min"]) / 
                                             (self.calibration["distance_max"] - self.calibration["distance_min"])))
                        volume_val = normalized * 10.0
                        
                        # Keep volume line and label white
                        vol_color = (255, 255, 255)  # White for volume
                        
                        # Don't draw the volume line anymore
                        # Only draw the midpoints for volume calculation reference
                        cv2.circle(image, left_midpoint, 5, vol_color, -1)
                        cv2.circle(image, right_midpoint, 5, vol_color, -1)
                        
                        # Updated display format - title over value
                        vol_label = "VOLUME"
                        vol_value = f"{volume_val:.1f}"
                        
                        # Positions for the text - centered above volume bar
                        label_pos = (mid_x - 40, mid_y - 70)  # Centered text, higher position
                        value_pos = (mid_x - 15, mid_y - 30)  # Centered value, 40 units lower
                        
                        # Draw the label and value
                        cv2.putText(image, vol_label, label_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, vol_color, 2)
                        cv2.putText(image, vol_value, value_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, vol_color, 2)
                        
                        # Draw a simple audio level indicator with white fill
                        bar_width = 150
                        bar_height = 10
                        bar_x = mid_x - bar_width // 2
                        # Move bar significantly lower
                        bar_y = mid_y + 50
                        
                        # Draw background bar
                        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                        
                        # Draw volume level in white
                        fill_width = int(volume_val / 10.0 * bar_width)
                        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), vol_color, -1)
                        
                        # Mark normal position (5.0)
                        normal_x = bar_x + bar_width // 2
                        cv2.line(image, (normal_x, bar_y - 3), (normal_x, bar_y + bar_height + 3), (255, 255, 255), 1)
                else:
                    # Show "No hands detected" when no hands are visible with white color
                    cv2.putText(image, "No hands detected - Show both hands to camera", 
                               (image.shape[1]//2 - 200, image.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add help text with white color
                cv2.putText(image, "Press 'q' to quit | 'r' to reset", (10, image.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add track change animation if active
                self.draw_track_change_animation(image)
                
                # Display the image
                cv2.imshow('Hand DJ', image)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF  # Reduced from 5ms to 1ms for ultra-low latency
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_parameters()
                elif key == ord('n'):
                    self.next_track()
                elif key == ord('p'):
                    self.prev_track()
                    
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Stop audio server
            self.server.stop()

if __name__ == "__main__":
    # Use provided audio file or default to sine wave
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    dj = HandDJ(audio_file)
    dj.run() 