import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import json
import os
import logging
from pyo import *

# Import specific pyo modules for better clarity
from pyo import Server, SndTable, TableRead, Sine, SfPlayer, Harmonizer, STRev, Mix, SigTo, Sig

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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
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
        
        # Smoothing parameters for gesture control
        self.speed_history = [1.0] * 5
        self.pitch_history = [0] * 5
        self.volume_history = [5.0] * 5
        
        # Video capture setup
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera is opened correctly
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
    
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
        except Exception as e:
            print(f"Error updating audio: {e}")
    
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
    
    def process_hands(self, left_hand_landmarks, right_hand_landmarks):
        # Track if any parameter has changed significantly
        params_changed = False
        
        # Store midpoints of pinch lines for volume calculation
        left_pinch_midpoint = None
        right_pinch_midpoint = None
        
        # Process left hand for speed control (thumb-index pinch)
        if left_hand_landmarks:
            left_thumb = np.array([
                left_hand_landmarks.landmark[4].x,
                left_hand_landmarks.landmark[4].y
            ])
            left_index = np.array([
                left_hand_landmarks.landmark[8].x,
                left_hand_landmarks.landmark[8].y
            ])
            
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
        
        # Process right hand for pitch/frequency control
        if right_hand_landmarks:
            right_thumb = np.array([
                right_hand_landmarks.landmark[4].x,
                right_hand_landmarks.landmark[4].y
            ])
            right_index = np.array([
                right_hand_landmarks.landmark[8].x,
                right_hand_landmarks.landmark[8].y
            ])
            
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
    
    def run(self):
        try:
            print("\nHand DJ started!")
            print("Controls:")
            print("  - Left hand pinch: Speed control (0.1x to 2.0x)")
            print("  - Right hand pinch: Pitch control (20Hz to 600Hz)")
            print("  - Distance between hands: Volume (0-10)")
            print("  - Press 'q' to quit")
            print("  - Press 'r' to reset all parameters to default")
            
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Failed to capture video frame.")
                    break
                
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
                        
                        # Draw circles on finger tips
                        cv2.circle(image, thumb_pos, 10, color, -1)
                        cv2.circle(image, index_pos, 10, color, -1)
                        
                        # Draw pinch line
                        cv2.line(image, thumb_pos, index_pos, color, 2)
                        
                        # Show pinch distance and coordinates
                        pinch_dist = np.linalg.norm(
                            np.array([thumb_tip.x, thumb_tip.y]) - 
                            np.array([index_tip.x, index_tip.y])
                        )
                        
                        # Display mapped value
                        if handedness == 'Left':
                            # Map pinch to speed
                            normalized = min(1.0, pinch_dist / self.calibration["pinch_max"])
                            speed_val = 0.1 + normalized * 1.9
                            value_text = f"Speed: {speed_val:.1f}x"
                        else:
                            # Map pinch to frequency
                            normalized = min(1.0, pinch_dist / self.calibration["pinch_max"])
                            freq_val = 20 + normalized * 580
                            value_text = f"Pitch: {int(freq_val)}Hz"
                            
                        # Larger, more visible parameter labels at hand points with white color
                        cv2.putText(image, value_text, 
                                   (thumb_pos[0] - 80, thumb_pos[1] + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
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
                        
                        # Draw line between midpoints for volume with white color
                        cv2.line(image, left_midpoint, right_midpoint, (255, 255, 255), 2)
                        
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
                        vol_text = f"Volume: {volume_val:.1f}"
                        
                        # Larger, more visible volume label with white color
                        cv2.putText(image, vol_text, 
                                   (mid_x - 70, mid_y - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                else:
                    # Show "No hands detected" when no hands are visible with white color
                    cv2.putText(image, "No hands detected - Show both hands to camera", 
                               (image.shape[1]//2 - 200, image.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add help text with white color
                cv2.putText(image, "Press 'q' to quit | 'r' to reset", (10, image.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the image
                cv2.imshow('Hand DJ', image)
                
                # Check for key presses
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_parameters()
                    
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