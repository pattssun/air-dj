#!/usr/bin/env python3
import cv2
import numpy as np
import pyo
from pyo import Server, SndTable, TableRead, Sine, SfPlayer, Harmonizer, Biquad, STRev, Mix
import time
import sys
import os

class SimpleHandDJ:
    """
    A simplified version of HandDJ that uses color tracking instead of MediaPipe.
    This is more compatible with systems that have issues with TensorFlow/MediaPipe.
    """
    def __init__(self, audio_file=None):
        """Initialize the application"""
        print("Initializing SimpleHandDJ...")
        
        # Initialize parameters
        self.running = True
        self.show_mask = False  # Toggle for showing color mask
        self.red_center = None
        self.blue_center = None
        
        # Color ranges for tracking
        # Blue color range in HSV
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # Red color range in HSV (red wraps around the hue spectrum)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Audio initialization
        self.server = None  # Will be initialized in setup_audio_server
        self.audio_path = audio_file
        
        # Smoothing parameters for gesture control
        self.speed_history = [1.0] * 5
        self.pitch_history = [0] * 5
        self.volume_history = [0.5] * 5
        
        # Video capture setup
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        
        if not ret:
            raise Exception("Could not access webcam. Please check your camera connection.")
        
        self.frame_height, self.frame_width = frame.shape[:2]
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
    
    def setup_audio_server(self):
        """Setup the audio server"""
        self.server = Server(duplex=0).boot()
        self.server.start()
        
    def setup_audio_player(self, sound_file=None):
        """Setup the audio player with the provided sound file or a default sine wave"""
        if self.server is None:
            self.setup_audio_server()
        
        self.audio_params = {
            'volume': 0.5,  # Overall volume (0-1)
            'pitch': 0.0,   # Pitch shifting in semitones (-12 to +12)
            'speed': 1.0,   # Playback speed (0.25-3.0)
            'filter': 0.5   # Low-pass filter cutoff (0-1)
        }
        
        # Try to load the audio file
        success = self.try_load_audio(sound_file)

        # Create a filter for additional control
        if hasattr(self, 'audio_player'):
            # Ensure the filter works on the proper audio source
            self.filter = Biquad(self.audio_player, freq=5000, q=1, type=0)
            # Set the filter as our output to ensure its effects are heard
            self.output = self.filter.out()
        
        # Ensure our audio is actually playing
        print("Starting audio playback...")
    
    def use_sine_wave(self):
        """Switch to sine wave audio source"""
        print("Using sine wave as audio source")
        self.sine = Sine(freq=440, mul=0.3)
        self.output = self.sine.out()
        self.audio_path = None
    
    def try_load_audio(self, sound_file=None):
        """Try multiple methods to load audio file, falling back to sine wave generation if all fail"""
        if sound_file and os.path.exists(sound_file):
            # Check if there's a WAV version of the same file
            wav_file = os.path.splitext(sound_file)[0] + ".wav"
            
            # Prefer WAV over MP3 if it exists
            if os.path.exists(wav_file):
                print(f"Found WAV version of the file: {wav_file}")
                sound_file = wav_file
            
            print(f"Loading audio file: {sound_file}")
            file_ext = os.path.splitext(sound_file)[1].lower()
            
            try:
                # Try SfPlayer for MP3 files with high quality settings
                if file_ext == '.mp3':
                    print("Using SfPlayer for MP3 file")
                    self.audio_table = SfPlayer(sound_file, loop=True, interp=4, mul=0.8)
                    self.audio_player = self.audio_table
                    # Add a Harmonizer for pitch shifting
                    self.pitch_shifter = Harmonizer(self.audio_player, transpo=0, mul=0.8)
                    self.audio_player = self.pitch_shifter
                    # Add reverb for better sound quality
                    self.reverb = STRev(self.audio_player, revtime=1.0, cutoff=10000, bal=0.1).out()
                    print("MP3 loaded successfully with SfPlayer")
                    return True
                
                # Use SndTable for WAV and other formats with high quality settings
                else:
                    print("Using SndTable for audio file")
                    self.audio_table = SndTable(sound_file)
                    # Store base rate for speed calculation
                    base_rate = self.audio_table.getRate()
                    print(f"Audio loaded with base rate: {base_rate} Hz")
                    
                    # Use a TableRead for speed control
                    self.table_player = TableRead(
                        table=self.audio_table, 
                        freq=base_rate,  # Initial rate matches the file
                        loop=True,
                        interp=4,
                        mul=0.8
                    )
                    self.audio_player = self.table_player
                    
                    # Add a Harmonizer for pitch control
                    self.pitch_shifter = Harmonizer(self.audio_player, transpo=0, mul=0.8)
                    self.audio_player = self.pitch_shifter
                    
                    # Add reverb for better sound quality
                    self.reverb = STRev(self.audio_player, revtime=1.0, cutoff=10000, bal=0.1)
                    self.audio_player = self.reverb
                    
                    # Start playing
                    self.audio_player.out()
                    
                    print("Audio file loaded successfully with SndTable and TableRead")
                    return True
            except Exception as e:
                print(f"Error loading audio file: {e}")
        
        # Fallback to sine wave if file loading fails
        print("Falling back to sine wave generation with harmonics")
        # Create a richer sine wave with some harmonics for better quality
        self.audio_table = Sine(freq=440, mul=0.3)
        self.harmonic1 = Sine(freq=880, mul=0.15)  # First harmonic
        self.harmonic2 = Sine(freq=1320, mul=0.08)  # Second harmonic
        self.mixer = Mix([self.audio_table, self.harmonic1, self.harmonic2], voices=3)
        self.audio_player = self.mixer
        # Add a slight reverb for better sound
        self.reverb = STRev(self.audio_player, revtime=0.8, cutoff=8000, bal=0.1)
        self.audio_player = self.reverb
        self.audio_player.out()
        return False
    
    def update_audio_params(self, position_red, position_blue):
        """Update audio parameters based on the position of the colored objects"""
        try:
            # Ensure all values are native Python floats
            volume = float(self.audio_params['volume'])
            pitch = float(self.audio_params['pitch'])
            speed = float(self.audio_params['speed'])
            
            print(f"DEBUG - Speed: {speed}, Pitch: {pitch}, Volume: {volume}")
            
            # Update volume
            if hasattr(self, 'audio_player'):
                self.audio_player.mul = volume
                
                # Also update sine wave harmonics if we're using them
                if hasattr(self, 'audio_table') and isinstance(self.audio_table, Sine):
                    self.audio_table.mul = volume * 0.6  # Adjusted for more volume
                    if hasattr(self, 'harmonic1'):
                        self.harmonic1.mul = volume * 0.3  # Adjusted for harmonics
                    if hasattr(self, 'harmonic2'):
                        self.harmonic2.mul = volume * 0.15  # Adjusted for harmonics
            
            # Update pitch using the transpo attribute rather than a method
            if hasattr(self, 'pitch_shifter') and isinstance(self.pitch_shifter, Harmonizer):
                self.pitch_shifter.transpo = pitch
            
            # Update speed (for TableRead) - THIS IS THE CRITICAL PART FOR SPEED CONTROL
            if hasattr(self, 'table_player') and isinstance(self.table_player, TableRead):
                if hasattr(self, 'audio_table') and hasattr(self.audio_table, 'getRate'):
                    try:
                        base_rate = self.audio_table.getRate()
                        print(f"DEBUG - Base rate: {base_rate}, New rate: {base_rate * speed}")
                        self.table_player.freq = base_rate * speed
                    except Exception as e:
                        print(f"Error updating table player speed: {e}")
            
            # Update filter
            if hasattr(self, 'filter') and position_blue:
                # Calculate filter cutoff from the y-position of the blue object
                filter_factor = 1.0 - (position_blue[1] / self.frame_height)
                filter_freq = 100 + (filter_factor * 14900)  # Map to frequency range (100-15000Hz)
                self.filter.freq = float(filter_freq)
            
            # Apply dynamic reverb based on speed
            if hasattr(self, 'reverb') and isinstance(self.reverb, STRev):
                # More reverb for slower speeds
                rev_time = 1.0 + (1.0 - min(1.0, speed)) * 2.0
                self.reverb.revtime = rev_time
                
                # Adjust filter cutoff based on pitch
                pitch_norm = (pitch + 12) / 24.0  # Normalize pitch to 0-1
                cutoff = 2000 + pitch_norm * 8000  # Map to frequency range
                self.reverb.cutoff = float(cutoff)
        except Exception as e:
            print(f"Error in update_audio_params: {e}")
    
    def smooth_value(self, new_value, history_list):
        """Apply smoothing to reduce jitter"""
        try:
            history_list.pop(0)
            history_list.append(float(new_value))  # Convert to native Python float
            return float(sum(history_list) / len(history_list))
        except Exception as e:
            print(f"Error in smooth_value: {e}")
            return history_list[-1]  # Return the last value on error
    
    def find_colored_objects(self, frame):
        """
        Find blue and red objects in the frame
        Returns center points of the largest blue and red objects
        """
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find blue objects
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        blue_mask = cv2.erode(blue_mask, None, iterations=2)
        blue_mask = cv2.dilate(blue_mask, None, iterations=2)
        
        # Find red objects (need to combine two ranges for red in HSV)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.erode(red_mask, None, iterations=2)
        red_mask = cv2.dilate(red_mask, None, iterations=2)
        
        # Capture the masks if toggle is on
        if self.show_mask:
            combined_mask = cv2.bitwise_or(blue_mask, red_mask)
            cv2.imshow('Color Mask', combined_mask)
        
        # Find contours
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_center = None
        red_center = None
        
        # Create a semi-transparent overlay for parameter labels
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Add parameter labels with clearer annotations
        cv2.putText(frame, "PARAMETERS:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Speed label (red object)
        speed_text = f"SPEED: {self.audio_params['speed']:.2f}x"
        if abs(self.audio_params['speed'] - 1.0) < 0.1:
            speed_text += " (NORMAL)"
        cv2.putText(frame, speed_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate semitones for display (from the 0.25-3.0 pitch range)
        pitch_semitones = self.audio_params['pitch']  # Already in semitones
        
        # Pitch label (blue object)
        pitch_text = f"PITCH: {pitch_semitones:.1f} semitones"
        if abs(pitch_semitones) < 1.0:
            pitch_text += " (NORMAL)"
        cv2.putText(frame, pitch_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Volume label (distance)
        volume_text = f"VOLUME: {self.audio_params['volume']:.2f}"
        if abs(self.audio_params['volume'] - 0.5) < 0.1:
            volume_text += " (NORMAL)"
        cv2.putText(frame, volume_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Process blue contours
        if len(blue_contours) > 0:
            # Find the largest blue contour
            c = max(blue_contours, key=cv2.contourArea)
            
            # Only process if contour is large enough
            if cv2.contourArea(c) > 100:
                # Calculate center point
                M = cv2.moments(c)
                if M["m00"] > 0:
                    blue_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Draw contour and center point
                    cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
                    cv2.circle(frame, blue_center, 7, (255, 255, 255), -1)
                    
                    # Draw parameter label above blue object
                    cv2.putText(frame, "BLUE - PITCH CONTROL", 
                               (blue_center[0] - 80, blue_center[1] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Show coordinates
                    coord_text = f"Coords: ({blue_center[0]}, {blue_center[1]})"
                    cv2.putText(frame, coord_text, 
                               (blue_center[0] - 80, blue_center[1] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Process red contours
        if len(red_contours) > 0:
            # Find the largest red contour
            c = max(red_contours, key=cv2.contourArea)
            
            # Only process if contour is large enough
            if cv2.contourArea(c) > 100:
                # Calculate center point
                M = cv2.moments(c)
                if M["m00"] > 0:
                    red_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Draw contour and center point
                    cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)
                    cv2.circle(frame, red_center, 7, (255, 255, 255), -1)
                    
                    # Draw parameter label above red object
                    cv2.putText(frame, "RED - SPEED CONTROL", 
                               (red_center[0] - 80, red_center[1] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Show coordinates
                    coord_text = f"Coords: ({red_center[0]}, {red_center[1]})"
                    cv2.putText(frame, coord_text, 
                               (red_center[0] - 80, red_center[1] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw a line between the objects if both are detected for volume control
        if blue_center and red_center:
            cv2.line(frame, blue_center, red_center, (0, 255, 0), 2)
            
            # Calculate and display distance
            distance = np.sqrt((blue_center[0] - red_center[0])**2 + 
                              (blue_center[1] - red_center[1])**2)
            
            midpoint = ((blue_center[0] + red_center[0])//2, 
                        (blue_center[1] + red_center[1])//2)
            
            # Add "VOLUME" label to the line
            cv2.putText(frame, "DISTANCE - VOLUME CONTROL", 
                       (midpoint[0] - 120, midpoint[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show distance value
            dist_text = f"Distance: {distance:.1f} px"
            cv2.putText(frame, dist_text, midpoint, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        if not blue_center and not red_center:
            # Show "No objects detected" message
            cv2.putText(frame, "No colored objects detected - Show blue and red objects to camera", 
                       (frame.shape[1]//2 - 300, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, blue_center, red_center
    
    def update_parameters(self, blue_center, red_center):
        """Update audio parameters based on detected objects"""
        try:
            # Update pitch based on blue object (horizontal position)
            if blue_center:
                # Normalize x-position to 0-1 range
                normalized_x = blue_center[0] / self.frame_width
                
                # Medium position (center) should be normal pitch (0)
                # Map [0, 0.5] to [-12, 0] and [0.5, 1] to [0, 12]
                if normalized_x < 0.5:
                    # Left half - negative pitch
                    pitch_value = -12 + (normalized_x * 2) * 12  # Map [0, 0.5] to [-12, 0]
                else:
                    # Right half - positive pitch
                    pitch_value = (normalized_x - 0.5) * 2 * 12  # Map [0.5, 1] to [0, 12]
                
                # Ensure we get Python float, not numpy.float64
                self.audio_params['pitch'] = self.smooth_value(pitch_value, self.pitch_history)
            
            # Update speed based on red object (vertical position)
            if red_center:
                # Normalize y-position to 0-1 range (invert so higher = faster)
                normalized_y = 1 - (red_center[1] / self.frame_height)
                
                # Medium position (center height) should be normal speed (1.0)
                # Map [0, 0.5] to [0.25, 1.0] and [0.5, 1] to [1.0, 3.0]
                if normalized_y < 0.5:
                    # Lower half - slower
                    speed_value = 0.25 + (normalized_y * 2) * 0.75  # Map [0, 0.5] to [0.25, 1.0]
                else:
                    # Upper half - faster
                    speed_value = 1.0 + (normalized_y - 0.5) * 2 * 2.0  # Map [0.5, 1] to [1.0, 3.0]
                
                # Ensure we get Python float, not numpy.float64
                self.audio_params['speed'] = self.smooth_value(speed_value, self.speed_history)
            
            # Update volume based on distance between objects
            if blue_center and red_center:
                # Calculate Euclidean distance
                distance = np.sqrt((blue_center[0] - red_center[0])**2 + 
                                (blue_center[1] - red_center[1])**2)
                
                # Normalize distance (based on diagonal length of frame)
                max_distance = np.sqrt(self.frame_width**2 + self.frame_height**2)
                normalized_distance = distance / max_distance
                
                # Map to volume range (0.0 to 1.0) with increased sensitivity
                # We'll use a power curve for better control
                raw_volume = min(1.0, (normalized_distance * 3) ** 1.5)  # Increased weight with power curve
                self.audio_params['volume'] = self.smooth_value(raw_volume, self.volume_history)
        except Exception as e:
            print(f"Error updating parameters: {e}")
    
    def run(self):
        try:
            print("\nSimple Hand DJ started!")
            print("Instructions:")
            print("  - Use BLUE colored object to control pitch (left=lower, center=normal, right=higher)")
            print("  - Use RED colored object to control speed (center=normal, higher=faster, lower=slower)")
            print("  - Distance between objects controls volume")
            print("  - Press 'q' to quit")
            print("  - Press 'm' to toggle color mask view")
            print("\nTip: Use colored paper, markers, or objects for tracking.")
            
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Failed to capture video frame.")
                    break
                
                # Find colored objects in the frame
                frame, blue_center, red_center = self.find_colored_objects(frame)
                
                # Update audio parameters based on object positions
                self.update_parameters(blue_center, red_center)
                
                # Update audio engine
                self.update_audio_params(red_center, blue_center)
                
                # Add help text
                cv2.putText(frame, "Press 'q' to quit | 'm' to toggle mask view", 
                           (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the image
                cv2.imshow('Simple Hand DJ', frame)
                
                # Check key presses
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.show_mask = not self.show_mask
                    if not self.show_mask and cv2.getWindowProperty('Color Mask', cv2.WND_PROP_VISIBLE) > 0:
                        cv2.destroyWindow('Color Mask')
                    
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Stop audio server
            if self.server:
                self.server.stop()

if __name__ == "__main__":
    # Use provided audio file or default to sine wave
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    dj = SimpleHandDJ(audio_file)
    dj.setup_audio_player(audio_file)
    dj.run() 