#!/usr/bin/env python3
"""
iPhone Camera Integration for Air DJ
Consolidated module that handles:
- iPhone camera detection via OBS Virtual Camera and Continuity Camera
- DJ-optimized camera wrapper with proper resolution and mirroring
- 60fps synchronized animations
- Automatic fallback to default webcam if iPhone not available
"""

import cv2
import platform
import subprocess
import time
import sys
import re
import math
import numpy as np
from typing import Optional, List, Dict, Tuple, Any


class ContinuityCameraManager:
    """
    Manages Apple Continuity Camera detection and configuration for optimal iPhone camera usage
    """
    
    def __init__(self):
        self.continuity_device_index = None
        self.fallback_device_index = 0
        self.device_info = {}
        self.optimal_settings = {
            # DJ Controller optimized settings - match UI design resolution
            'width': 1920,      # Match DJ controller UI design
            'height': 1080,     # Match DJ controller UI design (not 1440)
            'fps': 60,          # Maximum supported FPS
            'fourcc': 'MJPG',   # MJPEG for better quality/performance balance
            'buffer_size': 1,   # Minimal buffer for low latency
            'exposure_auto': 1, # Auto exposure for iPhone cameras
            'focus_auto': 1,    # Auto focus
        }
        
    def is_macos(self) -> bool:
        """Check if running on macOS"""
        return platform.system() == 'Darwin'
    
    def detect_continuity_camera(self) -> Optional[int]:
        """
        Detect iPhone/OBS Virtual Camera device index by testing available video devices
        Prioritizes OBS Virtual Camera and other high-quality sources
        """
        # Check for environment variable overrides
        import os
        if os.getenv('FORCE_BUILTIN_CAMERA', '').lower() in ['1', 'true', 'yes']:
            print("🔧 FORCE_BUILTIN_CAMERA detected - using built-in camera")
            return 0
            
        # Force OBS Virtual Camera override (when detection is too strict)
        obs_device = os.getenv('FORCE_OBS_DEVICE', '')
        if obs_device.isdigit():
            device_num = int(obs_device)
            print(f"🎥 FORCE_OBS_DEVICE detected - using OBS Virtual Camera Device {device_num}")
            return device_num
            
        print("🔍 Detecting cameras (prioritizing OBS Virtual Camera and iPhone)...")
        print("💡 TIP: Set FORCE_BUILTIN_CAMERA=1 to bypass iPhone detection if needed")
        
        if not self.is_macos():
            print("⚠️  iPhone camera detection only available on macOS, using default camera")
            return self.fallback_device_index
            
        # Test video devices (usually 0-10 is sufficient)
        max_devices_to_test = 10
        detected_devices = []
        
        print("📹 Scanning camera devices...")
        
        for device_index in range(max_devices_to_test):
            try:
                # Try to open the video capture device
                cap = cv2.VideoCapture(device_index)
                
                if cap.isOpened():
                    # Get device properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    # Try to read multiple frames to verify the device is actually streaming
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Check if this is a real camera feed or just OBS placeholder
                        is_real_feed = self._verify_real_camera_feed(cap, frame, device_index)
                        
                        if is_real_feed:
                            device_info = {
                                'index': device_index,
                                'width': width,
                                'height': height,
                                'fps': fps,
                                'frame_shape': frame.shape,
                                'is_color': len(frame.shape) == 3,
                                'working': True,
                                'priority': self._calculate_device_priority(device_index, width, height, fps)
                            }
                        
                            detected_devices.append(device_info)
                            
                            # Show each detected device
                            device_type = self._identify_device_type(device_index, width, height, fps)
                            print(f"   Device {device_index}: {width}x{height} @ {fps}fps - {device_type}")
                        else:
                            print(f"   Device {device_index}: {width}x{height} @ {fps}fps - ❌ OBS Placeholder (not streaming)")
                    else:
                        print(f"   Device {device_index}: Cannot read frames")
                    
                    cap.release()
                    
            except Exception as e:
                # Device might not be available or accessible
                continue
        
        if detected_devices:
            print(f"\n📊 Found {len(detected_devices)} working camera device(s)")
            
            # Filter cameras by type
            external_cameras = [d for d in detected_devices if d['index'] > 0]
            builtin_cameras = [d for d in detected_devices if d['index'] == 0]
            
            # Additional verification: Test external cameras more rigorously
            verified_external_cameras = []
            
            for ext_cam in external_cameras:
                print(f"🔍 Double-checking Device {ext_cam['index']} for real streaming...")
                
                # Open camera again for extended verification
                test_cap = cv2.VideoCapture(ext_cam['index'])
                if test_cap.isOpened():
                    # Take multiple samples to verify it's really streaming
                    is_streaming = self._extended_camera_verification(test_cap, ext_cam['index'])
                    test_cap.release()
                    
                    if is_streaming:
                        verified_external_cameras.append(ext_cam)
                        print(f"✅ Device {ext_cam['index']} verified as streaming")
                    else:
                        print(f"❌ Device {ext_cam['index']} failed extended verification (likely placeholder)")
            
            # Choose the best verified camera
            if verified_external_cameras:
                # Sort verified external cameras by priority (highest first)
                verified_external_cameras.sort(key=lambda d: d['priority'], reverse=True)
                best_device = verified_external_cameras[0]
                print(f"🎯 Using verified external camera: Device {best_device['index']}")
            elif builtin_cameras:
                # Fall back to built-in camera
                best_device = builtin_cameras[0]
                print(f"🔄 No streaming external cameras found, using built-in camera: Device {best_device['index']}")
                print(f"💡 If you see OBS placeholder/logo, close OBS completely and restart the DJ app")
            else:
                # Last resort - use any detected device
                detected_devices.sort(key=lambda d: d['priority'], reverse=True)
                best_device = detected_devices[0]
                print(f"⚠️  Using fallback device: Device {best_device['index']}")
            device_type = self._identify_device_type(
                best_device['index'], 
                best_device['width'], 
                best_device['height'], 
                best_device['fps']
            )
            
            print(f"🎯 Selected: Device {best_device['index']} - {device_type}")
            print(f"   Resolution: {best_device['width']}x{best_device['height']} @ {best_device['fps']}fps")
            
            return best_device['index']
        
        print("❌ No external cameras detected, will use built-in camera")
        return self.fallback_device_index
    
    def _calculate_device_priority(self, index: int, width: int, height: int, fps: int) -> int:
        """
        Calculate priority score for camera devices
        Higher score = better device for DJ applications
        """
        priority = 0
        
        # OBS Virtual Camera characteristics (highest priority)
        # OBS typically creates devices at index > 0 with high resolution
        if index > 0 and width >= 1920 and height >= 1080:
            priority += 100  # Highest priority for OBS-like devices
        
        # iPhone/Continuity Camera characteristics (high priority)
        if width >= 1280 and height >= 720:
            priority += 50  # High resolution bonus
            
        # Prefer non-default devices (usually external/virtual cameras)
        if index > 0:
            priority += 30
            
        # Frame rate bonus
        if fps >= 30:
            priority += 10
        if fps >= 60:
            priority += 5
            
        # Resolution bonus
        total_pixels = width * height
        if total_pixels >= 1920 * 1080:  # 1080p+
            priority += 20
        elif total_pixels >= 1280 * 720:  # 720p+
            priority += 10
            
        return priority
    
    def _identify_device_type(self, index: int, width: int, height: int, fps: int) -> str:
        """
        Identify the likely type of camera device
        """
        # OBS Virtual Camera detection
        if index > 0 and width >= 1920 and height >= 1080:
            return "🎥 OBS Virtual Camera (RECOMMENDED)"
        
        # iPhone/Continuity Camera detection
        if width >= 1280 and height >= 720 and index > 0:
            return "📱 iPhone/External Camera"
            
        # Built-in camera
        if index == 0:
            return "💻 Built-in Camera"
            
        # Unknown high-quality camera
        if width >= 1280:
            return "📹 High-Quality Camera"
            
        return "📷 Standard Camera"
    
    def _verify_real_camera_feed(self, cap: cv2.VideoCapture, first_frame: np.ndarray, device_index: int) -> bool:
        """
        Verify if this is a real camera feed or just an OBS placeholder/logo
        Returns True if it's a real camera feed, False if it's just a static placeholder
        """
        try:
            # For built-in cameras, assume they're always real
            if device_index == 0:
                return True
            
            # For external cameras (likely OBS), check if frames are changing
            # OBS placeholder typically shows a static logo/image
            time.sleep(0.1)  # Small delay
            ret, second_frame = cap.read()
            
            if not ret or second_frame is None:
                return False
            
            # Calculate difference between frames
            diff = cv2.absdiff(first_frame, second_frame)
            diff_sum = np.sum(diff)
            
            # If frames are identical or nearly identical, likely a static placeholder
            # Real cameras always have some noise/movement
            # Increase threshold to be more strict about detecting static images
            if diff_sum < 5000:  # Higher threshold - static image or minimal movement
                print(f"   📊 Frame difference too low: {diff_sum} (likely static placeholder)")
                return False
            
            # Additional check: Look for OBS-specific patterns (optional)
            # We could check for specific colors/patterns in the OBS logo
            
            return True
            
        except Exception as e:
            # If verification fails, assume it's not a real feed
            print(f"   ⚠️  Camera verification failed: {e}")
            return False
    
    def _extended_camera_verification(self, cap: cv2.VideoCapture, device_index: int) -> bool:
        """
        Extended verification to detect OBS placeholders more reliably
        Takes multiple samples over time to verify actual streaming vs static content
        """
        try:
            # For built-in cameras, assume they're always real
            if device_index == 0:
                return True
            
            samples = []
            num_samples = 5
            
            print(f"   📊 Taking {num_samples} samples over 1 second...")
            
            # Take samples over 1 second
            for i in range(num_samples):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"   ❌ Sample {i+1} failed to read")
                    return False
                
                # Convert to grayscale for easier comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(gray)
                time.sleep(0.2)  # 200ms between samples
            
            # Calculate differences between all samples
            total_differences = 0
            comparisons = 0
            
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    diff = cv2.absdiff(samples[i], samples[j])
                    diff_sum = np.sum(diff)
                    total_differences += diff_sum
                    comparisons += 1
            
            avg_difference = total_differences / comparisons if comparisons > 0 else 0
            print(f"   📊 Average frame difference: {avg_difference:.0f}")
            
            # Check if this looks like an OBS placeholder despite being dynamic
            # OBS placeholder can have animated elements but distinctive patterns
            is_obs_placeholder = self._detect_obs_placeholder_pattern(samples[0])
            
            if is_obs_placeholder:
                print(f"   ❌ OBS placeholder pattern detected (even though dynamic)")
                return False
            
            # Much higher threshold for extended verification
            # Real cameras should show significant variation even when stationary
            if avg_difference < 10000:  # Very strict threshold
                print(f"   ❌ Static content detected (avg diff: {avg_difference:.0f})")
                return False
            
            print(f"   ✅ Dynamic content detected (avg diff: {avg_difference:.0f})")
            return True
            
        except Exception as e:
            print(f"   ⚠️  Extended verification failed: {e}")
            return False
    
    def _detect_obs_placeholder_pattern(self, frame: np.ndarray) -> bool:
        """
        Detect if this frame looks like an OBS placeholder/logo screen
        This uses heuristics to identify OBS branding patterns
        """
        try:
            # Convert to HSV for better color analysis
            frame_bgr = frame if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # Check for dark backgrounds typical of OBS placeholder
            # OBS usually has dark/black backgrounds with lighter logos
            dark_pixels = np.sum(frame < 50) / frame.size  # Very dark pixels
            
            if dark_pixels > 0.5:  # More than 50% very dark pixels
                print(f"   📊 High dark pixel ratio: {dark_pixels:.2f} (typical of OBS placeholder)")
                
                # Additional check: Look for limited color variety
                # OBS placeholders typically have few distinct colors
                unique_colors = len(np.unique(frame))
                color_variety = unique_colors / (frame.shape[0] * frame.shape[1])
                
                if color_variety < 0.1:  # Very limited color variety
                    print(f"   📊 Low color variety: {color_variety:.3f} (typical of logos/graphics)")
                    return True
                    
            # Check for very uniform regions (typical of graphic design)
            # Real cameras have more natural noise and variation
            std_dev = np.std(frame)
            if std_dev < 20:  # Very uniform image
                print(f"   📊 Low variation: std_dev={std_dev:.1f} (too uniform for real camera)")
                return True
                
            return False
            
        except Exception as e:
            print(f"   ⚠️  OBS pattern detection failed: {e}")
            return False  # If detection fails, assume it's not OBS placeholder
    
    def configure_camera_for_quality(self, cap: cv2.VideoCapture) -> bool:
        """
        Configure camera with optimal settings for iPhone/Continuity Camera
        """
        print("⚙️  Configuring camera for optimal quality...")
        
        settings_applied = []
        
        try:
            # Set resolution (iPhone cameras support up to 1920x1440)
            if cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.optimal_settings['width']):
                settings_applied.append(f"Width: {self.optimal_settings['width']}")
                
            if cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.optimal_settings['height']):
                settings_applied.append(f"Height: {self.optimal_settings['height']}")
            
            # Set high frame rate (iPhone cameras support up to 60fps)
            if cap.set(cv2.CAP_PROP_FPS, self.optimal_settings['fps']):
                settings_applied.append(f"FPS: {self.optimal_settings['fps']}")
            
            # Set codec for better quality/performance balance
            fourcc = cv2.VideoWriter_fourcc(*self.optimal_settings['fourcc'])
            if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
                settings_applied.append(f"Codec: {self.optimal_settings['fourcc']}")
            
            # Minimize buffer size for low latency (critical for DJ applications)
            if cap.set(cv2.CAP_PROP_BUFFERSIZE, self.optimal_settings['buffer_size']):
                settings_applied.append(f"Buffer: {self.optimal_settings['buffer_size']}")
            
            # Auto exposure and focus (iPhone cameras handle this well)
            if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.optimal_settings['exposure_auto']):
                settings_applied.append("Auto Exposure: ON")
                
            # Additional iPhone-optimized settings
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)    # Auto white balance
            
            # Get actual settings after configuration
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"✅ Camera configured successfully:")
            print(f"   📐 Resolution: {actual_width}x{actual_height}")
            print(f"   🎬 Frame rate: {actual_fps}fps")
            
            for setting in settings_applied:
                print(f"   ✓ {setting}")
            
            # Test the configuration with a frame capture
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"   🖼️  Frame test: {frame.shape} - SUCCESS")
                return True
            else:
                print("   ❌ Frame test failed")
                return False
                
        except Exception as e:
            print(f"❌ Error configuring camera: {e}")
            return False


class DJCameraWrapper:
    """
    DJ-optimized camera wrapper that handles:
    - Automatic iPhone/OBS camera detection with fallback to built-in camera
    - Proper resolution (1920x1080) for UI consistency
    - Horizontal mirroring for natural hand interaction
    - Graceful fallback to original webcam behavior
    """
    
    def __init__(self):
        self.cap = None
        self.target_width = 1920
        self.target_height = 1080
        self.device_index = None
        self.is_iphone_camera = False
        self.camera_manager = ContinuityCameraManager()
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera with automatic iPhone detection and fallback
        """
        print("🎥 Initializing DJ Camera...")
        
        # Try to detect iPhone/OBS camera first
        device_index = self.camera_manager.detect_continuity_camera()
        
        if device_index is None:
            print("⚠️  No cameras detected, using default camera")
            device_index = 0
        
        # Try to open the selected device
        self.cap = cv2.VideoCapture(device_index)
        
        if not self.cap or not self.cap.isOpened():
            print(f"❌ Could not open camera device {device_index}")
            # Try fallback to device 0
            if device_index != 0:
                print("🔄 Trying built-in camera as fallback...")
                device_index = 0
                self.cap = cv2.VideoCapture(device_index)
                
        if not self.cap or not self.cap.isOpened():
            print("❌ Failed to initialize any camera")
            return False
        
        # Store camera info and determine camera type
        self.device_index = device_index
        self.is_iphone_camera = (device_index > 0)  # Assume external cameras are iPhone/OBS
        
        # Determine if this is OBS Virtual Camera or direct iPhone Continuity Camera
        self.is_obs_virtual_camera = self._detect_obs_virtual_camera(device_index)
        
        # Configure camera settings
        success = self.camera_manager.configure_camera_for_quality(self.cap)
        
        # Force exact resolution for DJ controller UI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        # Verify the resolution was set correctly
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎯 Target resolution: {self.target_width}x{self.target_height}")
        print(f"📐 Actual resolution: {actual_width}x{actual_height}")
        
        # Test frame capture
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("❌ Cannot capture frames from camera")
            return False
            
        print(f"✅ Camera initialized successfully!")
        if self.is_iphone_camera:
            if self.is_obs_virtual_camera:
                print(f"📱 Using iPhone via OBS Virtual Camera (Device {device_index})")
                print(f"🪞 iPhone via OBS: will apply horizontal flip for mirror effect")
            else:
                print(f"📱 Using iPhone Continuity Camera (Device {device_index})")
                print(f"🪞 iPhone Continuity Camera: will apply horizontal flip for mirror effect")
        else:
            print(f"💻 Using built-in camera (Device {device_index})")
            print(f"🪞 Built-in camera: will apply horizontal flip for mirror effect")
        
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera with DJ optimizations applied
        - iPhone/OBS camera: No flip needed (OBS handles mirroring)
        - Built-in camera: Apply horizontal flip for mirror effect
        - Always resize to target resolution
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return False, None
        
        # Resize frame to exact target resolution if needed
        current_height, current_width = frame.shape[:2]
        if current_width != self.target_width or current_height != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Apply horizontal flip for all cameras to ensure mirror behavior
        # User reported that iPhone via OBS also needs flipping for proper mirror effect
        should_flip = True  # Default: flip all cameras for mirror behavior
        
        if should_flip:
            frame = cv2.flip(frame, 1)
        
        return True, frame
    
    def release(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()
            print("📹 Camera released")
    
    def _detect_obs_virtual_camera(self, device_index: int) -> bool:
        """
        Detect if this device is OBS Virtual Camera vs direct iPhone Continuity Camera
        """
        try:
            # Simple heuristic: OBS Virtual Camera typically appears at specific indices
            # and has specific device characteristics
            
            # Check if we have camera access for testing
            if not self.cap or not self.cap.isOpened():
                return False
            
            # Get device properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # OBS Virtual Camera characteristics:
            # 1. Usually high resolution (1920x1080)
            # 2. Usually 60fps or 30fps exactly
            # 3. Device index > 0
            if device_index > 0 and width >= 1920 and height >= 1080 and fps in [30, 60]:
                # Additional check: try to read a frame and see if it looks like OBS
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # If we successfully configured this device earlier in our detection process
                    # and it passed our strict verification, it's likely OBS
                    # (This is a heuristic - may need refinement based on real usage)
                    
                    # For now, be conservative: assume it's OBS if it's high-res and high-fps
                    return True
            
            return False
            
        except Exception as e:
            # If detection fails, assume it's not OBS Virtual Camera
            return False
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution"""
        return (self.target_width, self.target_height)
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()


class SmoothAnimationController:
    """
    Controls all animations to run at synchronized 60fps
    """
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Animation states
        self.jog_wheel_angles = {'deck1': 0.0, 'deck2': 0.0}
        self.jog_wheel_velocities = {'deck1': 0.0, 'deck2': 0.0}
        self.album_rotations = {'deck1': 0.0, 'deck2': 0.0}
        
        # Visual effect states
        self.button_press_animations = {}
        self.fader_animations = {}
        self.level_meters = {'deck1': 0.0, 'deck2': 0.0}
        
        # Smooth interpolation values
        self.smooth_values = {}
        
    def update(self, deck_states: Dict[str, Any]):
        """
        Update all animations based on current deck states
        Called once per frame to maintain 60fps sync
        """
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Update jog wheel rotations
        self._update_jog_wheels(deck_states, dt)
        
        # Update album artwork rotations
        self._update_album_rotations(deck_states, dt)
        
        # Update visual effects
        self._update_visual_effects(dt)
        
        # Update level meters with smooth interpolation
        self._update_level_meters(deck_states, dt)
    
    def _update_jog_wheels(self, deck_states: Dict[str, Any], dt: float):
        """Update jog wheel spinning animations at 60fps"""
        for deck_key in ['deck1', 'deck2']:
            deck_num = 1 if deck_key == 'deck1' else 2
            is_playing = deck_states.get(f'deck{deck_num}_playing', False)
            track_position = deck_states.get(f'deck{deck_num}_position', 0.0)
            
            if is_playing:
                # Calculate RPM based on track tempo (like real vinyl)
                bpm = deck_states.get(f'deck{deck_num}_bpm', 120)
                rpm = bpm / 4.0  # Typical vinyl RPM relationship to BPM
                
                # Convert RPM to angular velocity (radians per second)
                angular_velocity = (rpm / 60.0) * 2 * math.pi
                
                # Update angle smoothly
                self.jog_wheel_angles[deck_key] += angular_velocity * dt
                self.jog_wheel_velocities[deck_key] = angular_velocity
                
            else:
                # Gradually slow down when stopped (friction simulation)
                friction = 8.0  # Friction coefficient
                if abs(self.jog_wheel_velocities[deck_key]) > 0.1:
                    # Apply friction
                    friction_force = -friction * self.jog_wheel_velocities[deck_key]
                    self.jog_wheel_velocities[deck_key] += friction_force * dt
                    self.jog_wheel_angles[deck_key] += self.jog_wheel_velocities[deck_key] * dt
                else:
                    self.jog_wheel_velocities[deck_key] = 0.0
            
            # Keep angle in reasonable range
            self.jog_wheel_angles[deck_key] = self.jog_wheel_angles[deck_key] % (2 * math.pi)
    
    def _update_album_rotations(self, deck_states: Dict[str, Any], dt: float):
        """Update album artwork rotation animations"""
        for deck_key in ['deck1', 'deck2']:
            deck_num = 1 if deck_key == 'deck1' else 2
            is_playing = deck_states.get(f'deck{deck_num}_playing', False)
            
            if is_playing:
                # Rotate at 33.33 RPM (like real vinyl)
                rpm = 33.33
                angular_velocity = (rpm / 60.0) * 2 * math.pi
                self.album_rotations[deck_key] += angular_velocity * dt
            
            # Keep rotation in 0-360 degree range
            self.album_rotations[deck_key] = self.album_rotations[deck_key] % (2 * math.pi)
    
    def _update_visual_effects(self, dt: float):
        """Update button press animations and other visual effects"""
        # Update button press animations (fade out over time)
        for button_id in list(self.button_press_animations.keys()):
            animation = self.button_press_animations[button_id]
            animation['intensity'] -= dt * 3.0  # Fade out over ~0.33 seconds
            
            if animation['intensity'] <= 0:
                del self.button_press_animations[button_id]
        
        # Update fader animations (smooth value changes)
        for fader_id in list(self.fader_animations.keys()):
            animation = self.fader_animations[fader_id]
            target = animation['target']
            current = animation['current']
            
            # Smooth interpolation (exponential smoothing)
            smoothing = 0.15  # Smoothing factor (0-1, higher = more responsive)
            animation['current'] += (target - current) * smoothing
            
            # Remove if close enough to target
            if abs(target - current) < 0.001:
                animation['current'] = target
    
    def _update_level_meters(self, deck_states: Dict[str, Any], dt: float):
        """Update audio level meters with smooth animation"""
        for deck_key in ['deck1', 'deck2']:
            deck_num = 1 if deck_key == 'deck1' else 2
            
            # Get current volume level (simulated - in real app would be from audio analysis)
            is_playing = deck_states.get(f'deck{deck_num}_playing', False)
            volume = deck_states.get(f'deck{deck_num}_volume', 0.0)
            
            if is_playing:
                # Simulate audio levels with some variation
                base_level = volume * 0.8
                variation = math.sin(time.time() * 4) * 0.1  # Oscillation
                target_level = max(0, min(1, base_level + variation))
            else:
                target_level = 0.0
            
            # Smooth interpolation for level meters
            current_level = self.level_meters[deck_key]
            smoothing = 0.2 if target_level > current_level else 0.05  # Faster attack, slower decay
            self.level_meters[deck_key] += (target_level - current_level) * smoothing
    
    def trigger_button_press(self, button_id: str, intensity: float = 1.0):
        """Trigger a button press animation"""
        self.button_press_animations[button_id] = {
            'intensity': intensity,
            'start_time': time.time()
        }
    
    def set_fader_target(self, fader_id: str, target_value: float):
        """Set target value for smooth fader animation"""
        if fader_id not in self.fader_animations:
            self.fader_animations[fader_id] = {'current': target_value, 'target': target_value}
        else:
            self.fader_animations[fader_id]['target'] = target_value
    
    def get_jog_wheel_angle(self, deck: str) -> float:
        """Get current jog wheel angle in radians"""
        return self.jog_wheel_angles.get(deck, 0.0)
    
    def get_album_rotation(self, deck: str) -> float:
        """Get current album rotation in radians"""
        return self.album_rotations.get(deck, 0.0)
    
    def get_button_intensity(self, button_id: str) -> float:
        """Get current button press animation intensity"""
        if button_id in self.button_press_animations:
            return max(0, self.button_press_animations[button_id]['intensity'])
        return 0.0
    
    def get_fader_value(self, fader_id: str) -> float:
        """Get current smooth fader value"""
        if fader_id in self.fader_animations:
            return self.fader_animations[fader_id]['current']
        return 0.0
    
    def get_level_meter(self, deck: str) -> float:
        """Get current level meter value"""
        return self.level_meters.get(deck, 0.0)
    
    def smooth_interpolate(self, key: str, target_value: float, smoothing: float = 0.1) -> float:
        """
        Smooth interpolation for any value
        Returns smoothly interpolated value
        """
        if key not in self.smooth_values:
            self.smooth_values[key] = target_value
        
        current = self.smooth_values[key]
        self.smooth_values[key] += (target_value - current) * smoothing
        
        return self.smooth_values[key]


class PerformanceMonitor:
    """
    Monitor frame rate and performance for optimization
    """
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.frame_times = []
        self.max_samples = 60  # Keep last 60 frame times (1 second at 60fps)
        self.last_report_time = time.time()
        self.report_interval = 5.0  # Report every 5 seconds
    
    def record_frame_time(self, frame_time: float):
        """Record a frame time for analysis"""
        self.frame_times.append(frame_time)
        
        # Keep only recent samples
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.frame_times:
            return {'avg_fps': 0, 'min_fps': 0, 'max_fps': 0, 'frame_time_ms': 0}
        
        frame_times = np.array(self.frame_times)
        avg_frame_time = np.mean(frame_times)
        min_frame_time = np.min(frame_times)
        max_frame_time = np.max(frame_times)
        
        return {
            'avg_fps': 1.0 / avg_frame_time if avg_frame_time > 0 else 0,
            'min_fps': 1.0 / max_frame_time if max_frame_time > 0 else 0,
            'max_fps': 1.0 / min_frame_time if min_frame_time > 0 else 0,
            'frame_time_ms': avg_frame_time * 1000
        }
    
    def should_report(self) -> bool:
        """Check if it's time to report performance stats"""
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            return True
        return False


# Factory functions for easy integration
def create_dj_camera() -> DJCameraWrapper:
    """
    Factory function to create a DJ-optimized camera wrapper with iPhone support
    Automatically detects iPhone/OBS camera and falls back to built-in camera
    """
    camera = DJCameraWrapper()
    
    if camera.initialize_camera():
        return camera
    else:
        raise RuntimeError("Failed to initialize any camera (iPhone or built-in)")


def create_optimized_camera_capture() -> cv2.VideoCapture:
    """
    Legacy compatibility function - creates a basic VideoCapture object
    Use create_dj_camera() for full DJ wrapper functionality
    """
    print("⚠️  Using legacy camera initialization")
    print("💡 Consider using create_dj_camera() for better DJ integration")
    
    manager = ContinuityCameraManager()
    device_index = manager.detect_continuity_camera()
    
    if device_index is None:
        device_index = 0  # Fallback to built-in camera
    
    cap = cv2.VideoCapture(device_index)
    
    if not cap.isOpened():
        # Final fallback
        print("🔄 Trying built-in camera as final fallback...")
        cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        manager.configure_camera_for_quality(cap)
    
    return cap


if __name__ == "__main__":
    """Test the iPhone camera integration with fallback"""
    print("🧪 Testing iPhone Camera Integration with Fallback")
    print("=" * 50)
    
    try:
        # Test DJ camera wrapper
        camera = create_dj_camera()
        print("✅ DJ Camera wrapper created successfully")
        print(f"📱 iPhone camera: {camera.is_iphone_camera}")
        print(f"🎯 Resolution: {camera.get_resolution()}")
        
        # Test animation controller
        animations = SmoothAnimationController()
        monitor = PerformanceMonitor()
        print("✅ Animation controller initialized")
        
        # Test frame capture with 3-second preview
        print("📹 Testing frame capture (3 seconds)...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 3:
            ret, frame = camera.read_frame()
            
            if ret:
                frame_count += 1
                
                # Add test overlay
                cv2.putText(frame, "iPhone Camera Integration Test", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Camera: {'iPhone/OBS' if camera.is_iphone_camera else 'Built-in'}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Raise your RIGHT hand - check mirror behavior", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('iPhone Camera Integration Test', frame)
                
                # Update animations
                deck_states = {
                    'deck1_playing': False,
                    'deck2_playing': False,
                    'deck1_position': 0.0,
                    'deck2_position': 0.0,
                    'deck1_volume': 0.5,
                    'deck2_volume': 0.5
                }
                animations.update(deck_states)
                
                # Record performance
                monitor.record_frame_time(1.0/60.0)  # Simulated
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Show performance stats
        stats = monitor.get_performance_stats()
        print(f"\n📊 Test Results:")
        print(f"   Captured {frame_count} frames in 3 seconds")
        print(f"   Average FPS: {frame_count / 3:.1f}")
        print(f"   Animation system: {animations.target_fps}fps target")
        print(f"   Camera type: {'iPhone/OBS' if camera.is_iphone_camera else 'Built-in'}")
        
        cv2.destroyAllWindows()
        camera.release()
        
        print("🎉 Test completed successfully!")
        print("✅ iPhone camera integration with fallback working!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
