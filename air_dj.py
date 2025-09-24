#!/usr/bin/env python3
"""
Air DJ Controller Launcher
Simple launcher script for the DJ controller with webcam interface
"""

import sys
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Air DJ Controller - Hand Tracking DJ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python air_dj.py           # Start with BPM sync enabled (default)
  python air_dj.py unsync    # Start with BPM sync disabled

Commands:
  unsync     Disable automatic BPM synchronization between decks
        """
    )
    
    parser.add_argument('command', nargs='?', choices=['unsync'], 
                       help='Optional command to modify behavior')
    
    args = parser.parse_args()
    
    # Determine BPM sync setting
    enable_bpm_sync = args.command != 'unsync'
    
    print("=" * 60)
    print("          🎧 AIR DJ CONTROLLER - Hand Tracking DJ 🎧")
    print("=" * 60)
    
    # Show sync status
    sync_status = "ENABLED" if enable_bpm_sync else "DISABLED"
    print(f"\n🔄 BPM SYNC: {sync_status}")
    if not enable_bpm_sync:
        print("   📊 Tracks will play at their original BPM")
        print("   🔄 Use 'python air_dj.py' to enable sync")
    
    print("\nFeatures:")
    print("• Transparent DJ controller overlay on camera feed")
    print("• Improved pinch detection with expanded hit areas")
    print("• Crystal clear stem isolation (ONLY vocals + instrumentals loaded)")
    print("• Professional track position management (like Rekordbox/Serato)")
    print("• Automatic stem track loading from songs folder")
    print("• Real-time audio stem control and mixing")
    print("• Independent volume control for each deck")
    print("• Mac default audio output support")
    print("\nProfessional DJ Controls:")
    print("• Use PINCH gestures (thumb + index finger close together) to interact")
    print("• CUE: Jumps to cue point (beginning) for preview")
    print("• PLAY/PAUSE: Continues from current position (maintains playback)")
    print("• VOCAL/INSTRUMENTAL: Real-time toggle at current position")
    print("• VOLUME FADERS: Drag up/down to control deck volume")
    print("• Position management just like real DJ software!")
    print("• Press 'q' to quit")
    print("\n" + "=" * 60)
    
    try:
        # Import and start the controller
        from dj_controller import DJController
        
        print("\nStarting DJ Controller...")
        print("Position yourself in front of the camera and use pinch gestures!")
        
        controller = DJController(enable_bpm_sync=enable_bpm_sync)
        controller.run()
        
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nPlease make sure you're running this in the virtual environment:")
        print("source venv_py311/bin/activate")
        print("python air_dj.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError starting DJ Controller: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
