#!/usr/bin/env python3

def test_imports():
    """Test if all required packages can be imported."""
    
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ Error importing OpenCV: {e}")
    
    try:
        import mediapipe
        print("✓ MediaPipe imported successfully")
    except ImportError as e:
        print(f"✗ Error importing MediaPipe: {e}")
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Error importing NumPy: {e}")
    
    try:
        import pyo
        print("✓ Pyo imported successfully")
    except ImportError as e:
        print(f"✗ Error importing Pyo: {e}")

if __name__ == "__main__":
    test_imports() 