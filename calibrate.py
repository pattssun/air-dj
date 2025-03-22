import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import logging

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Force TensorFlow to use CPU only and disable AVX/AVX2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Check if camera is opened correctly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Calibration data
    pinch_min = 1.0
    pinch_max = 0.0
    distance_min = 1.0
    distance_max = 0.0
    
    # Instructions
    instructions = [
        "1. Show both hands in frame",
        "2. Pinch your fingers (minimum)",
        "3. Spread your fingers (maximum)",
        "4. Move hands close together (minimum)",
        "5. Move hands far apart (maximum)",
        "6. Press 's' to save calibration"
    ]
    
    current_step = 0
    calibration_complete = False
    
    print("Starting calibration...")
    print("Follow the on-screen instructions.")
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture video frame.")
                break
            
            # Flip the image horizontally for a mirrored view
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(rgb_image)
            
            # Clear the image with a semi-transparent overlay for instructions
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
            
            # Display instructions
            cv2.putText(image, instructions[current_step], (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw calibration status
            status_text = f"Pinch: [{pinch_min:.3f} - {pinch_max:.3f}] | Distance: [{distance_min:.3f} - {distance_max:.3f}]"
            cv2.putText(image, status_text, (20, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                left_hand_landmarks = None
                right_hand_landmarks = None
                
                # Identify left and right hands
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    
                    # Store landmarks for specific hand (note we flip due to mirror)
                    if handedness == 'Left':  
                        right_hand_landmarks = hand_landmarks
                    elif handedness == 'Right':
                        left_hand_landmarks = hand_landmarks
                    
                    # Draw the hand landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Draw circles at thumb and index tips with enhanced visibility
                    h, w, c = image.shape
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    
                    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                    
                    color = (0, 255, 0) if handedness == 'Left' else (0, 0, 255)
                    
                    # Draw circles and line for the pinch
                    cv2.circle(image, thumb_pos, 15, color, -1)
                    cv2.circle(image, index_pos, 15, color, -1)
                    cv2.line(image, thumb_pos, index_pos, color, 3)
                    
                    # Calculate pinch distance
                    thumb = np.array([thumb_tip.x, thumb_tip.y])
                    index = np.array([index_tip.x, index_tip.y])
                    pinch_dist = np.linalg.norm(thumb - index)
                    
                    # Update min and max pinch distances
                    pinch_min = min(pinch_min, pinch_dist)
                    pinch_max = max(pinch_max, pinch_dist)
                    
                    # Show pinch distance
                    cv2.putText(image, f"Pinch: {pinch_dist:.3f}", 
                               (thumb_pos[0] - 30, thumb_pos[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # If both hands are detected, measure distance between them
                if left_hand_landmarks and right_hand_landmarks:
                    left_wrist = left_hand_landmarks.landmark[0]
                    right_wrist = right_hand_landmarks.landmark[0]
                    
                    left_pos = (int(left_wrist.x * w), int(left_wrist.y * h))
                    right_pos = (int(right_wrist.x * w), int(right_wrist.y * h))
                    
                    # Draw line between wrists
                    cv2.line(image, left_pos, right_pos, (255, 255, 0), 3)
                    
                    # Calculate hand distance
                    left_point = np.array([left_wrist.x, left_wrist.y])
                    right_point = np.array([right_wrist.x, right_wrist.y])
                    hand_distance = np.linalg.norm(left_point - right_point)
                    
                    # Update min and max hand distances
                    distance_min = min(distance_min, hand_distance)
                    distance_max = max(distance_max, hand_distance)
                    
                    # Show hand distance
                    mid_x = (left_pos[0] + right_pos[0]) // 2
                    mid_y = (left_pos[1] + right_pos[1]) // 2
                    cv2.putText(image, f"Distance: {hand_distance:.3f}", 
                               (mid_x - 50, mid_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Display the image
            cv2.imshow('Hand DJ Calibration', image)
            
            # Process key presses
            key = cv2.waitKey(5) & 0xFF
            
            # Next calibration step
            if key == ord('n'):
                current_step = (current_step + 1) % len(instructions)
            
            # Save calibration
            elif key == ord('s'):
                save_calibration(pinch_min, pinch_max, distance_min, distance_max)
                calibration_complete = True
                print("Calibration saved successfully!")
                break
            
            # Quit
            elif key == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if not calibration_complete:
            print("Calibration was not completed. Default values will be used.")

def save_calibration(pinch_min, pinch_max, distance_min, distance_max):
    """Save calibration values to a JSON file"""
    calibration = {
        "pinch_min": pinch_min,
        "pinch_max": pinch_max,
        "distance_min": distance_min,
        "distance_max": distance_max
    }
    
    with open('calibration.json', 'w') as f:
        json.dump(calibration, f, indent=4)
    
    print(f"Calibration values saved to calibration.json:")
    print(f"Pinch range: {pinch_min:.3f} - {pinch_max:.3f}")
    print(f"Distance range: {distance_min:.3f} - {distance_max:.3f}")

if __name__ == "__main__":
    main()
    print("Calibration complete. You can now run hand_dj.py which will use these calibrated values.") 