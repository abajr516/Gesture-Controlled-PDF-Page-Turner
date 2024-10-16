# Angel Bajracharya
# Topics in Computer Science Project
# 10/15/24

import cv2
import mediapipe as mp
import fitz
import numpy as np
import time
import math

# Initialize MediaPipe and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Identifying hand gestures

def is_thumbs_up(hand_landmarks):
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    # Thumb is extended upwards
    is_thumb_extended = thumb_tip_y < thumb_ip_y
    # Other fingers are folded
    is_index_folded = index_tip_y > thumb_tip_y
    is_middle_folded = middle_tip_y > thumb_tip_y
    is_ring_folded = ring_tip_y > thumb_tip_y
    is_pinky_folded = pinky_tip_y > thumb_tip_y

    return is_thumb_extended and is_index_folded and is_middle_folded and is_ring_folded and is_pinky_folded

def is_thumbs_down(hand_landmarks):
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

    # Thumb is extended downwards
    is_thumb_extended_down = thumb_tip_y > thumb_ip_y
    # Other fingers can be folded or not fully extended
    are_other_fingers_folded = index_tip_y < thumb_tip_y and middle_tip_y < thumb_tip_y

    return is_thumb_extended_down and are_other_fingers_folded

'''
# Saving these methods for adding future gestures plus functionality
def is_peace_sign(hand_landmarks):
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    # Index and middle fingers are extended
    is_index_extended = index_tip_y < thumb_ip_y
    is_middle_extended = middle_tip_y < thumb_ip_y

    # Ring and pinky fingers are folded
    is_ring_folded = ring_tip_y > thumb_ip_y
    is_pinky_folded = pinky_tip_y > thumb_ip_y

    # Optional: Ignore thumb state or allow flexibility
    # For simplicity, we're not considering the thumb's state here

    return is_index_extended and is_middle_extended and is_ring_folded and is_pinky_folded

def is_frame_with_hands(landmarks1, landmarks2):
    """
    Checks if the gesture resembles a "frame with hands" using the thumb and index fingers of both hands.

    Parameters:
    - landmarks1: Landmarks for the first hand.
    - landmarks2: Landmarks for the second hand.

    Returns:
    - True if the gesture forms a frame, False otherwise.
    """
    # Get relevant landmarks
    index_tip_1 = landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_1 = landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip_2 = landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_2 = landmarks2.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Calculate distances between corresponding fingers of both hands
    horizontal_distance_1 = math.sqrt((index_tip_1.x - thumb_tip_1.x)**2 + (index_tip_1.y - thumb_tip_1.y)**2)
    horizontal_distance_2 = math.sqrt((index_tip_2.x - thumb_tip_2.x)**2 + (index_tip_2.y - thumb_tip_2.y)**2)
    vertical_distance_1 = math.sqrt((index_tip_1.x - index_tip_2.x)**2 + (index_tip_1.y - index_tip_2.y)**2)
    vertical_distance_2 = math.sqrt((thumb_tip_1.x - thumb_tip_2.x)**2 + (thumb_tip_1.y - thumb_tip_2.y)**2)

    # Define thresholds for frame detection (normalized coordinates)
    horizontal_threshold = 0.1  # Adjust based on sensitivity
    vertical_threshold = 0.1    # Adjust based on sensitivity

    # Check if distances are similar enough to form a frame
    is_horizontal_similar = abs(horizontal_distance_1 - horizontal_distance_2) < horizontal_threshold
    is_vertical_similar = abs(vertical_distance_1 - vertical_distance_2) < vertical_threshold

    return is_horizontal_similar and is_vertical_similar
'''

# Open the PDF document
doc = fitz.open("THE-VERY-HUNGRY-CATERPILLAR.pdf")
# Start webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    current_page = 0
    
    # Initialize gesture info
    gesture = None  # Gesture before opening cap
    prev_gesture_time = 0
    gesture_cooldown = 1 # Delay pd before processing next gesture
    current_time = 0
    
    while cap.isOpened():
        gesture=None
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and detect hands
        results = hands.process(image_rgb)

        # Convert back to BGR for OpenCV
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            # Iterate through each detected hand
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                    
                # Detect and annotate gestures
                if is_thumbs_up(hand_landmarks):
                    cv2.putText(image_bgr, "Thumbs Up Detected", (10, 30 + i*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    gesture = "thumbs up"
                elif is_thumbs_down(hand_landmarks):
                    cv2.putText(image_bgr, "Thumbs Down Detected", (10, 60 + i*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    gesture = "thumbs down"
                else: 
                    gesture = None
        current_time = time.time()
        if ((gesture is not None) and ((current_time - prev_gesture_time) > gesture_cooldown)):
            if ((gesture == "thumbs up") and (current_page < doc.page_count - 1)):
                current_page += 1
                print(f"Page advanced to {(current_page+1)%(doc.page_count+1)}")
            elif ((gesture == "thumbs down") and (current_page > 0)):
                current_page -= 1
                print(f"Page reverted to {(current_page+1)%(doc.page_count+1)}")
                
            # Update gesture info
            prev_gesture_time = current_time
                
        # Display the annotated image
        cv2.imshow('MediaPipe Hands', image_bgr)

        # Extract the current page as an image
        page = doc.load_page(current_page%doc.page_count)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # Handle images with an alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow("Page", img)
        
        # Handle user input for navigation
        key = cv2.waitKey(1) & 0xFF  # Adjust to a small delay
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
