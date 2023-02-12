import numpy as np
import cv2
import mediapipe as mp

def key_equal(key, key_char):
    return key & 0xFF == ord(key_char)

def overlay_image(img, img_overlay):
    x, y, _ = img.shape
    a, b, _ = img_overlay.shape
    
    out = img
    out[0:a, 0:b] = img_overlay

    return out

def landmark_vector_dir(lm1, lm2):
    vec = np.array([lm1.x - lm2.x, lm1.y - lm2.y, lm1.z - lm2.z]) 
    return vec / np.linalg.norm(vec)

def landmark_flatten(lm):
    lm = np.array([[t for t in [s.x, s.y, s.z]] for s in lm.landmark])
    return lm.flatten()

def get_hand_landmark(file_name):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    image = cv2.imread(file_name)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None
    
    lm = results.multi_hand_landmarks[0]
    return landmark_flatten(lm)

