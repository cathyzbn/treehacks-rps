import cv2
import mediapipe as mp
from player_pose import *
from gesture_classifier import *
from helpers import *


""" Result Display """
GES_SIZE = (100, 100)
rock_gesture = cv2.resize(cv2.imread("assets/rock.png"), GES_SIZE)
paper_gesture = cv2.resize(cv2.imread("assets/paper.png"), GES_SIZE)
scissors_gesture = cv2.resize(cv2.imread("assets/scissors.png"), GES_SIZE)

VISIBILITY_THRESHOLD = 0.8

def execute():
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    player_pose = PlayerPose()
    gesture_classifier = GestureClassifier()

    next_move = None
    def get_result_frame(frame, pose, hand, mp_pose, pose_classifier, gesture_classifier):
        nonlocal next_move
        landmarks = pose.pose_world_landmarks
        if not landmarks:
            return frame
        
        wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if min(wrist.visibility, elbow.visibility, shoulder.visibility) < VISIBILITY_THRESHOLD:
            next_move = None
            return frame
        
        player_pose.update_pose(shoulder, elbow, wrist)
        
        if not player_pose.is_move():
            print('exit')
            next_move = None
            return frame
        
        if not hand.multi_hand_landmarks:
            next_move = None
            return frame
        
        if next_move is None:
            next_move = gesture_classifier.classify_gesture(hand.multi_hand_landmarks[0])
            
        if next_move == 0:
            return overlay_image(frame, scissors_gesture)
        elif next_move == 1:
            return overlay_image(frame, rock_gesture)
        else:  
            return overlay_image(frame, paper_gesture)
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.1) as hand:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                p = pose.process(image)
                h = hand.process(cv2.flip(image, 1))

                show_img = get_result_frame(frame, p, h, mp_pose, player_pose, gesture_classifier)
                cv2.imshow("Let\'s play Rock, Paper, Scissors!", show_img)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()

if __name__ == "__main__":
    execute()