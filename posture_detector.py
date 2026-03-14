import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import calculate_angle, calculate_distance, calculate_ear

class PostureDetector:
    def __init__(self, model_asset_path='pose_landmarker_heavy.task', min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize the new Tasks API for MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, 
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
            )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

        # Initialize the Face API
        base_options_face = python.BaseOptions(model_asset_path='face_landmarker.task')
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        # Initialize the Hands API
        base_options_hands = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options_hands = vision.HandLandmarkerOptions(
            base_options=base_options_hands,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hands)
        
        self.mp_drawing = python.vision.drawing_utils    
        self.mp_drawing_styles = python.vision.drawing_styles
        
        # Results cache
        self.results_pose = None
        self.results_face = None
        self.results_hands = None

    def find_pose(self, img, draw=True):
        """
        Processes the image and extracts pose, face, and hand landmarks.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect landmarks
        self.results_pose = self.pose_landmarker.detect(mp_image)
        self.results_face = self.face_landmarker.detect(mp_image)
        self.results_hands = self.hand_landmarker.detect(mp_image)
        
        # Draw the resulting landmarks
        if draw:
            h, w, c = img.shape
            
            # Draw Posture Body Parts
            if self.results_pose and self.results_pose.pose_landmarks:
                for pose_landmarks in self.results_pose.pose_landmarks:
                    for lm in pose_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 3, (245, 117, 66), cv2.FILLED)
                        
            # Draw Face Mesh Outline
            if self.results_face and self.results_face.face_landmarks:
                for face_landmarks in self.results_face.face_landmarks:
                    for lm in face_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # Make face dots tiny
                        cv2.circle(img, (cx, cy), 1, (255, 255, 255), cv2.FILLED)
                        
            # Draw Hands
            if self.results_hands and self.results_hands.hand_landmarks:
                for hand_landmarks in self.results_hands.hand_landmarks:
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 2, (121, 22, 76), cv2.FILLED)
                        
        return img

    def get_landmarks(self, img):
        """
        Extracts landmark coordinates for pose, face, and hands.
        Returns a dictionary containing each list.
        """
        landmarks = {
            'pose': [],
            'face': [],
            'hands': []
        }
        
        h, w, c = img.shape
        
        # Pose
        if self.results_pose and self.results_pose.pose_landmarks:
            main_pose_landmarks = self.results_pose.pose_landmarks[0]
            for id, lm in enumerate(main_pose_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks['pose'].append({'id': id, 'x': cx, 'y': cy, 'z': lm.z, 'visibility': lm.visibility})
                
        # Face
        if self.results_face and self.results_face.face_landmarks:
            main_face_landmarks = self.results_face.face_landmarks[0]
            for id, lm in enumerate(main_face_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks['face'].append({'id': id, 'x': cx, 'y': cy})
                
        # Hands (Can be multiple, keeping them in flat array for leaning logic)
        if self.results_hands and self.results_hands.hand_landmarks:
            for hand_landmarks in self.results_hands.hand_landmarks:
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks['hands'].append({'id': id, 'x': cx, 'y': cy})
                    
        return landmarks

    def check_posture(self, landmarks):
        """
        Analyzes landmarks to determine if the user is slouching, sleeping, or leaning on their hands.
        Returns a tuple: (status_text, color)
        """
        pose_lms = landmarks.get('pose', [])
        face_lms = landmarks.get('face', [])
        hands_lms = landmarks.get('hands', [])

        if not pose_lms:
            return "No body detected", (255, 255, 255)

        try:
            # 1. Check for Drowsiness (EAR)
            # MediaPipe FaceMesh eye indices:
            # Left Eye:  33, 160, 158, 133, 153, 144
            # Right Eye: 362, 385, 387, 263, 373, 380
            if face_lms and len(face_lms) > 387:
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                
                left_eye_pts = [(face_lms[i]['x'], face_lms[i]['y']) for i in left_eye_indices]
                right_eye_pts = [(face_lms[i]['x'], face_lms[i]['y']) for i in right_eye_indices]
                
                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Typical EAR threshold for closed eyes is < 0.21
                if avg_ear < 0.21:
                    return f"Eyes Closed! (EAR: {avg_ear:.2f})", (0, 0, 255)

            # 2. Check for Hand Support (Leaning on hands)
            if face_lms and hands_lms:
                # Get chin coordinate
                chin_index = 152
                if len(face_lms) > chin_index:
                    chin_pt = [face_lms[chin_index]['x'], face_lms[chin_index]['y']]
                    
                    # See if any hand landmarks are too close to the chin (threshold distance)
                    # Hand landmarks are often spread, so checking any point vs chin.
                    for hand_pt_dict in hands_lms:
                        h_pt = [hand_pt_dict['x'], hand_pt_dict['y']]
                        distance = calculate_distance(chin_pt, h_pt)
                        
                        # Distance threshold (usually 4% - 6% of the screen depending on resolution)
                        # We are using absolute pixel distance, so 40-50 pixels is a good start.
                        if distance < 50:
                            return "Hand Support Detected!", (0, 0, 255)

            # 3. Check for Posture (Slouching)
            RIGHT_EAR = 8
            RIGHT_SHOULDER = 12
            RIGHT_HIP = 24
            
            if len(pose_lms) > RIGHT_HIP:
                r_ear = [pose_lms[RIGHT_EAR]['x'], pose_lms[RIGHT_EAR]['y']]
                r_shoulder = [pose_lms[RIGHT_SHOULDER]['x'], pose_lms[RIGHT_SHOULDER]['y']]
                r_hip = [pose_lms[RIGHT_HIP]['x'], pose_lms[RIGHT_HIP]['y']]

                r_shoulder_vertical = [r_shoulder[0], r_shoulder[1] - 100] 
                neck_inclination = calculate_angle(r_ear, r_shoulder, r_shoulder_vertical)

                r_hip_vertical = [r_hip[0], r_hip[1] - 100]
                torso_inclination = calculate_angle(r_shoulder, r_hip, r_hip_vertical)

                if neck_inclination > 35 or torso_inclination > 15:
                    return f"Slouching! (Neck: {int(neck_inclination)} deg)", (0, 0, 255) 
                else:
                    return f"Good Posture (Neck: {int(neck_inclination)} deg)", (0, 255, 0) 
            
            return "Cannot see full upper body", (255, 255, 255)

        except IndexError:
            return "Landmarks incomplete", (255, 255, 255)
