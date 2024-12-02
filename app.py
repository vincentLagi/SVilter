import os
import random
import cv2.data
from django import conf
from flask import Flask, Response, flash, jsonify, redirect, session, url_for, render_template, request
from flask_sqlalchemy import SQLAlchemy
import cv2
from matplotlib import use
import mediapipe as mp
import threading
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import pygame
from sklearn.metrics import confusion_matrix
from sqlalchemy import exists, false, null
import seaborn as sns
import tensorflow as tf
import dlib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'VL24-1'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/bpcv'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model User
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), nullable=False, unique=True)
    password = db.Column(db.String(50), nullable=False)

    def __init__(self, name, password):
        self.username = name
        self.password = password

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FPS, 30)
face_mesh_frame = None


# filter one
cat_ears_img = cv2.imread('static/image/CatEar.png', cv2.IMREAD_UNCHANGED)
cat_mask_img = cv2.imread('static/image/CatMask.png', cv2.IMREAD_UNCHANGED)
def overlay_cat_ears(frame, landmarks, left_ear_idx, right_ear_idx):
    h, w, _ = frame.shape
    left_ear = landmarks[left_ear_idx]
    right_ear = landmarks[right_ear_idx]
    
    # Calculate ear coordinates
    left_x, left_y = int(left_ear.x * w), int(left_ear.y * h)
    right_x, right_y = int(right_ear.x * w), int(right_ear.y * h)
    
    # Center and angle calculations
    center_x, center_y = (left_x + right_x) // 2, (left_y + right_y) // 2
    angle = -math.degrees(math.atan2(right_y - left_y, right_x - left_x))
    
    # Set ear dimensions based on distance
    ear_width = int(np.linalg.norm([right_x - left_x, right_y - left_y]) * 1.4)
    ear_height = int(ear_width * cat_ears_img.shape[0] / cat_ears_img.shape[1])
    
    # Check if overlay dimensions fit within frame boundaries
    top_left_x = center_x - ear_width // 2
    top_left_y = center_y - ear_height - 30  # Adjust position above the head
    
    # If overlay is outside the frame bounds, skip overlaying
    if top_left_x < 0 or top_left_y < 0 or (top_left_x + ear_width) > w or (top_left_y + ear_height) > h:
        return  # Skip overlay if it goes out of bounds
    
    # Flip the cat ears image if facing right
    if right_x > left_x:
        flipped_cat_ears_img = cv2.flip(cat_ears_img, 1)  # Horizontal flip
    else:
        flipped_cat_ears_img = cat_ears_img
    
    # Resize and rotate the cat ears
    resized_cat_ears = cv2.resize(flipped_cat_ears_img, (ear_width, ear_height))
    M = cv2.getRotationMatrix2D((ear_width // 2, ear_height // 2), angle, 1.0)
    rotated_cat_ears = cv2.warpAffine(resized_cat_ears, M, (ear_width, ear_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Overlay cat ears
    for c in range(3):
        frame[top_left_y:top_left_y + ear_height, top_left_x:top_left_x + ear_width, c] = (
            rotated_cat_ears[:ear_height, :ear_width, c] * (rotated_cat_ears[:ear_height, :ear_width, 3] / 255.0) +
            frame[top_left_y:top_left_y + ear_height, top_left_x:top_left_x + ear_width, c] * (1.0 - rotated_cat_ears[:ear_height, :ear_width, 3] / 255.0)
        )
    

def overlay_cat_mask(frame, landmarks, nose_idx, chin_idx):
    h, w, _ = frame.shape
    # Get the nose and chin landmarks
    nose = landmarks[nose_idx]
    chin = landmarks[chin_idx]
    
    # Calculate the position of the nose and chin
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    chin_x, chin_y = int(chin.x * w), int(chin.y * h)
    
    # Calculate the center position and angle for the mask
    center_x, center_y = (nose_x + chin_x) // 2, (nose_y + chin_y) // 2
    angle = -math.degrees(math.atan2(chin_y - nose_y, chin_x - nose_x)) + 90
    
    # Calculate the mask dimensions based on the distance from the nose to the chin
    mask_width = int(np.linalg.norm([chin_x - nose_x, chin_y - nose_y]) * 1.6)
    mask_height = int(mask_width * cat_mask_img.shape[0] / cat_mask_img.shape[1])
    
    # Resize the cat mask image to match the calculated dimensions
    resized_cat_mask = cv2.resize(cat_mask_img, (mask_width, mask_height))
    
    # Check the direction of the head and flip the mask if facing right
    if chin_x > nose_x:
        resized_cat_mask = cv2.flip(resized_cat_mask, 1)  # Flip horizontally
    
    # Rotate the cat mask image to match the angle of the line between the nose and chin
    M = cv2.getRotationMatrix2D((mask_width // 2, mask_height // 2), angle, 1.0)
    rotated_cat_mask = cv2.warpAffine(resized_cat_mask, M, (mask_width, mask_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Calculate where to place the mask, aligning it from the nose to the chin area
    top_left_x = max(center_x - mask_width // 2, 0)
    top_left_y = max(center_y - mask_height // 2, 0)
    
    # Ensure the mask stays within the frame boundaries
    overlay_width = min(mask_width, w - top_left_x)
    overlay_height = min(mask_height, h - top_left_y)
    
    # Resize the rotated mask to fit within the frame
    rotated_cat_mask = cv2.resize(rotated_cat_mask, (overlay_width, overlay_height))
    
    # Overlay the cat mask onto the frame
    for c in range(3):
        frame[top_left_y:top_left_y + overlay_height, top_left_x:top_left_x + overlay_width, c] = (
            rotated_cat_mask[:overlay_height, :overlay_width, c] * (rotated_cat_mask[:overlay_height, :overlay_width, 3] / 255.0) +
            frame[top_left_y:top_left_y + overlay_height, top_left_x:top_left_x + overlay_width, c] * (1.0 - rotated_cat_mask[:overlay_height, :overlay_width, 3] / 255.0)
        )

# filter no 2
dog_ears_img = cv2.imread('static/image/DogEar.png', cv2.IMREAD_UNCHANGED)
dog_mask_img = cv2.imread('static/image/DogMask.png', cv2.IMREAD_UNCHANGED)

def overlay_dog_ears(frame, landmarks, left_ear_idx, right_ear_idx):
    h, w, _ = frame.shape
    left_ear = landmarks[left_ear_idx]
    right_ear = landmarks[right_ear_idx]
    
    # Calculate ear coordinates
    left_x, left_y = int(left_ear.x * w), int(left_ear.y * h)
    right_x, right_y = int(right_ear.x * w), int(right_ear.y * h)
    
    # Center and angle calculations
    center_x, center_y = (left_x + right_x) // 2, (left_y + right_y) // 2
    angle = -math.degrees(math.atan2(right_y - left_y, right_x - left_x))
    
    # Set ear dimensions based on distance
    ear_width = int(np.linalg.norm([right_x - left_x, right_y - left_y]) * 1.4)
    ear_height = int(ear_width * dog_ears_img.shape[0] / dog_ears_img.shape[1])
    
    # Check if overlay dimensions fit within frame boundaries
    top_left_x = center_x - ear_width // 2
    top_left_y = center_y - ear_height - 30  # Adjust position above the head
    
    # If overlay is outside the frame bounds, skip overlaying
    if top_left_x < 0 or top_left_y < 0 or (top_left_x + ear_width) > w or (top_left_y + ear_height) > h:
        return  # Skip overlay if it goes out of bounds
    
    # Flip the cat ears image if facing right
    if right_x > left_x:
        flipped_dog_ears_img = cv2.flip(dog_ears_img, 1)  # Horizontal flip
    else:
        flipped_dog_ears_img = dog_ears_img
    
    # Resize and rotate the  ears
    resized_dog_ears = cv2.resize(flipped_dog_ears_img, (ear_width, ear_height))
    M = cv2.getRotationMatrix2D((ear_width // 2, ear_height // 2), angle, 1.0)
    rotated_dog_ears = cv2.warpAffine(resized_dog_ears, M, (ear_width, ear_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Overlay cat ears
    for c in range(3):
        frame[top_left_y:top_left_y + ear_height, top_left_x:top_left_x + ear_width, c] = (
            rotated_dog_ears[:ear_height, :ear_width, c] * (rotated_dog_ears[:ear_height, :ear_width, 3] / 255.0) +
            frame[top_left_y:top_left_y + ear_height, top_left_x:top_left_x + ear_width, c] * (1.0 - rotated_dog_ears[:ear_height, :ear_width, 3] / 255.0)
        )
    

def overlay_dog_mask(frame, landmarks, nose_idx, chin_idx):
    h, w, _ = frame.shape
    # Get the nose and chin landmarks
    nose = landmarks[nose_idx]
    chin = landmarks[chin_idx]
    
    # Calculate the position of the nose and chin
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    chin_x, chin_y = int(chin.x * w), int(chin.y * h)
    
    # Calculate the center position and angle for the mask
    center_x, center_y = (nose_x + chin_x) // 2, (nose_y + chin_y) // 2
    angle = -math.degrees(math.atan2(chin_y - nose_y, chin_x - nose_x)) + 90
    
    # Calculate the mask dimensions based on the distance from the nose to the chin
    mask_width = int(np.linalg.norm([chin_x - nose_x, chin_y - nose_y]) * 1.6)
    mask_height = int(mask_width * dog_mask_img.shape[0] / dog_mask_img.shape[1])
    
    # Resize the cat mask image to match the calculated dimensions
    resized_dog_mask = cv2.resize(dog_mask_img, (mask_width, mask_height))
    
    # Check the direction of the head and flip the mask if facing right
    if chin_x > nose_x:
        resized_dog_mask = cv2.flip(resized_dog_mask, 1)  # Flip horizontally
    
    # Rotate the cat mask image to match the angle of the line between the nose and chin
    M = cv2.getRotationMatrix2D((mask_width // 2, mask_height // 2), angle, 1.0)
    rotated_dog_mask = cv2.warpAffine(resized_dog_mask, M, (mask_width, mask_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Calculate where to place the mask, aligning it from the nose to the chin area
    top_left_x = max(center_x - mask_width // 2, 0)
    top_left_y = max(center_y - mask_height // 2, 0)
    
    # Ensure the mask stays within the frame boundaries
    overlay_width = min(mask_width, w - top_left_x)
    overlay_height = min(mask_height, h - top_left_y)
    
    # Resize the rotated mask to fit within the frame
    rotated_cat_mask = cv2.resize(rotated_dog_mask, (overlay_width, overlay_height))
    
    # Overlay the cat mask onto the frame
    for c in range(3):
        frame[top_left_y:top_left_y + overlay_height, top_left_x:top_left_x + overlay_width, c] = (
            rotated_cat_mask[:overlay_height, :overlay_width, c] * (rotated_cat_mask[:overlay_height, :overlay_width, 3] / 255.0) +
            frame[top_left_y:top_left_y + overlay_height, top_left_x:top_left_x + overlay_width, c] * (1.0 - rotated_cat_mask[:overlay_height, :overlay_width, 3] / 255.0)
        )

# filter no 3
glasses_img = cv2.imread('static/image/Glasses.png', cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread('static/image/Mustache.png', cv2.IMREAD_UNCHANGED)
cowboy_hat_img = cv2.imread('static/image/CowboyHat.png', cv2.IMREAD_UNCHANGED)

def overlay_glasses(frame, landmarks, left_eye_idx, right_eye_idx):
    # Get the coordinates of the left and right eyes
    left_eye = (int(landmarks[left_eye_idx].x * frame.shape[1]), int(landmarks[left_eye_idx].y * frame.shape[0]))
    right_eye = (int(landmarks[right_eye_idx].x * frame.shape[1]), int(landmarks[right_eye_idx].y * frame.shape[0]))

    # Calculate the center point between the eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2 + 10)
    
    # Calculate the angle between the eyes in degrees
    angle = -np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Calculate the distance between the eyes
    eye_distance = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)))

    # Resize the glasses image to fit the width between the eyes
    glasses_width = int(eye_distance * 2)
    glasses_height = int(glasses_img.shape[0] * (glasses_width / glasses_img.shape[1]))
    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

    # Rotate the glasses image by the calculated angle
    M = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), angle, 1)
    rotated_glasses = cv2.warpAffine(resized_glasses, M, (glasses_width, glasses_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Determine the region where the glasses will be overlaid
    y1 = int(eye_center[1] - glasses_height // 2)
    y2 = y1 + glasses_height
    x1 = int(eye_center[0] - glasses_width // 2)
    x2 = x1 + glasses_width

    # Ensure the coordinates are within the frame boundaries
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return  # Do not overlay if out of bounds

    # Extract the alpha channel from the rotated glasses image
    alpha_glasses = rotated_glasses[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_glasses

    # Overlay the rotated glasses image on the frame
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (
            alpha_glasses * rotated_glasses[:, :, c] +
            alpha_frame * frame[y1:y2, x1:x2, c]
        )

def overlay_mustache(frame, landmarks, left_mouth_idx, right_mouth_idx, upper_lip_idx):
    # Get the coordinates of the left and right corners of the mouth and the upper lip
    left_mouth = (int(landmarks[left_mouth_idx].x * frame.shape[1]), int(landmarks[left_mouth_idx].y * frame.shape[0]))
    right_mouth = (int(landmarks[right_mouth_idx].x * frame.shape[1]), int(landmarks[right_mouth_idx].y * frame.shape[0]))
    upper_lip = (int(landmarks[upper_lip_idx].x * frame.shape[1]), int(landmarks[upper_lip_idx].y * frame.shape[0]))

    # Calculate the center point of the mustache overlay
    mustache_center = ((left_mouth[0] + right_mouth[0]) // 2, upper_lip[1] - 15 )


    angle = -np.degrees(np.arctan2(right_mouth[1] - left_mouth[1], right_mouth[0] - left_mouth[0]))

    # Calculate the width and height of the mustache based on the distance between the mouth corners
    mustache_width = int(np.linalg.norm(np.array(left_mouth) - np.array(right_mouth)) * 2)
    mustache_height = int(mustache_img.shape[0] * (mustache_width / mustache_img.shape[1]))
    resized_mustache = cv2.resize(mustache_img, (mustache_width, mustache_height), interpolation=cv2.INTER_AREA)

    # Rotate the glasses image by the calculated angle
    M = cv2.getRotationMatrix2D((mustache_width // 2, mustache_height // 2), angle, 1)
    rotated_glasses = cv2.warpAffine(resized_mustache, M, (mustache_width, mustache_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


    # Determine the region where the mustache will be overlaid
    y1 = int(mustache_center[1] - mustache_height // 2)
    y2 = y1 + mustache_height
    x1 = int(mustache_center[0] - mustache_width // 2)
    x2 = x1 + mustache_width

    # Ensure the coordinates are within the frame boundaries
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return  # Do not overlay if out of bounds

    # Extract the alpha channel from the resized mustache image (assuming it's an RGBA image)
    alpha_mustache = rotated_glasses[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_mustache

    # Overlay the mustache image on the frame
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (
            alpha_mustache * rotated_glasses[:, :, c] +
            alpha_frame * frame[y1:y2, x1:x2, c]
        )

def overlay_cowboy_hat(frame, landmarks, upper_head_idx):
    # Get the coordinates of the upper head (assuming index 10 is the upper head)
    upper_head = (int(landmarks[upper_head_idx].x * frame.shape[1]), int(landmarks[upper_head_idx].y * frame.shape[0]))

    # Calculate the width of the face (distance between eyes or mouth) to scale the hat size
    face_width = int(np.linalg.norm(np.array((landmarks[33].x, landmarks[33].y)) - np.array((landmarks[263].x, landmarks[263].y))) * frame.shape[1])

    # Resize the cowboy hat to match the width of the face
    hat_width = face_width * 3
    hat_height = int(cowboy_hat_img.shape[0] * (hat_width / cowboy_hat_img.shape[1]))
    resized_hat = cv2.resize(cowboy_hat_img, (int(hat_width), hat_height), interpolation=cv2.INTER_AREA)

    # Position the hat above the head landmark (a little above the upper head)
    hat_center = (upper_head[0], upper_head[1] - int(hat_height * 0.4))  

    # Determine the region where the cowboy hat will be overlaid
    y1 = int(hat_center[1] - hat_height // 2)
    y2 = int(y1 + hat_height)  # Ensure y2 is also an integer
    x1 = int(hat_center[0] - hat_width // 2)
    x2 = int(x1 + hat_width)  # Ensure x2 is also an integer

    # Ensure the coordinates are within the frame boundaries
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return  # Do not overlay if out of bounds

    # Extract the alpha channel from the resized cowboy hat image (assuming it's an RGBA image)
    alpha_hat = resized_hat[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_hat

    # Overlay the cowboy hat image on the frame
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (
            alpha_hat * resized_hat[:, :, c] +
            alpha_frame * frame[y1:y2, x1:x2, c]
        )
choice = 0
def generate_filter_face_frames():
    global choice
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Proses deteksi wajah
        if results.multi_face_landmarks:
            overlay_frame = frame.copy()

            # Loop untuk setiap wajah yang terdeteksi
            for face_landmarks in results.multi_face_landmarks:
                # mp_drawing.draw_landmarks(overlay_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                landmarks = face_landmarks.landmark

                # Terapkan filter sesuai pilihan
                if choice == 0:
                    pass
                elif choice == 1:
                    overlay_cat_ears(overlay_frame, landmarks, 454, 234)  
                    overlay_cat_mask(overlay_frame, landmarks, 5, 152)   
                elif choice == 2:
                    overlay_dog_ears(overlay_frame, landmarks, 454, 234)  
                    overlay_dog_mask(overlay_frame, landmarks, 5, 152)  
                elif choice == 3:
                    overlay_glasses(overlay_frame, landmarks, 33, 263)
                    overlay_mustache(overlay_frame, landmarks, 61, 291, 11)
                    overlay_cowboy_hat(overlay_frame, landmarks, 10)

        _, buffer = cv2.imencode('.jpg', overlay_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/filter_face_feed')
def filter_face_feed():
    return Response(generate_filter_face_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/filterdetail/<int:id>")
def filterdetailpage(id):
    global choice
    choice =int(id)
    return render_template('FilterDetailPage.html', filter_id=id)

# game
# Initialize MediaPipe Hands and Drawing modules
pygame.mixer.init()
bomb_hit_time = None
game_started = False
game_over = False
time_out = None
score = 0
game_timer = 0  # Track game duration
max_game_duration = 60
last_spawn_time = time.time()
remaining_time = 0
wait_metal_cd = None

objects = []
splashes = []

slash_points = []
slash_color = (255, 255, 255)
slash_length = 2

# Load images (ensure paths are correct)
watermelon_img = cv2.imread('static/image/watermelon.png', cv2.IMREAD_UNCHANGED)
pineapple_img = cv2.imread('static/image/pineapple.png', cv2.IMREAD_UNCHANGED)
banana_img = cv2.imread('static/image/banana.png', cv2.IMREAD_UNCHANGED)
apple_img = cv2.imread('static/image/apple.png', cv2.IMREAD_UNCHANGED)
bomb_img = cv2.imread('static/image/bomb.png', cv2.IMREAD_UNCHANGED)

splash_red = cv2.imread('static/image/splash_red.png', cv2.IMREAD_UNCHANGED)
splash_yellow = cv2.imread('static/image/splash_yellow.png', cv2.IMREAD_UNCHANGED)
splash_explosive = cv2.imread('static/image/explosion.png', cv2.IMREAD_UNCHANGED)

score_logo = cv2.imread('static/image/score.png', cv2.IMREAD_UNCHANGED) 
game_bg = cv2.imread('static/image/background.jpg', cv2.IMREAD_UNCHANGED)
game_bg = cv2.resize(game_bg, (640, 480))

game_over_img = cv2.imread('static/image/game-over.png', cv2.IMREAD_UNCHANGED)

game_start_sound = 'static/sound/Game-start.wav'
game_over_sound = 'static/sound/Game-over.wav'
throw_bomb_sound = 'static/sound/Throw-bomb.wav'
throw_fruit_sound = 'static/sound/Throw-fruit.wav'
bomb_explode_sound = 'static/sound/powerup-deflect-explode.wav'
slice_sound = 'static/sound/pome-slice-1.wav'


# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def is_peace_sign(hand_landmarks):
    """Detects if the hand is making a 'peace' sign"""
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    
    index_up = index_finger_tip.y < hand_landmarks.landmark[6].y
    middle_up = middle_finger_tip.y < hand_landmarks.landmark[10].y

    thumb_finger_tip = hand_landmarks.landmark[4]
    thumb_curl = thumb_finger_tip.y > hand_landmarks.landmark[3].y

    ring_finger_tip = hand_landmarks.landmark[16]
    ring_curl = ring_finger_tip.y > hand_landmarks.landmark[14].y

    pinky_tip = hand_landmarks.landmark[20]
    pinky_curl = pinky_tip.y > hand_landmarks.landmark[18].y

    return index_up and middle_up and thumb_curl and ring_curl and pinky_curl

def is_metal_pose(hand_landmarks):
    """
    Detects if the hand is making the 'metal' pose.
    The index and pinky fingers are extended, while the middle and ring fingers are folded.
    The thumb is extended outward.
    """
    landmarks = hand_landmarks.landmark

    thumb_extended = landmarks[4].x > landmarks[3].x
    pinky_extended = landmarks[20].y < landmarks[18].y

    index_extended = landmarks[8].y < landmarks[6].y

    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y

    return thumb_extended and index_extended and pinky_extended and middle_curled and ring_curled


image_width = 800  
image_height = 600  

@app.route('/update_frame_size', methods=['POST'])
def update_frame_size():
    global image_width, image_height
    data = request.get_json()
    image_width = data['width']
    image_height = data['height']
    return jsonify({"status": "success"})

global out_off_game
out_off_game = False
def game_generate_frame():
    """Generate a frame to stream to the browser"""
    global game_started, game_over, can_piece, score, game_timer, last_spawn_time, objects, splashes, slash_color, slash_points, slash_length, remaining_time, bomb_hit_time, time_out, out_off_game, wait_metal_cd , can_metal
    out_off_game = False
    # cap = cv2.VideoCapture(0)
    game_over = False
    can_piece = True
    can_metal = True
    play_over_sound = True
    
    print(image_width, image_height)

    while camera.isOpened() and out_off_game == False:
        ret, frame = camera.read()
        if not ret:
            print("Ignoring empty frame.")
            break

        h, w, c = frame.shape
        frame = cv2.flip(frame, 1)
        
        frame = cv2.resize(frame, (w, h))
        # print(h,w)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # h = image_height
        # w = image_width
        results = hands.process(rgb_frame)
        output_frame = game_bg.copy()
        # output_frame = cv2.resize(output_frame,(1397,928))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output_frame,
                    None,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )   

                if is_metal_pose(hand_landmarks) and can_metal:
                    pygame.mixer.music.load(game_over_sound)
                    pygame.mixer.music.play()
                    game_started = False
                    can_metal = False
                    score = 0
                    remaining_time = 0
                    wait_metal_cd = time.time()
                    game_over = True
                    objects.clear()


                if not game_started and is_peace_sign(hand_landmarks) and can_piece:
                    # Start or Restart Game
                    game_started = True
                    game_over = False
                    can_piece = False
                    score = 0
                    game_timer = time.time()
                    last_spawn_time = time.time()
                    objects.clear()
                    splashes.clear()
                    print("Game Started!")
                    pygame.mixer.music.load(game_start_sound)
                    pygame.mixer.music.play()

                if bomb_hit_time is not None:
                        elapsed_over_time = time.time() - bomb_hit_time
                        elapsed_can_start_time = time.time() - bomb_hit_time
                        if elapsed_over_time >= 1.5 and play_over_sound:  
                            pygame.mixer.music.load(game_over_sound)
                            pygame.mixer.music.play()
                            game_over = True
                            play_over_sound = False
                        if elapsed_can_start_time >= 4.2:
                            can_piece = True
                            bomb_hit_time = None
                            play_over_sound = True


                if game_started:
                    
                    index_finger_tip = hand_landmarks.landmark[8]
                    index_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                    slash_points.append(index_pos)
                    if len(slash_points) > slash_length:
                        slash_points.pop(0)

                    for i in range(1, len(slash_points)):
                        cv2.line(output_frame, slash_points[i - 1], slash_points[i], slash_color, 3)

                    for obj in objects[:]:
                        distance = math.sqrt((index_pos[0] - obj["x"])**2 + (index_pos[1] - obj["y"])**2)
                        if distance < obj["radius"]:
                            if obj["type"] == "bomb":
                                print("Bomb hit! Game Over!")
                                bomb_hit_time = None
                                bomb_hit_time = time.time() 
                                splashes.append({"x": obj["x"], "y": obj["y"], "image": splash_explosive, "ttl": 20})
                                objects.remove(obj)
                                game_started = False
                                pygame.mixer.music.load(bomb_explode_sound)
                                pygame.mixer.music.play()
                            else:
                                score += 1
                                splash_img = splash_red if obj["type"] in ["apple", "watermelon"] else splash_yellow
                                splashes.append({"x": obj["x"], "y": obj["y"], "image": splash_img, "ttl": 20})
                                objects.remove(obj)
                                pygame.mixer.music.load(slice_sound)
                                pygame.mixer.music.play()
                            
        if wait_metal_cd is not None:
            wait_metal_time = time.time() - wait_metal_cd
            if wait_metal_time >= 4.3 :
                out_off_game = True
                wait_metal_cd = None
                can_metal = True

        if game_started:
            remaining_time = max_game_duration - int(time.time() - game_timer)
            if remaining_time <= 0:
                print("Time's up! Game Over!")
                pygame.mixer.music.load(game_over_sound)
                pygame.mixer.music.play()
                time_out = time.time()
                game_started = False
                game_over = True

        if time_out is not None:
            e_time_out = time.time() - time_out
            if e_time_out >= 4.2:
                can_piece = True
                time_out = None

        current_time = time.time()
        if current_time - last_spawn_time >= 1.5 and game_started:
            last_spawn_time = current_time
            for _ in range(random.randint(1, 4)):  
                obj_type = "bomb" if random.random() <= 0.2 else random.choice(["watermelon", "pineapple", "banana", "apple"])
                if obj_type == "bomb":
                    pygame.mixer.music.load(throw_bomb_sound)
                    pygame.mixer.music.play()
                else:
                    pygame.mixer.music.load(throw_fruit_sound)
                    pygame.mixer.music.play()

                # Random horizontal position (scaled by frame width)
                x = random.randint(80, w - 40)

                # Adjust the vertical launch height (scaled by frame height)
                # Increase the launch height to a higher proportion of the frame height
                launch_height = random.randint(40, int(h * 0.1))  # Adjusted to 70% of frame height for higher launch

                # Vertical velocity (scaled by frame height)
                vy = random.uniform(-14, -12) * (h / h)  # Increase the vertical velocity for a higher launch

                # Random horizontal velocity
                vx = random.uniform(-2, 2)

                # Random angle and rotation speed
                angle = random.randint(0, 360)
                rotation_speed = random.uniform(2, 6)

                # Get the object image
                img = eval(f"{obj_type}_img")

                # Create object with adjusted launch position and velocity
                obj = {
                    "type": obj_type,
                    "x": x,
                    "y": h - launch_height,  # Adjusted launch height
                    "vx": vx,
                    "vy": vy,  # Adjusted vertical velocity
                    "radius": 40,
                    "image": img,
                    "angle": angle,
                    "rotation_speed": rotation_speed
                }
                objects.append(obj)


        # Iterate over all objects
        for obj in objects[:]:
            obj["vy"] += 0.2  # Apply gravity
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            # Remove object if it goes below the bottom of the screen
            if obj["y"] - obj["radius"] > h:
                objects.remove(obj)
                continue
            
            # Reverse horizontal velocity if the object hits the left or right frame edge
            if obj["x"] - obj["radius"] < 0 or obj["x"] + obj["radius"] > w:
                obj["vx"] = -obj["vx"]
                # Ensure the object stays within the frame horizontally
                obj["x"] = max(obj["radius"], min(w - obj["radius"], obj["x"]))

            obj["angle"] += obj["rotation_speed"]
            obj["angle"] %= 360

            if obj["image"] is not None:
                img = obj["image"]
                img_h, img_w = img.shape[:2]

                # Calculate scale_factor based on the frame size
                scale_factor = 1  # You can adjust this as necessary based on the frame size
                new_width = int(img_w * scale_factor)
                new_height = int(img_h * scale_factor)

                # Resize the image
                img_resized = cv2.resize(img, (new_width, new_height))

                # Apply rotation and positioning
                center = (img_resized.shape[1] // 2, img_resized.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, obj["angle"], 1.0)
                img_rotated = cv2.warpAffine(
                    img_resized, rotation_matrix,
                    (img_resized.shape[1], img_resized.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )

                # Calculate position and check bounds
                x1, y1 = int(obj["x"] - img_rotated.shape[1] / 2), int(obj["y"] - img_rotated.shape[0] / 2)
                x2, y2 = x1 + img_rotated.shape[1], y1 + img_rotated.shape[0]

                # Calculate cropping bounds to keep the image within the frame
                crop_x1 = max(0, x1)
                crop_y1 = max(0, y1)
                crop_x2 = min(w, x2)
                crop_y2 = min(h, y2)

                # Determine the corresponding region in the rotated image
                img_crop_x1 = max(0, -x1)
                img_crop_y1 = max(0, -y1)
                img_crop_x2 = img_crop_x1 + (crop_x2 - crop_x1)
                img_crop_y2 = img_crop_y1 + (crop_y2 - crop_y1)

                # Apply alpha mask for transparency
                alpha_mask = img_rotated[img_crop_y1:img_crop_y2, img_crop_x1:img_crop_x2, 3] / 255.0
                cropped_img = img_rotated[img_crop_y1:img_crop_y2, img_crop_x1:img_crop_x2, :3]

                # Blend the cropped image with the output frame
                output_frame[crop_y1:crop_y2, crop_x1:crop_x2] = (
                    output_frame[crop_y1:crop_y2, crop_x1:crop_x2] * (1 - alpha_mask[:, :, None]) +
                    cropped_img * alpha_mask[:, :, None]
                )




        for splash in splashes[:]:
            splash_img = splash["image"]
            splash_x, splash_y = splash["x"], splash["y"]
            splash_ttl = splash["ttl"]
        
            splash["ttl"] -= 1
            if splash_ttl <= 0:
                splashes.remove(splash)
                continue
            
            splash_h, splash_w = splash_img.shape[:2]
            scale_factor = 0.6
            splash_resized = cv2.resize(splash_img, (int(splash_w * scale_factor), int(splash_h * scale_factor)))
            splash_x1 = int(splash_x - splash_resized.shape[1] / 2)
            splash_y1 = int(splash_y - splash_resized.shape[0] / 2)
            splash_x2 = splash_x1 + splash_resized.shape[1]
            splash_y2 = splash_y1 + splash_resized.shape[0]
        
            # Clip the coordinates to ensure the splash fits within the frame
            splash_x1_clipped = max(0, splash_x1)
            splash_y1_clipped = max(0, splash_y1)
            splash_x2_clipped = min(w, splash_x2)
            splash_y2_clipped = min(h, splash_y2)
        
            if splash_x2_clipped > splash_x1_clipped and splash_y2_clipped > splash_y1_clipped:
                splash_resized_clipped = splash_resized[splash_y1_clipped - splash_y1:splash_y2_clipped - splash_y1,
                                                        splash_x1_clipped - splash_x1:splash_x2_clipped - splash_x1]
                alpha_mask = splash_resized_clipped[:, :, 3] / 255.0
        
                output_frame[splash_y1_clipped:splash_y2_clipped, splash_x1_clipped:splash_x2_clipped] = \
                    output_frame[splash_y1_clipped:splash_y2_clipped, splash_x1_clipped:splash_x2_clipped] * (1 - alpha_mask[:, :, None]) + \
                    splash_resized_clipped[:, :, :3] * alpha_mask[:, :, None]

        # Overlay the score image (score_img) on the frame
        score_img_h, score_img_w = score_logo.shape[:2]
        x_score_img, y_score_img = 20, 20  # Top-left corner of the score image
        x_score_end, y_score_end = x_score_img + score_img_w, y_score_img + score_img_h

        # Ensure the overlay fits within the frame
        if x_score_end <= w and y_score_end <= h:
            alpha_score = score_logo[:, :, 3] / 255.0  # Alpha channel for transparency
            for c in range(3):  # Apply to BGR channels
                output_frame[y_score_img:y_score_end, x_score_img:x_score_end, c] = (
                    score_logo[:, :, c] * alpha_score +
                    output_frame[y_score_img:y_score_end, x_score_img:x_score_end, c] * (1 - alpha_score)
                )

        # Add numeric score next to the score image
        text_x = x_score_end + 10  # Position the text slightly to the right of the score image
        text_y = y_score_img + score_img_h // 2 + 10  # Center the text vertically with the score image
        cv2.putText(output_frame, str(score), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if game_started:
            remaining_time = max_game_duration - int(time.time() - game_timer)
        cv2.putText(output_frame, f"Time: {remaining_time}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if game_over:
            # Get the dimensions of the overlay image
            overlay_height, overlay_width = game_over_img.shape[:2]

            # Get the dimensions of the frame
            frame_height, frame_width = output_frame.shape[:2]

            # Calculate the center position of the overlay image
            x_center = (frame_width - overlay_width) // 2
            y_center = (frame_height - overlay_height) // 2

            # Resize the overlay if necessary (preserve aspect ratio)
            overlay_img_resized = cv2.resize(game_over_img, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

            # Overlay the image (considering alpha channel if present)
            if game_over_img.shape[2] == 4:  # Check if the image has an alpha channel
                alpha_channel = game_over_img[:, :, 3] / 255.0  # Normalize alpha channel
                for c in range(3):  # Iterate over BGR channels
                    output_frame[y_center:y_center + overlay_height, x_center:x_center + overlay_width, c] = (
                        game_over_img[:, :, c] * alpha_channel + 
                        output_frame[y_center:y_center + overlay_height, x_center:x_center + overlay_width, c] * (1 - alpha_channel)
                    )
            else:  # If no alpha channel, directly overlay the image
                output_frame[y_center:y_center + overlay_height, x_center:x_center + overlay_width] = overlay_img_resized


        # Encode frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/game_video_feed')
def game_video_feed():
    return Response(game_generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game')
def gamepage():
    return render_template('GamePage.html')

@app.route('/sse_game_status')
def sse_game_status():
    def event_stream():
        global out_off_game
        while not out_off_game:
            yield f"data: running\n\n"
            time.sleep(1)  # Interval untuk mengirim status (1 detik)
        yield f"data: redirect\n\n"

    return Response(event_stream(), content_type='text/event-stream')

# register data train
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()

cap_register = cv2.VideoCapture(0)
userid = None

def is_smiling(landmarks):
    mouth_points = landmarks[48:68]

    top_lip = mouth_points[2].y
    bottom_lip = mouth_points[10].y
    left_corner = mouth_points[0].x
    right_corner = mouth_points[6].x

    vertical_distance = bottom_lip - top_lip
    horizontal_distance = right_corner - left_corner

    mar = vertical_distance / horizontal_distance

    # print(mar)

    if mar < 0.3:
        return True
    else:
        return False

def register_face_generate_frames():
    global userid
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray_frame)
        
        ori_frame = frame.copy()

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        
        center_width = width // 6  
        center_height = height // 6  

        cv2.rectangle(frame, (center_x - center_width, center_y - center_height), (center_x + center_width, center_y + center_height), (0, 255, 255), 2)


        for face in faces:
            face_center_x = (face.left() + face.right()) // 2
            face_center_y = (face.top() + face.bottom()) // 2

            if (center_x - center_width <= face_center_x <= center_x + center_width) and \
               (center_y - center_height <= face_center_y <= center_y + center_height):

                landmarks = landmark_predictor(gray_frame, face)

                if is_smiling(landmarks.parts()):
                    folder_path = f"static/facerecog/train/{userid}"
                    os.makedirs(folder_path, exist_ok=True)
                    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
                    screenshot_filename = os.path.join(folder_path, f"face{file_count + 1}.jpg")

                    cv2.imwrite(screenshot_filename, ori_frame)
                    print(f"Saved: {screenshot_filename}")

                    cv2.rectangle(frame, (center_x - center_width, center_y - center_height), 
                      (center_x + center_width, center_y + center_height), (255, 0, 0), 2)
                else:
                    cv2.rectangle(frame, (center_x - center_width, center_y - center_height), 
                      (center_x + center_width, center_y + center_height), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (center_x - center_width, center_y - center_height), (center_x + center_width, center_y + center_height), (0, 255, 255), 2)        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/register_face_video_feed')
def register_face_video_feed():
    return Response(register_face_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registerface/<int:id>')
def registerfacepage(id):
    global userid
    userid = id
    folder_path = f'static/facerecog/train/{id}'
    os.makedirs(folder_path, exist_ok=True)
    return render_template('RegisterFacePage.html')

def train_model():
    faces = []
    labels = []
    label_map = {}  # Map to store label -> user_id

    for user_id in os.listdir('static/facerecog/train'):
        user_folder_path = f'static/facerecog/train/{user_id}'
        
        if os.path.isdir(user_folder_path):  
            label_map[len(label_map)] = user_id  # Store label with user_id as folder name
            
            for image_name in os.listdir(user_folder_path):
                image_path = os.path.join(user_folder_path, image_name)
                image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                faces_detected = face_detector(gray_image)
                if len(faces_detected) == 1:  # Ensure exactly one face is detected
                    face = faces_detected[0]
                    landmarks = landmark_predictor(gray_image, face)
                    face_region = gray_image[face.top():face.bottom(), face.left():face.right()]
                    
                    faces.append(face_region)
                    labels.append(len(label_map) - 1)  # Use the current label index

    lbph_recognizer.train(faces, np.array(labels))  # Train the model
    lbph_recognizer.save('face_recognizer_model.yml')  # Save the trained model
    np.save('label_map.npy', label_map)  # Save label map for later use
    print("Model trained and saved successfully.")



def test_face(image_frame):
    gray_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_detector(gray_image)
    correct_predictions = 0

    # Load the label map (saved during training)
    label_map = np.load('label_map.npy', allow_pickle=True).item()

    for face in faces_detected:
        face_region = gray_image[face.top():face.bottom(), face.left():face.right()]

        label, confidence = lbph_recognizer.predict(face_region)
        print(f"Predicted Label: {label_map[label]}")
        print(f"Confidence: {confidence}")

        if confidence < 60:
            return int(label_map[label])
        else:
            return -1



global redirection_triggered, test_data
redirection_triggered = False


def login_face_generate_frames():
    global redirection_triggered, test_data
    smile_start_time = None  
    smile_duration = 3  

    while not redirection_triggered:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)
        ori_frame = frame.copy()

        # Center region coordinates
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        center_width = width // 6  
        center_height = height // 6  

        cv2.rectangle(frame, (center_x - center_width, center_y - center_height), 
                      (center_x + center_width, center_y + center_height), (0, 255, 255), 2)

        for face in faces:
            face_center_x = (face.left() + face.right()) // 2
            face_center_y = (face.top() + face.bottom()) // 2

            if (center_x - center_width <= face_center_x <= center_x + center_width) and \
               (center_y - center_height <= face_center_y <= center_y + center_height):
                landmarks = landmark_predictor(gray_frame, face)

                if is_smiling(landmarks.parts()):
                    if smile_start_time is None:
                        smile_start_time = time.time()

                    elapsed_time = time.time() - smile_start_time
                    if elapsed_time >= smile_duration:
                        train_model()
                        test_data = test_face(ori_frame)
                        redirection_triggered = True
                        break

                    cv2.rectangle(frame, (center_x - center_width, center_y - center_height), 
                                  (center_x + center_width, center_y + center_height), (255, 0, 0), 2)
                else:
                    smile_start_time = None
                    cv2.rectangle(frame, (center_x - center_width, center_y - center_height), 
                                  (center_x + center_width, center_y + center_height), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (center_x - center_width, center_y - center_height), 
                              (center_x + center_width, center_y + center_height), (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if redirection_triggered:
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'redirect\r\n')

@app.route('/login_face_video_feed')
def login_face_video_feed():
    return Response( login_face_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sse_status')
def sse_status():
    def event_stream():
        global redirection_triggered
        while not redirection_triggered:
            yield f"data: running\n\n"
            time.sleep(1)  # Interval untuk mengirim status (1 detik)
        yield f"data: redirect\n\n"

    return Response(event_stream(), content_type='text/event-stream')

@app.route("/")
def index():
    return redirect(url_for("loginpage"))

@app.route("/filter")
def filterpage():
    return render_template('FilterPage.html')


@app.route("/profile")
def profilepage():
    global userid
    userid = session['id']

    return render_template('ProfilePage.html')

@app.route("/home")
def homepage():
    return render_template('HomePage.html')

@app.route("/login", methods=["POST", "GET"])
def loginpage():
    global redirection_triggered, test_data
    if redirection_triggered:
        if test_data != -1:
            existing_user = User.query.filter_by(id=test_data).first()
            if existing_user:
                session['id'] = existing_user.id
                session['username'] = existing_user.username
                redirection_triggered = False
                return redirect(url_for('homepage')) 
        else:
            redirection_triggered = False
            flash("Try Again")
            return redirect(url_for('loginpage'))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            if existing_user.password == password:
                print(existing_user.id)
                session['id'] = existing_user.id
                session['username'] = username
                return redirect(url_for('homepage')) 
            else:
                flash("Your username or password is incorrect")
                return redirect(url_for('loginpage'))
        else:
            flash("Username does not exist. Please register.")
            return redirect(url_for('registerpage')) 

    return render_template('LoginPage.html')

@app.route("/register", methods=["POST", "GET"])
def registerpage():
    if request.method == "POST":
        username = request.form["Username"]
        password = request.form["Password"]
        confirmPassword = request.form["ConfirmPassword"]

        if len(username) < 6 or len(username) > 15:
            flash("Username must be between 6 and 15 characters")
            return redirect(url_for('registerpage'))
        
        if len(password) < 7:
            flash("Password must be more than 6 characters")
            return redirect(url_for('registerpage'))

        if not any(char.isalpha() for char in password) or not any(char.isdigit() for char in password):
            flash("Password must contain at least 1 letter and 1 number")
            return redirect(url_for('registerpage'))

        if password != confirmPassword:
            flash("Passwords do not match")
            return redirect(url_for('registerpage'))

        new_user = User(username, password)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully!")
        return redirect(url_for('loginpage'))
    return render_template('RegisterPage.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('loginpage'))

if __name__ == "__main__":
    app.run(debug=True)
