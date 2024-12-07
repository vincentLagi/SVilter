import base64
import logging
import os
import random
import re
import cv2.data
from django import conf
from flask import Flask, Response, flash, jsonify, redirect, send_file, session, url_for, render_template, request
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
from PIL import Image, ImageDraw, ImageFont

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
camera.set(cv2.CAP_PROP_FPS, 30)


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

    # Overlay position
    top_left_x = center_x - ear_width // 2
    top_left_y = center_y - ear_height - 30  # Adjust position above the head

    # Compute overlay cropping boundaries
    overlay_start_x = max(0, -top_left_x)
    overlay_start_y = max(0, -top_left_y)
    overlay_end_x = min(ear_width, w - top_left_x)
    overlay_end_y = min(ear_height, h - top_left_y)

    # Check for valid overlay dimensions
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # Skip overlay if out of bounds

    # Adjust frame boundaries
    frame_start_x = max(0, top_left_x)
    frame_start_y = max(0, top_left_y)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)

    # Flip the cat ears image if facing right
    if right_x > left_x:
        flipped_cat_ears_img = cv2.flip(cat_ears_img, 1)  # Horizontal flip
    else:
        flipped_cat_ears_img = cat_ears_img

    # Resize and rotate the cat ears
    resized_cat_ears = cv2.resize(flipped_cat_ears_img, (ear_width, ear_height))
    M = cv2.getRotationMatrix2D((ear_width // 2, ear_height // 2), angle, 1.0)
    rotated_cat_ears = cv2.warpAffine(resized_cat_ears, M, (ear_width, ear_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Crop the rotated image to fit the valid region
    cropped_cat_ears = rotated_cat_ears[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Overlay cropped cat ears onto the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            cropped_cat_ears[:, :, c] * (cropped_cat_ears[:, :, 3] / 255.0) +
            frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] * (1.0 - cropped_cat_ears[:, :, 3] / 255.0)
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

    # Calculate where to place the mask
    top_left_x = center_x - mask_width // 2
    top_left_y = center_y - mask_height // 2

    # Compute overlay cropping boundaries
    overlay_start_x = max(0, -top_left_x)
    overlay_start_y = max(0, -top_left_y)
    overlay_end_x = min(mask_width, w - top_left_x)
    overlay_end_y = min(mask_height, h - top_left_y)

    # Ensure valid overlay dimensions
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # Skip overlay if it goes out of bounds

    # Adjust frame boundaries
    frame_start_x = max(0, top_left_x)
    frame_start_y = max(0, top_left_y)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)

    # Crop the rotated mask to fit the valid region
    cropped_cat_mask = rotated_cat_mask[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Overlay the cropped cat mask onto the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            cropped_cat_mask[:, :, c] * (cropped_cat_mask[:, :, 3] / 255.0) +
            frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] * (1.0 - cropped_cat_mask[:, :, 3] / 255.0)
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
    
    # Initial placement for overlay
    top_left_x = center_x - ear_width // 2
    top_left_y = center_y - ear_height - 30  # Adjust position above the head
    
    # Calculate cropping bounds for overlay
    overlay_start_x = max(0, -top_left_x)
    overlay_start_y = max(0, -top_left_y)
    overlay_end_x = min(ear_width, w - top_left_x)
    overlay_end_y = min(ear_height, h - top_left_y)
    
    # Skip overlay if the overlay is completely out of bounds
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # No valid area to overlay
    
    # Adjust frame bounds for the overlay
    frame_start_x = max(0, top_left_x)
    frame_start_y = max(0, top_left_y)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)
    
    # Flip the dog ears image if facing right
    if right_x > left_x:
        flipped_dog_ears_img = cv2.flip(dog_ears_img, 1)  # Horizontal flip
    else:
        flipped_dog_ears_img = dog_ears_img
    
    # Resize and rotate the dog ears
    resized_dog_ears = cv2.resize(flipped_dog_ears_img, (ear_width, ear_height))
    M = cv2.getRotationMatrix2D((ear_width // 2, ear_height // 2), angle, 1.0)
    rotated_dog_ears = cv2.warpAffine(resized_dog_ears, M, (ear_width, ear_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Crop the rotated dog ears to fit the valid overlay area
    cropped_dog_ears = rotated_dog_ears[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]
    
    # Overlay the cropped dog ears onto the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            cropped_dog_ears[:, :, c] * (cropped_dog_ears[:, :, 3] / 255.0) +
            frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] * (1.0 - cropped_dog_ears[:, :, 3] / 255.0)
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
    
    # Resize the mask to match the calculated dimensions
    resized_dog_mask = cv2.resize(dog_mask_img, (mask_width, mask_height))
    
    # Check the direction of the head and flip the mask if necessary
    if chin_x > nose_x:
        resized_dog_mask = cv2.flip(resized_dog_mask, 1)  # Flip horizontally
    
    # Rotate the mask to match the angle of the line between the nose and chin
    M = cv2.getRotationMatrix2D((mask_width // 2, mask_height // 2), angle, 1.0)
    rotated_dog_mask = cv2.warpAffine(resized_dog_mask, M, (mask_width, mask_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Calculate the top-left position for placing the mask
    top_left_x = center_x - mask_width // 2
    top_left_y = center_y - mask_height // 2
    
    # Calculate cropping bounds for the overlay
    overlay_start_x = max(0, -top_left_x)
    overlay_start_y = max(0, -top_left_y)
    overlay_end_x = min(mask_width, w - top_left_x)
    overlay_end_y = min(mask_height, h - top_left_y)
    
    # Skip overlaying if the mask is completely out of bounds
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # No valid area to overlay
    
    # Adjust the frame bounds for the overlay
    frame_start_x = max(0, top_left_x)
    frame_start_y = max(0, top_left_y)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)
    
    # Crop the rotated mask to fit the valid overlay area
    cropped_mask = rotated_dog_mask[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]
    
    # Overlay the cropped mask onto the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            cropped_mask[:, :, c] * (cropped_mask[:, :, 3] / 255.0) +
            frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] * (1.0 - cropped_mask[:, :, 3] / 255.0)
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

    # Calculate cropping bounds for the overlay
    overlay_start_x = max(0, -x1)
    overlay_start_y = max(0, -y1)
    overlay_end_x = min(glasses_width, frame.shape[1] - x1)
    overlay_end_y = min(glasses_height, frame.shape[0] - y1)

    # Skip overlaying if the overlay is completely out of bounds
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # No valid area to overlay

    # Adjust the frame bounds for the overlay
    frame_start_x = max(0, x1)
    frame_start_y = max(0, y1)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)

    # Crop the rotated glasses image to fit the valid overlay area
    cropped_glasses = rotated_glasses[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Extract the alpha channel from the cropped glasses image
    alpha_glasses = cropped_glasses[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_glasses

    # Overlay the cropped glasses image on the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            alpha_glasses * cropped_glasses[:, :, c] +
            alpha_frame * frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c]
        )


def overlay_mustache(frame, landmarks, left_mouth_idx, right_mouth_idx, upper_lip_idx):
    # Get the coordinates of the left and right corners of the mouth and the upper lip
    left_mouth = (int(landmarks[left_mouth_idx].x * frame.shape[1]), int(landmarks[left_mouth_idx].y * frame.shape[0]))
    right_mouth = (int(landmarks[right_mouth_idx].x * frame.shape[1]), int(landmarks[right_mouth_idx].y * frame.shape[0]))
    upper_lip = (int(landmarks[upper_lip_idx].x * frame.shape[1]), int(landmarks[upper_lip_idx].y * frame.shape[0]))

    # Calculate the center point of the mustache overlay
    mustache_center = ((left_mouth[0] + right_mouth[0]) // 2, upper_lip[1] - 15)

    # Calculate the angle between the mouth corners in degrees
    angle = -np.degrees(np.arctan2(right_mouth[1] - left_mouth[1], right_mouth[0] - left_mouth[0]))

    # Calculate the width and height of the mustache based on the distance between the mouth corners
    mustache_width = int(np.linalg.norm(np.array(left_mouth) - np.array(right_mouth)) * 2)
    mustache_height = int(mustache_img.shape[0] * (mustache_width / mustache_img.shape[1]))
    resized_mustache = cv2.resize(mustache_img, (mustache_width, mustache_height), interpolation=cv2.INTER_AREA)

    # Rotate the mustache image by the calculated angle
    M = cv2.getRotationMatrix2D((mustache_width // 2, mustache_height // 2), angle, 1)
    rotated_mustache = cv2.warpAffine(resized_mustache, M, (mustache_width, mustache_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Determine the region where the mustache will be overlaid
    y1 = int(mustache_center[1] - mustache_height // 2)
    y2 = y1 + mustache_height
    x1 = int(mustache_center[0] - mustache_width // 2)
    x2 = x1 + mustache_width

    # Calculate cropping bounds for the overlay
    overlay_start_x = max(0, -x1)
    overlay_start_y = max(0, -y1)
    overlay_end_x = min(mustache_width, frame.shape[1] - x1)
    overlay_end_y = min(mustache_height, frame.shape[0] - y1)

    # Skip overlaying if the overlay is completely out of bounds
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # No valid area to overlay

    # Adjust the frame bounds for the overlay
    frame_start_x = max(0, x1)
    frame_start_y = max(0, y1)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)

    # Crop the rotated mustache image to fit the valid overlay area
    cropped_mustache = rotated_mustache[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Extract the alpha channel from the cropped mustache image (assuming it's an RGBA image)
    alpha_mustache = cropped_mustache[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_mustache

    # Overlay the cropped mustache image on the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            alpha_mustache * cropped_mustache[:, :, c] +
            alpha_frame * frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c]
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

    # Calculate cropping bounds for the overlay
    overlay_start_x = max(0, -x1)
    overlay_start_y = max(0, -y1)
    overlay_end_x = min(hat_width, frame.shape[1] - x1)
    overlay_end_y = min(hat_height, frame.shape[0] - y1)

    # Skip overlaying if the overlay is completely out of bounds
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # No valid area to overlay

    # Adjust the frame bounds for the overlay
    frame_start_x = max(0, x1)
    frame_start_y = max(0, y1)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)

    # Crop the resized cowboy hat image to fit the valid overlay area
    cropped_hat = resized_hat[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Extract the alpha channel from the cropped cowboy hat image (assuming it's an RGBA image)
    alpha_hat = cropped_hat[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_hat

    # Overlay the cropped cowboy hat image on the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            alpha_hat * cropped_hat[:, :, c] +
            alpha_frame * frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c]
        )

# 4
ssn_img = cv2.imread('static/image/ssn.png', cv2.IMREAD_UNCHANGED)
ss2_img = cv2.imread('static/image/ss2.png', cv2.IMREAD_UNCHANGED)
ssb_img = cv2.imread('static/image/ssb.png', cv2.IMREAD_UNCHANGED)

def overlay_hair(frame, landmarks, head_center_idx):
    h, w, _ = frame.shape
    head_center = landmarks[head_center_idx]

    # Calculate head center coordinates
    center_x, center_y = int(head_center.x * w), int(head_center.y * h)

    # Set hair dimensions based on head width (can use distance between eyes or other facial landmarks)
    # For simplicity, let's assume you want the width based on a predefined scale for head size
    head_width = int(0.35 * w)  # Adjust the width to be 35% of the frame's width
    hair_width = int(head_width * 1.8)  # Hair width can be wider than the head
    hair_height = int(hair_width * ssn_img.shape[0] / ssn_img.shape[1])  # Keep aspect ratio

    # Overlay position
    top_left_x = center_x - hair_width // 2
    top_left_y = center_y - hair_height - 50  # Adjust position above the head

    # Compute overlay cropping boundaries
    overlay_start_x = max(0, -top_left_x)
    overlay_start_y = max(0, -top_left_y)
    overlay_end_x = min(hair_width, w - top_left_x)
    overlay_end_y = min(hair_height, h - top_left_y)

    # Check for valid overlay dimensions
    if overlay_end_x <= overlay_start_x or overlay_end_y <= overlay_start_y:
        return  # Skip overlay if out of bounds

    # Adjust frame boundaries
    frame_start_x = max(0, top_left_x)
    frame_start_y = max(0, top_left_y)
    frame_end_x = frame_start_x + (overlay_end_x - overlay_start_x)
    frame_end_y = frame_start_y + (overlay_end_y - overlay_start_y)

    # Resize hair image
    resized_hair = cv2.resize(ssn_img, (hair_width, hair_height))

    # Crop the resized hair to fit the valid region
    cropped_hair = resized_hair[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

    # Overlay cropped hair onto the frame
    for c in range(3):  # Loop over color channels
        frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] = (
            cropped_hair[:, :, c] * (cropped_hair[:, :, 3] / 255.0) +
            frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x, c] * (1.0 - cropped_hair[:, :, 3] / 255.0)
        )

aura_mp4 = 'static/mp4/YellowAura.mp4'
aura_video = cv2.VideoCapture(aura_mp4)

def is_mouth_open(landmarks, frame_width, frame_height, top_lip_index, bottom_lip_index, threshold=0.02):
    """
    Check if the mouth is open based on facial landmarks.
    """
    top_lip = landmarks[top_lip_index]
    bottom_lip = landmarks[bottom_lip_index]

    top_y = top_lip.y * frame_height
    bottom_y = bottom_lip.y * frame_height

    mouth_open_distance = bottom_y - top_y
    return mouth_open_distance > (threshold * frame_height)


def overlay_aura(frame, aura_frame):
    """
    Overlay the aura video frame onto the input frame.
    """
    aura_h, aura_w, _ = aura_frame.shape
    h, w, _ = frame.shape

    # Resize the aura frame to fit the input frame
    scale_factor = 0.5  # Adjust as needed
    resized_aura = cv2.resize(aura_frame, (int(w * scale_factor), int(h * scale_factor)))

    # Calculate position for overlay (center of frame)
    x_offset = (w - resized_aura.shape[1]) // 2
    y_offset = (h - resized_aura.shape[0]) // 2

    # Blend aura frame into the main frame
    for y in range(resized_aura.shape[0]):
        for x in range(resized_aura.shape[1]):
            if resized_aura[y, x, 3] > 0:  # Alpha channel check
                alpha = resized_aura[y, x, 3] / 255.0
                for c in range(3):  # Blend RGB channels
                    frame[y + y_offset, x + x_offset, c] = (
                        alpha * resized_aura[y, x, c] +
                        (1 - alpha) * frame[y + y_offset, x + x_offset, c]
                    )


global choice
choice = 0
def generate_filter_face_frames(width, height):
    global choice, overlay_frame
    while True:
        # print("2")
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        
        # Resize only after processing
        overlay_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Proses deteksi wajah
        if results.multi_face_landmarks:
            # Loop untuk setiap wajah yang terdeteksi
            for face_landmarks in results.multi_face_landmarks:
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

        # Resize the final frame before sending it to the frontend
        resized_frame = cv2.resize(overlay_frame, (width, height))
        
        # Encode and yield the frame
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/filter_face_feed')
def filter_face_feed():
    width = int(request.args.get('width', 640))  # Default width is 640
    height = int(request.args.get('height', 480))  # Default height is 480
    
    # Generate the filter face frames with the given width and height
    return Response(generate_filter_face_frames(width, height), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/filterdetail/<int:id>")
def filterdetailpage(id):
    global choice
    choice =int(id)
    return redirect(url_for("filterpage"))


@app.route('/save_image')
def save_image():
    success, buffer = cv2.imencode('.png', overlay_frame)
    if not success:
        return "Image encoding error", 500

    return Response(
        buffer.tobytes(),
        mimetype='image/png',  
        headers={
            'Content-Disposition': 'attachment; filename="captured_image.png"'  
        }
    )

# @app.route('/download_image', methods=['GET'])
# def download_image():
#     # Path to the processed image (could be dynamic depending on your logic)
#     image_path = 'path_to_your_image/processed_feed.jpg'

#     # Send the image as a downloadable file
#     return send_file(image_path, as_attachment=True, download_name="processed_feed.jpg")




# game
# Initialize MediaPipe Hands and Drawing modules
pygame.mixer.init()

fruit_ninja_font_style = 'static/fontstyle/FruitNinjaFontStyle.ttf'

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

class AllState:
    def __init__(self):
        self.takePic = 0
        self.userid = None
        self.redirection_triggered = False
        self.cameraloss = False
        self.test_data =  int(0)

all_state = AllState()

class GameState:
    def __init__(self):
        self.bomb_hit_time = float(-1)
        self.game_started = False
        self.game_over = False
        self.time_out = float(-1)
        self.wait_metal_cd = float(-1)
        self.out_off_game = False
        self.can_piece = True
        self.can_metal = True
        self.play_over_sound = True
        self.score = 0
        self.game_timer = float(-1)  
        self.max_game_duration = 60
        self.last_spawn_time = time.time()
        self.remaining_time = 0
        self.objects = []
        self.splashes = []
        self.slash_points = []
        self.slash_color = (255, 255, 255)
        self.slash_length = 5
    
    def reset(self):
        self.bomb_hit_time = float(-1)
        self.game_started = False
        self.game_over = False
        self.time_out = float(-1)
        self.wait_metal_cd = float(-1)
        self.out_off_game = False
        self.can_piece = True
        self.can_metal = True
        self.play_over_sound = True
        self.score = 0
        self.game_timer = float(-1)  
        self.max_game_duration = 60
        self.last_spawn_time = time.time()
        self.remaining_time = 0
        self.objects = []
        self.splashes = []
        self.slash_points = []
        self.slash_color = (255, 255, 255)
        self.slash_length = 5



game_state = GameState()

def game_generate_frame():
    """Generate a frame to stream to the browser"""

    # cap = cv2.VideoCapture(0)

    # print(image_width, image_height)

    while camera.isOpened() and game_state.out_off_game == False:
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

                if is_metal_pose(hand_landmarks) and game_state.can_metal:
                    pygame.mixer.music.load(game_over_sound)
                    pygame.mixer.music.play()
                    game_state.game_started = False
                    game_state.can_metal = False
                    game_state.can_piece = False
                    game_state.score = 0
                    game_state.remaining_time = 0
                    game_state.wait_metal_cd = time.time()
                    game_state.game_over = True
                    game_state.objects.clear()


                if not game_state.game_started and is_peace_sign(hand_landmarks) and game_state.can_piece:
                    # Start or Restart Game
                    game_state.game_started = True
                    game_state.game_over = False
                    game_state.can_piece = False
                    game_state.score = 0
                    game_state.game_timer = time.time()
                    game_state.last_spawn_time = time.time()
                    game_state.objects.clear()
                    game_state.splashes.clear()
                    print("Game Started!")
                    pygame.mixer.music.load(game_start_sound)
                    pygame.mixer.music.play()

                if game_state.bomb_hit_time != -1:
                        print("cek hellow ashiafshias")
                        elapsed_over_time = time.time() - game_state.bomb_hit_time
                        elapsed_can_start_time = time.time() - game_state.bomb_hit_time
                        if elapsed_over_time >= 1.5 and game_state.play_over_sound:  
                            pygame.mixer.music.load(game_over_sound)
                            pygame.mixer.music.play()
                            game_state.game_over = True
                            game_state.play_over_sound = False
                        if elapsed_can_start_time >= 4.2:
                            game_state.can_piece = True
                            game_state.bomb_hit_time = -1
                            game_state.play_over_sound = True


                if game_state.game_started:
                    
                    index_finger_tip = hand_landmarks.landmark[8]
                    index_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                    game_state.slash_points.append(index_pos)
                    if len(game_state.slash_points) > game_state.slash_length:
                        game_state.slash_points.pop(0)

                    for i in range(1, len(game_state.slash_points)):
                        cv2.line(output_frame, game_state.slash_points[i - 1], game_state.slash_points[i], game_state.slash_color, 3)

                    cv2.circle(output_frame, index_pos, 5, game_state.slash_color, -1)
                    for obj in game_state.objects[:]:
                        distance = math.sqrt((index_pos[0] - obj["x"])**2 + (index_pos[1] - obj["y"])**2)
                        if distance < obj["radius"]:
                            if obj["type"] == "bomb":
                                print("Bomb hit! Game Over!")
                                game_state.bomb_hit_time = -1
                                game_state.bomb_hit_time = time.time() 
                                game_state.splashes.append({"x": obj["x"], "y": obj["y"], "image": splash_explosive, "ttl": 20})
                                game_state.objects.remove(obj)
                                game_state.game_started = False
                                pygame.mixer.music.load(bomb_explode_sound)
                                pygame.mixer.music.play()
                            else:
                                game_state.score += 1
                                splash_img = splash_red if obj["type"] in ["apple", "watermelon"] else splash_yellow
                                game_state.splashes.append({"x": obj["x"], "y": obj["y"], "image": splash_img, "ttl": 20})
                                game_state.objects.remove(obj)
                                pygame.mixer.music.load(slice_sound)
                                pygame.mixer.music.play()
                            
        if game_state.wait_metal_cd != -1:
            wait_metal_time = time.time() - game_state.wait_metal_cd
            if wait_metal_time >= 4.3 :
                game_state.out_off_game = True
                game_state.wait_metal_cd = -1
                game_state.can_metal = True
                

        if game_state.game_started:
            remaining_time = game_state.max_game_duration - int(time.time() - game_state.game_timer)
            if remaining_time < 0:
                print("Time's up! Game Over!")
                pygame.mixer.music.load(game_over_sound)
                pygame.mixer.music.play()
                game_state.time_out = time.time()
                game_state.game_started = False
                game_state.game_over = True

        if game_state.time_out != -1:
            e_time_out = time.time() - game_state.time_out
            if e_time_out >= 4.2:
                game_state.can_piece = True
                game_state.time_out = -1

        current_time = time.time()
        if current_time - game_state.last_spawn_time >= 2 and game_state.game_started:
            game_state.last_spawn_time = current_time
            for _ in range(random.randint(1, 3)):  
                obj_type = "bomb" if random.random() <= 0.1 else random.choice(["watermelon", "pineapple", "banana", "apple"])
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
                vy = random.uniform(-14, -8) * (h / h)  # Increase the vertical velocity for a higher launch

                # Random horizontal velocity
                vx = random.uniform(-4, 2)

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
                game_state.objects.append(obj)


        # Iterate over all game_state.objects
        for obj in game_state.objects[:]:
            obj["vy"] += 0.2  # Apply gravity
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            # Remove object if it goes below the bottom of the screen
            if obj["y"] - obj["radius"] > h:
                game_state.objects.remove(obj)
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




        for splash in game_state.splashes[:]:
            splash_img = splash["image"]
            splash_x, splash_y = splash["x"], splash["y"]
            splash_ttl = splash["ttl"]
        
            splash["ttl"] -= 1
            if splash_ttl <= 0:
                game_state.splashes.remove(splash)
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
        pil_image = Image.fromarray(output_frame)
        font = ImageFont.truetype(fruit_ninja_font_style, 40) 
        draw = ImageDraw.Draw(pil_image)
        # Add numeric score next to the score image
        text_x = x_score_end + 10  # Position the text slightly to the right of the score image
        text_y = y_score_img + score_img_h // 2 - 20  # Center the text vertically with the score image
        draw.text((text_x,text_y), str(game_state.score), font=font, fill=(0, 255, 255))

        # cv2.putText(output_frame, str(score), (text_x, text_y), fruit_ninja_font_style, 1, (0, 255, 255), 2)
        if game_state.game_started:
            game_state.remaining_time = game_state.max_game_duration - int(time.time() - game_state.game_timer)
        # cv2.putText(output_frame, f"Time: {remaining_time}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        time_text = str(game_state.remaining_time)

        # Get the bounding box of the text
        bbox = draw.textbbox((0, 0), time_text, font=font)  # (left, top, right, bottom)
        text_width = bbox[2] - bbox[0]  # width of the text

        # Position the text on the right
        text_x_time = w - text_width - 20  # Right-align the text with a margin of 20 pixels
        text_y_time = 20  # Top margin
        draw.text((text_x_time, text_y_time), time_text, font=font, fill=(255, 255, 255))

        output_frame = np.array(pil_image)
        if game_state.game_over:
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
            return None

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
        print(game_state.out_off_game)
        while not game_state.out_off_game:
            yield f"data: running\n\n"
            time.sleep(1)  # Interval untuk mengirim status (1 detik)
        yield f"data: redirect\n\n"

    return Response(event_stream(), content_type='text/event-stream')



# register data train
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()

cap_register = cv2.VideoCapture(0)

face_outline_img = cv2.imread('static/image/faceOutline.png', cv2.IMREAD_UNCHANGED)

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
    circle_angle = 0
    while True:
        if all_state.takePic == 1:
            ret, frame = camera.read()

            if not ret:
                all_state.cameraloss = True
                # print("Failed to grab frame from camera")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray_frame)
            
            ori_frame = frame.copy()

            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            
            center_width = width // 6  
            center_height = height // 6  

            color = (0, 255, 255)

            for face in faces:
                face_center_x = (face.left() + face.right()) // 2
                face_center_y = (face.top() + face.bottom()) // 2

                if (center_x - center_width <= face_center_x <= center_x + center_width) and \
                    (center_y - center_height <= face_center_y <= center_y + center_height):

                    landmarks = landmark_predictor(gray_frame, face)

                    if is_smiling(landmarks.parts()) and all_state.takePic == 1:
                        folder_path = f"static/facerecog/train/{all_state.userid}"
                        os.makedirs(folder_path, exist_ok=True)
                        file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
                        screenshot_filename = os.path.join(folder_path, f"face{file_count + 1}.jpg")

                        cv2.imwrite(screenshot_filename, ori_frame)
                        print(f"Saved: {screenshot_filename}")
                        
                        color = (255, 0, 0)

                    else:
                        color = (0, 255, 0)
                else:
                    color = (0, 255, 255)        
            face_outline_resized = cv2.resize(face_outline_img, (height // 2, height // 2))  

            alpha_channel = face_outline_resized[:, :, 3] / 255.0  
            overlay_color = face_outline_resized[:, :, :3]

            overlay_color[:, :, 0] = overlay_color[:, :, 0] * (1 - alpha_channel) + color[0] * alpha_channel
            overlay_color[:, :, 1] = overlay_color[:, :, 1] * (1 - alpha_channel) + color[1] * alpha_channel
            overlay_color[:, :, 2] = overlay_color[:, :, 2] * (1 - alpha_channel) + color[2] * alpha_channel

            face_outline_resized[:, :, :3] = overlay_color

            y1, y2 = center_y - (face_outline_resized.shape[0] // 2), center_y + (face_outline_resized.shape[0] // 2)
            x1, x2 = center_x - (face_outline_resized.shape[1] // 2), center_x + (face_outline_resized.shape[1] // 2)

            for c in range(0, 3):  
                frame[y1:y2, x1:x2, c] = frame[y1:y2, x1:x2, c] * (1 - alpha_channel) + face_outline_resized[:, :, c] * alpha_channel

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            frame_size = (480, 640, 3)
            solid_color = (255, 123, 0)  
            
            base_frame = np.full(frame_size, solid_color, dtype=np.uint8)

            overlay = np.zeros(frame_size, dtype=np.uint8)
            overlay[:] = (255, 255, 255)  

            alpha = 0.3 
            placeholder_frame = cv2.addWeighted(base_frame, 1 - alpha, overlay, alpha, 0)

            center = (320, 240)  
            radius = 50 
            circle_thickness = 5
            circle_color = (255, 255, 255)  

            start_angle = circle_angle  
            end_angle = circle_angle + 45  

            if end_angle > 360:
                cv2.ellipse(placeholder_frame, center, (radius, radius), 0, start_angle, 360, circle_color, circle_thickness)
                cv2.ellipse(placeholder_frame, center, (radius, radius), 0, 0, end_angle - 360, circle_color, circle_thickness)
            else:
                cv2.ellipse(placeholder_frame, center, (radius, radius), 0, start_angle, end_angle, circle_color, circle_thickness)

            circle_angle = (circle_angle + 5) % 360

            pil_image = Image.fromarray(placeholder_frame)
            font = ImageFont.truetype(fruit_ninja_font_style, 30)
            draw = ImageDraw.Draw(pil_image)

            text = "Waiting for camera to open"
            bbox = draw.textbbox((0, 0), text, font=font)  
            text_width = bbox[2] - bbox[0]  
            text_height = bbox[3] - bbox[1]  

            text_x = (frame_size[1] - text_width) // 2
            text_y = (frame_size[0] - text_height) // 2
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

            placeholder_frame = np.array(pil_image)

            _, buffer = cv2.imencode('.jpg', placeholder_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(0.05)

        
@app.route('/register_face_video_feed')
def register_face_video_feed():
    return Response(register_face_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registerface/<int:id>')
def registerfacepage(id):
    all_state.userid = id
    folder_path = f'static/facerecog/train/{id}'
    os.makedirs(folder_path, exist_ok=True)
    return render_template('RegisterFacePage.html')

def train_model():
    faces = []
    labels = []
    label_map = {} 

    for user_id in os.listdir('static/facerecog/train'):
        user_folder_path = f'static/facerecog/train/{user_id}'
        
        if os.path.isdir(user_folder_path):  
            label_map[len(label_map)] = user_id 
            
            for image_name in os.listdir(user_folder_path):
                image_path = os.path.join(user_folder_path, image_name)
                image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                faces_detected = face_detector(gray_image)
                if len(faces_detected) == 1: 
                    face = faces_detected[0]
                    landmarks = landmark_predictor(gray_image, face)
                    face_region = gray_image[face.top():face.bottom(), face.left():face.right()]
                    
                    faces.append(face_region)
                    labels.append(len(label_map) - 1)   

    if len(faces) > 0:
        lbph_recognizer.train(faces, np.array(labels))  
        lbph_recognizer.save('face_recognizer_model.yml')  # Save the model
        np.save('label_map.npy', label_map)  # Save the label map
        print("Model trained and saved successfully.")
    else:
        # If no faces are found, save an empty model as a "null" model
        lbph_recognizer.empty()  # Reset the model (this makes it "null")
        lbph_recognizer.save('face_recognizer_model.yml')  # Save the empty model
        np.save('label_map.npy', {})  # Save an empty label map
        print("No faces found. Null model saved.")

train_model()


def test_face(image_frame):
 
    gray_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_detector(gray_image)

    label_map = np.load('label_map.npy', allow_pickle=True).item()
    print(lbph_recognizer)

    for face in faces_detected:
        print("cek")
        face_region = gray_image[face.top():face.bottom(), face.left():face.right()]
        print(face_region)
        try:
            label, confidence = lbph_recognizer.predict(face_region)
        except cv2.error:
            print('disini -2')
            return -2 
        print(f"Predicted Label: {label_map[label]}")
        print(f"Confidence: {confidence}")

        if confidence < 60:
            return int(label_map[label])
        else:
            return -1
    return -1


class SmileDuration:
    def __init__(self):
        self.smile_start_time = float(-1)
    
    def reset (self):
        self.smile_start_time = float(-1)


smileDuration = SmileDuration()


def login_face_generate_frames():
    smile_duration = 3  
    circle_angle = 0
    while not all_state.redirection_triggered :
        if all_state.takePic == 1:
            ret, frame = camera.read()
            if not ret:
                all_state.cameraloss = True
                # print("Failed to grab frame from camera")
                break  

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray_frame)  
            ori_frame = frame.copy()

            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            center_width, center_height = width // 6, height // 6

            color = (0, 255, 255)

            for face in faces:
                face_center_x = (face.left() + face.right()) // 2
                face_center_y = (face.top() + face.bottom()) // 2

                
                if (center_x - center_width <= face_center_x <= center_x + center_width) and \
                (center_y - center_height <= face_center_y <= center_y + center_height):

                    landmarks = landmark_predictor(gray_frame, face)  

                    if is_smiling(landmarks.parts()) and all_state.takePic == 1:  
                        print('Smiling detected')

                        if smileDuration.smile_start_time == -1:
                            smileDuration.smile_start_time = time.time()

                        elapsed_time = time.time() - smileDuration.smile_start_time
                        if elapsed_time >= smile_duration:
                            all_state.test_data = test_face(ori_frame)  
                            all_state.redirection_triggered = True
                            break

                        color = (255, 0, 0)

                    else:
                        smileDuration.smile_start_time = -1  

                        color = (0, 255, 0)  

                else:
                    color = (0, 255, 255)  

            face_outline_resized = cv2.resize(face_outline_img, (height // 2, height // 2))  

            alpha_channel = face_outline_resized[:, :, 3] / 255.0  
            overlay_color = face_outline_resized[:, :, :3]

            overlay_color[:, :, 0] = overlay_color[:, :, 0] * (1 - alpha_channel) + color[0] * alpha_channel
            overlay_color[:, :, 1] = overlay_color[:, :, 1] * (1 - alpha_channel) + color[1] * alpha_channel
            overlay_color[:, :, 2] = overlay_color[:, :, 2] * (1 - alpha_channel) + color[2] * alpha_channel

            face_outline_resized[:, :, :3] = overlay_color

            y1, y2 = center_y - (face_outline_resized.shape[0] // 2), center_y + (face_outline_resized.shape[0] // 2)
            x1, x2 = center_x - (face_outline_resized.shape[1] // 2), center_x + (face_outline_resized.shape[1] // 2)

            for c in range(0, 3):  
                frame[y1:y2, x1:x2, c] = frame[y1:y2, x1:x2, c] * (1 - alpha_channel) + face_outline_resized[:, :, c] * alpha_channel

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # print("disini kah?")
            frame_size = (480, 640, 3)
            solid_color = (255, 123, 0)  
            
            base_frame = np.full(frame_size, solid_color, dtype=np.uint8)

            overlay = np.zeros(frame_size, dtype=np.uint8)
            overlay[:] = (255, 255, 255)  

            alpha = 0.3 
            placeholder_frame = cv2.addWeighted(base_frame, 1 - alpha, overlay, alpha, 0)

            center = (320, 240)  
            radius = 50 
            circle_thickness = 5
            circle_color = (255, 255, 255)  

            start_angle = circle_angle  
            end_angle = circle_angle + 45  

            if end_angle > 360:
                cv2.ellipse(placeholder_frame, center, (radius, radius), 0, start_angle, 360, circle_color, circle_thickness)
                cv2.ellipse(placeholder_frame, center, (radius, radius), 0, 0, end_angle - 360, circle_color, circle_thickness)
            else:
                cv2.ellipse(placeholder_frame, center, (radius, radius), 0, start_angle, end_angle, circle_color, circle_thickness)

            circle_angle = (circle_angle + 5) % 360

            pil_image = Image.fromarray(placeholder_frame)
            font = ImageFont.truetype(fruit_ninja_font_style, 30)
            draw = ImageDraw.Draw(pil_image)

            text = "Waiting for camera to open"
            bbox = draw.textbbox((0, 0), text, font=font)  
            text_width = bbox[2] - bbox[0]  
            text_height = bbox[3] - bbox[1]  

            text_x = (frame_size[1] - text_width) // 2
            text_y = (frame_size[0] - text_height) // 2
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

            placeholder_frame = np.array(pil_image)

            _, buffer = cv2.imencode('.jpg', placeholder_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(0.05)
    
    print('check')
    if all_state.redirection_triggered:
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'redirect\r\n')

    

@app.route('/login_face_video_feed')
def login_face_video_feed():

    return Response( login_face_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sse_status')
def sse_status():
    def event_stream():
        while not all_state.redirection_triggered and not all_state.cameraloss:
            yield f"data: running\n\n"
            time.sleep(1)  
        if all_state.redirection_triggered:
            yield f"data: redirect\n\n"
        elif all_state.cameraloss:
            yield f"data: cameraloss\n\n"
            all_state.cameraloss = False

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/training/<int:id>', methods=['POST'])
def training(id):
    all_state.takePic = int(id)

    if(all_state.takePic == 0):

        try:
            train_model()
            return jsonify({'success': True, 'message': 'Model training started successfully'}), 200
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    else:
        return jsonify({'success': True, 'message': 'Nice'}), 200

@app.route('/takeVid/<int:id>', methods=['POST'])
def takeVid(id):
    all_state.takePic = int(id)
    return jsonify({'success': True, 'message': 'Nice'}), 200

@app.route("/")
def index():
    return redirect(url_for("loginpage"))

@app.route("/filter")
def filterpage():
    return render_template('FilterPage.html')


@app.route("/profile")
def profilepage():
    all_state.userid = session['id']
    return render_template('ProfilePage.html')

@app.route("/home")
def homepage():
    game_state.reset()
    print(all_state.redirection_triggered)
    print(all_state.test_data)
    return render_template('HomePage.html')

@app.route("/login", methods=["POST", "GET"])
def loginpage():
    smileDuration.reset()
    if all_state.redirection_triggered:
        if all_state.test_data == -1:
            all_state.redirection_triggered = False
            flash("Try Again")
            return redirect(url_for('loginpage'))
        elif all_state.test_data == -2:
            all_state.redirection_triggered = False
            flash("No Models")
            return redirect(url_for('loginpage'))
        else:
            try:
                existing_user = User.query.filter_by(id=all_state.test_data).first()
                if existing_user:
                    session['id'] = existing_user.id
                    session['username'] = existing_user.username
                    all_state.redirection_triggered = False
                    return redirect(url_for('homepage')) 
            except Exception as e:
                flash("Connection to DB Error")
                all_state.redirection_triggered = False
                return redirect(url_for('loginpage'))


    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        try:
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
                return redirect(url_for('loginpage')) 
        except Exception as e:
            flash("Connection to DB Error")
            all_state.redirection_triggered = False
            return redirect(url_for('loginpage'))


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

        try:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash("Username already used")
                return redirect(url_for('registerpage'))
        except Exception as e:
            flash("Connection to DB Error")
            all_state.redirection_triggered = False
            return redirect(url_for('loginpage'))



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



@app.route('/check_camera', methods=['POST'])
def check_camera():
    try:
        print("Checking camera connection...")
        
        camera.open(0)    
        if camera.isOpened() and camera.grab():
            print("Camera detected")
            return jsonify({"camera_detected": True})
        else:
            print("No camera detected")
            time.sleep(2)
            return jsonify({"camera_detected": False})
    except Exception as e:
        print(f"Error checking camera: {e}")
        return jsonify({"camera_detected": False})

if __name__ == "__main__":
    app.run(debug=True)
