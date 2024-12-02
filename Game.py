import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# Load images with alpha channel
watermelon_img = cv2.imread('static/image/watermelon.png', cv2.IMREAD_UNCHANGED)
pineapple_img = cv2.imread('static/image/pineapple.png', cv2.IMREAD_UNCHANGED)
banana_img = cv2.imread('static/image/banana.png', cv2.IMREAD_UNCHANGED)
apple_img = cv2.imread('static/image/apple.png', cv2.IMREAD_UNCHANGED)
bomb_img = cv2.imread('static/image/bomb.png', cv2.IMREAD_UNCHANGED)

splash_red = cv2.imread('static/image/splash_red.png', cv2.IMREAD_UNCHANGED)
splash_yellow = cv2.imread('static/image/splash_yellow.png', cv2.IMREAD_UNCHANGED)
splash_explosive = cv2.imread('static/image/explosion.png', cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the MediaPipe Hands module
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Parameters for the slash effect
slash_points = []
slash_color = (255, 255, 255)
slash_length = 2

# Physics parameters
gravity = 0.2
spawn_interval = 1.5  # Spawn objects every 1.5 seconds
last_spawn_time = time.time()

# Object and splash lists
objects = []
splashes = []
score = 0
game_started = False
game_over = False
game_timer = 0  # Track game duration
max_game_duration = 60

# Open the webcam
cap = cv2.VideoCapture(0)

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty frame.")
        break

    h, w, c = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            if not game_started and is_peace_sign(hand_landmarks):
                # Start or Restart Game
                game_started = True
                game_over = False
                score = 0
                game_timer = time.time()
                last_spawn_time = time.time()
                objects.clear()
                splashes.clear()
                slash_points.clear()
                print("Game Started!")

            if game_started:
                index_finger_tip = hand_landmarks.landmark[8]
                index_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                slash_points.append(index_pos)
                if len(slash_points) > slash_length:
                    slash_points.pop(0)

                for i in range(1, len(slash_points)):
                    cv2.line(frame, slash_points[i - 1], slash_points[i], slash_color, 3)

                for obj in objects[:]:
                    distance = math.sqrt((index_pos[0] - obj["x"])**2 + (index_pos[1] - obj["y"])**2)
                    if distance < obj["radius"]:
                        if obj["type"] == "bomb":
                            print("Bomb hit! Game Over!")
                            splashes.append({"x": obj["x"], "y": obj["y"], "image": splash_explosive, "ttl": 20})
                            objects.remove(obj)
                            game_started = False
                            game_over = True
                        else:
                            score += 100
                            splash_img = splash_red if obj["type"] in ["apple", "watermelon"] else splash_yellow
                            splashes.append({"x": obj["x"], "y": obj["y"], "image": splash_img, "ttl": 20})
                            objects.remove(obj)

    if game_started:
        remaining_time = max_game_duration - int(time.time() - game_timer)
        if remaining_time <= 0:
            print("Time's up! Game Over!")
            game_started = False
            game_over = True

    if game_over:
        cv2.putText(frame, "Game Over! Show Peace Sign to Retry", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    current_time = time.time()
    if current_time - last_spawn_time >= spawn_interval and game_started:
        last_spawn_time = current_time
        for _ in range(random.randint(1, 4)):  # Generate 1-2 objects per spawn interval
            obj_type = "bomb" if random.random() <= 0.2 else random.choice(["watermelon", "pineapple", "banana", "apple"])
            x = random.randint(40, w - 40)
            vx, vy = random.uniform(-2, 2), random.uniform(-12, -10)
            angle, rotation_speed = random.randint(0, 360), random.uniform(2, 6)
            img = eval(f"{obj_type}_img")
            obj = {"type": obj_type, "x": x, "y": h - 40, "vx": vx, "vy": vy, "radius": 40, "image": img, "angle": angle, "rotation_speed": rotation_speed}
            objects.append(obj)

    for obj in objects[:]:
        obj["vy"] += gravity  # Apply gravity
        obj["x"] += obj["vx"]
        obj["y"] += obj["vy"]

        if obj["y"] - obj["radius"] > h:
            objects.remove(obj)
            continue

        # Reverse horizontal velocity if the object hits the left or right frame edge
        if obj["x"] - obj["radius"] < 0 or obj["x"] + obj["radius"] > w:
            obj["vx"] = -obj["vx"]
            # Ensure the object stays within the frame
            obj["x"] = max(obj["radius"], min(w - obj["radius"], obj["x"]))


        obj["angle"] += obj["rotation_speed"]
        obj["angle"] %= 360

        if obj["image"] is not None:
            img = obj["image"]
            img_h, img_w = img.shape[:2]
            scale_factor = obj["radius"] * 2 / max(img_w, img_h)
            img_resized = cv2.resize(img, (int(img_w * scale_factor), int(img_h * scale_factor)))
            center = (img_resized.shape[1] // 2, img_resized.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, obj["angle"], 1.0)
            img_rotated = cv2.warpAffine(img_resized, rotation_matrix, (img_resized.shape[1], img_resized.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            x1, y1 = int(obj["x"] - img_rotated.shape[1] / 2), int(obj["y"] - img_rotated.shape[0] / 2)
            x2, y2 = x1 + img_rotated.shape[1], y1 + img_rotated.shape[0]

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            alpha_mask = img_rotated[:, :, 3] / 255.0
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha_mask[:, :, None]) + img_rotated[:, :, :3] * alpha_mask[:, :, None]

    for splash in splashes[:]:
        splash_img = splash["image"]
        splash_x, splash_y = splash["x"], splash["y"]
        splash_ttl = splash["ttl"]

        if splash_img is not None:
            # Scale factor to reduce the size of the splash (e.g., 0.7 for 70% size)
            scale_factor = 0.7

            # Get the original splash dimensions
            splash_h, splash_w = splash_img.shape[:2]

            # Calculate new dimensions
            new_h, new_w = int(splash_h * scale_factor), int(splash_w * scale_factor)

            # Resize the splash image
            splash_resized = cv2.resize(splash_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Recalculate positions based on the resized dimensions
            x1, y1 = int(splash_x - new_w / 2), int(splash_y - new_h / 2)
            x2, y2 = x1 + new_w, y1 + new_h


            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            alpha_mask = splash_resized[:, :, 3] / 255.0
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha_mask[:, :, None]) + splash_resized[:, :, :3] * alpha_mask[:, :, None]


        splash["ttl"] -= 1
        if splash["ttl"] <= 0:
            splashes.remove(splash)

    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if game_started:
        remaining_time = max_game_duration - int(time.time() - game_timer)
        cv2.putText(frame, f"Time: {remaining_time}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Fruit Ninja Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()