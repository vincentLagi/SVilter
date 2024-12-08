    # <!-- <div id="gameContainer">
    #     <img src="{{ url_for('filter_face_feed', width=1928, height=640) }}" alt="Cat Ears Feed" id="gameImage">
    # </div> -->


# choice = 0
# frame_rate = 30  # Target frame rate (fps)
# last_time = time.time()

def generate_filter_face_frames(width, height):
    global choice, last_time, frame_rate, overlay_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        
        # Control the frame rate (process every Nth frame)
        if time.time() - last_time < 1 / frame_rate:
            continue
        last_time = time.time()

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

# @app.route('/update_frame_size', methods=['POST'])
# def update_frame_size():
#     global image_width, image_height
#     data = request.get_json()
#     image_width = data['width']
#     image_height = data['height']
#     return jsonify({"status": "success"})

    # <!-- <div id="main">
    #     <img src="{{ url_for('game_video_feed') }}" alt="Video Feed" id="gameImage">
    # </div> -->

# @app.route('/game_video_feed')
# def game_video_feed():

#     game_state.out_off_game = False
#     return Response(game_generate_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# <!-- <img src="{{ url_for('register_face_video_feed') }}" width="100%" height="400px" id="face_video_feed"> -->

# @app.route('/register_face_video_feed')
# def register_face_video_feed():
#     return Response(register_face_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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