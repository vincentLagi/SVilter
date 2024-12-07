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