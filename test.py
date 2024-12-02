import os

# Specify the folder path
folder_path = 'static/facerecog/train/1'

# Check if the folder exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    print("The folder 'kamu' exists.")
else:
    print("The folder 'kamu' does not exist.")
