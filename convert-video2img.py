import cv2
import os
import uuid

# train
video_path = 'tp19112023am.MP4'
out_folder = os.path.join('.', 'rawdata-train')
# val
#video_path = 'tp18112023am.MP4'
#out_folder = os.path.join('.', 'rawdata-validation')

os.makedirs(out_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("error opening the video")
    exit()

# UUID functiomn
def generate_uuid():
    return str(uuid.uuid4())[:8]

 # Generate a UUID for the image
video_uuid = generate_uuid()

# Set the frame interval (e.g. skip every 25 frames )
# 10min video of 15000 total frame -> 25 frame/s
frame_interval = 5*25  # 5 *25 training #validition :25 only 25s

# initialize frame counter
frame_count = 0

while True:
    # Read a frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break
    
    if frame_count % frame_interval == 0:

        # Save the frame as an image
        frame_nm = f'{video_uuid}_frame_{frame_count}.png'

        cv2.imwrite(os.path.join(out_folder, frame_nm), frame)

        print(f'saved {frame_nm}')

    # Increament frame counter
    frame_count += 1
    
cap.release()
print(f'video contains a total of {frame_count} frames')
