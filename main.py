import cv2
import numpy as np
import math
from ultralytics import YOLO


# load yolo
#model = YOLO("yolo-Weights/yolov8n.pt")
model = YOLO("runs/detect/train3/weights/best.pt")

classNames = ["surfing"]

# Object detection from stable camera
# Use background Substractor for idle-surfer detection (avoid labeling)
object_detector = cv2.createBackgroundSubtractorKNN()

cap = cv2.VideoCapture('tp18112023am.MP4')  #test data: tp231120230934a.MP4

if not cap.isOpened():
    print("error opening the video")

while True:
    ret, frame = cap.read()
    height, width,_ = frame.shape
    # height=720,  #width=1280 (double check)
    roi_surf = frame[round(height/2)-200: round(height/2)+200,round(width/2) - 200: round(width/2)+200]

    # Extrac Region of interest

    if not ret:
        break
    
    # Object detection 
    mask = object_detector.apply(roi_surf)
    # threshold for tracking
    #_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if (area > 5) & (area < 50): #todo: reduce false na
            # > 100 tracks waves 
            #cv2.drawContours(roi_surf, [cnt], -1, (0, 255,0), 1)
            x, y, w, h = cv2.boundingRect(cnt) 
            cv2.rectangle(roi_surf, (x,y), (x+w, y+h), (0,255,0),1)

    results = model(frame, stream=True)
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

             # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
        surfer_count = len(boxes)
        cv2.putText(frame, f'Surfers: {surfer_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 2)

    cv2.imshow('ROI', roi_surf)
    cv2.imshow('Frame', frame)
    #cv2.imshow('Mask', mask)

     # check for the q key to quit video (todo: not working)
    if cv2.waitKey(25) & 0xFF == ord('s'):
        break

# todo: what does this do?
cap.release()
cv2.destroyAllWindows()