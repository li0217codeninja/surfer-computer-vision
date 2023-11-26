import cv2
import numpy as np
import math
from ultralytics import YOLO


# load yolo
#model = YOLO("yolo-Weights/yolov8n.pt")
model = YOLO("yolo-Weights/last.pt")

classNames = ["surfing", "watching"]

object_detector = cv2.createBackgroundSubtractorKNN()


cap = cv2.VideoCapture('tp231120230934a.MP4')

if not cap.isOpened():
    print("error opening the video")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    mask = object_detector.apply(frame)

    results = model(frame, stream=True)
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1,  x2, y2 = int(x1), int(y1), int(x2), int(y2)

             # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

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

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(25) == ord('q'):
        break

# todo: what does this do?
cap.release()
cv2.destroyAllWindows()