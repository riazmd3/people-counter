import cv2
from ultralytics import YOLO
from tracker import Tracker  # Ensure 'tracker' is your tracking algorithm module
import datetime

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Function to capture mouse events (not used in this example)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture("HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4")

# Read class list for YOLO
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize tracking variables
count = 0
tracker = Tracker()
entry_count = 0
exit_count = 0

# Lines for up/down detection
cy1, cy2 = 194, 220

while cap.isOpened():    
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Skip frames for faster processing if needed
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO detection
    results = model.predict(frame)
    detections = results[0].boxes.data  # Extract boxes
    detected_persons = []

    for row in detections:
        x1, y1, x2, y2, confidence, class_id = map(int, row[:6])
        class_name = class_list[class_id]
        
        if class_name == 'person':
            detected_persons.append([x1, y1, x2, y2])

    # Track detected persons
    bbox_id = tracker.update(detected_persons)

    # Draw tracking info with labels and update entry/exit count
    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Draw the bounding box and label
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        # Check if the person crossed the lines to enter or exit
        if cy1 - 10 < cy < cy1 + 10:  # Enter line
            entry_count += 1
        elif cy2 - 10 < cy < cy2 + 10:  # Exit line
            exit_count += 1

    # Draw lines
    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

    # Display the frame with counts
    cv2.putText(frame, f"Entries: {entry_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("RGB", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
