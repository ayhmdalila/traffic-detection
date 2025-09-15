import cv2
import torch
import time
from ultralytics import YOLO

# Load YOLO model with GPU acceleration
model = YOLO("yolo11n.pt").to("cuda")

# Open video capture
cap = cv2.VideoCapture("trafic_video.mkv")

# Get video properties
width = 640
height = 360
fps_input = cap.get(cv2.CAP_PROP_FPS)

# Define VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
out = cv2.VideoWriter('output_video.mp4', fourcc, fps_input, (width, height))

fps = 0
frame_count = 0
start_time = time.time()
frame_skip = 1  # Process every 2nd frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # if frame_count % frame_skip != 0:
    #     continue  # Skip frames for efficiency

    frame = cv2.resize(frame, (width, height))
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Run YOLOv8 object detection
    results = model(frame, conf=0.4, iou=0.5, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()