import cv2
from ultralytics import YOLO
import numpy as np

# Load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

tracer_path = []
tracing = False  # Will become True after pressing 's'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model.predict(source=frame, conf=0.5, verbose=False)[0]

    for box in results.boxes:
        # Get center of bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # If tracing is on, record the center point
        if tracing:
            tracer_path.append((cx, cy))

    # Draw the tracer path if any
    if tracer_path:
        cv2.polylines(frame, [np.array(tracer_path, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=8)

    # Display instructions
    cv2.putText(frame, "Press 's' to Start, 'r' to Reset, 'q' to Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Golf Ball Tracker", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        tracing = True
        print("Started tracing.")
    elif key == ord('r'):
        tracer_path = []
        tracing = False
        print("Reset tracing.")

cap.release()
cv2.destroyAllWindows()
