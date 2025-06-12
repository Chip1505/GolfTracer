import cv2
from ultralytics import YOLO
from collections import deque

# Load model
model = YOLO('runs/detect/train4/weights/best.pt')

# Use deque to store ball center points (up to 1000)
flight_path = deque(maxlen=1000)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ball
    results = model.predict(frame, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes:
            # Get center of bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Save to flight path
            flight_path.append((cx, cy))

            # Optional: draw detection box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Draw the flight path
    for i in range(1, len(flight_path)):
        if flight_path[i - 1] and flight_path[i]:
            cv2.line(frame, flight_path[i - 1], flight_path[i], (255, 0, 0), thickness=8)

    # Display frame
    cv2.imshow("Golf Ball Tracker with Tracer", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
