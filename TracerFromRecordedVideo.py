import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("runs/detect/train4/weights/best.pt")
conf_threshold = 0.15

# Open video
cap = cv2.VideoCapture("golf_swing4.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_kalman_tracer_fixed.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Kalman filter setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Tracker state
prev_centers = []
kalman_initialized = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=conf_threshold, verbose=False)
    detections = results[0].boxes
    detected = False

    if detections is not None:
        for box in detections:
            conf = float(box.conf)
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Use first valid detection to initialize Kalman
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                if not kalman_initialized:
                    # Set initial state
                    kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    kalman_initialized = True

                kalman.correct(measurement)
                prev_centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                detected = True
                break

    if not detected and kalman_initialized:
        prediction = kalman.predict()
        px, py = int(prediction[0]), int(prediction[1])
        prev_centers.append((px, py))
        cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

    # Draw red tracer line
    for i in range(1, len(prev_centers)):
        cv2.line(frame, prev_centers[i - 1], prev_centers[i], (0, 0, 255), 2)

    # Show and save
    cv2.imshow("Kalman Tracer", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

