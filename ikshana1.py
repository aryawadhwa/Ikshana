import cv2
import time
from ultralytics import YOLO
import pyttsx3
import csv
from datetime import datetime

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to provide voice feedback
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Parameters
last_announcement_time = 0
announcement_interval = 5  # Time interval for announcements in seconds
detection_threshold = 0.5  # Confidence threshold
log_file = "detections_log.csv"

# Initialize detection log
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Object", "Confidence", "Bounding Box"])

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

speak("Webcam initialized. Starting object detection.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Unable to capture the frame.")
            break

        # Perform object detection
        results = model(frame, show=False)
        detections = results[0].boxes
        detected_objects = []
        object_count = {}

        for box in detections:
            confidence = box.conf[0]
            if confidence > detection_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_objects.append(class_name)

                # Count objects
                object_count[class_name] = object_count.get(class_name, 0) + 1

                # Draw bounding box and label
                (x1, y1, x2, y2) = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Log detection
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([datetime.now(), class_name, f"{confidence:.2f}", (x1, y1, x2, y2)])

        # Display object count
        count_text = ", ".join([f"{obj}: {count}" for obj, count in object_count.items()])
        cv2.putText(frame, f"Detected: {count_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Voice announcements every interval
        current_time = time.time()
        if current_time - last_announcement_time > announcement_interval and object_count:
            speak(f"I see {count_text}")
            last_announcement_time = current_time

        # Display frame with updated title
        cv2.imshow("ikshanaOB", frame)

        # Break the loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Object detection stopped.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    speak("Object detection stopped.")