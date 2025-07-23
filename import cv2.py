import cv2
import pyttsx3
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained MobileNet-SSD model
prototxt_path = "models/deploy.prototxt"
model_path = "models/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# List of class labels MobileNet-SSD is trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Open the laptop's webcam
cap = cv2.VideoCapture(0)

# Function to speak detected object
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Main loop
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []

    # Loop through detections
    for i in range(detections.shape[2]):
        # Confidence of prediction
        confidence = detections[0, 0, i, 2]

        # Filter detections by confidence threshold
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            detected_objects.append(label)

            # Get bounding box dimensions
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Provide voice feedback
    if detected_objects:
        speak(f"I see {', '.join(detected_objects)}")

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()