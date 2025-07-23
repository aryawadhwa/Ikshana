import os
import sys
import cv2
import time
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Add src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.picamera_utils import is_raspberry_camera, get_picamera

CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IS_RASPI_CAMERA = is_raspberry_camera()
fps = 0
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the cascade
face_cascade = cv2.CascadeClassifier(os.path.join(base_dir, 'haarcascade_frontalface_default.xml'))

# Load pre-trained currency recognition model
currency_model = tf.keras.models.load_model('currency_model.h5')

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def visualize_fps(image, fps: int):
    if len(np.shape(image)) < 3:
        text_color = (255, 255, 255)  # white
    else:
        text_color = (0, 255, 0)  # green
    row_size = 20  # pixels
    left_margin = 24  # pixels
    font_size = 1
    font_thickness = 1

    # Draw the FPS counter
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    return image

def detect_faces_and_speak(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        text_to_speech("Face detected!")
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

def recognize_currency(frame):
    # Preprocess the image for the model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    
    prediction = currency_model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Assuming class 0 = 10 INR, class 1 = 20 INR, class 2 = 50 INR, etc.
    if predicted_class == 0:
        text_to_speech("Detected 10 INR")
    elif predicted_class == 1:
        text_to_speech("Detected 20 INR")
    elif predicted_class == 2:
        text_to_speech("Detected 50 INR")
    
    return frame

# To capture video from webcam.
if IS_RASPI_CAMERA:
    cap = get_picamera(IMAGE_WIDTH, IMAGE_HEIGHT)
    cap.start()
else:
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
    cap.set(3, IMAGE_WIDTH)
    cap.set(4, IMAGE_HEIGHT)

while True:
    start_time = time.time()
    if IS_RASPI_CAMERA:
        frame = cap.capture_array()
    else:
        _, frame = cap.read()
    
    # Detect faces and currency
    frame = detect_faces_and_speak(frame)
    frame = recognize_currency(frame)
    
    cv2.imshow('Live Feed', visualize_fps(frame, fps))
    
    end_time = time.time()
    seconds = end_time - start_time
    fps = 1.0 / seconds
    print("Estimated fps:{0:0.1f}".format(fps))
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.close() if IS_RASPI_CAMERA else cap.release()
cv2.destroyAllWindows()