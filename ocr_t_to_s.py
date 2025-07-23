import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO
import pyttsx3
import csv
from datetime import datetime
from PyQt5 import QtCore, QtWidgets, QtGui
from PIL import Image
import pytesseract
from gtts import gTTS
import os
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

# Initialize YOLO model
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    logging.error(f"YOLO Model Loading Error: {e}")
    sys.exit(1)

# Multi-Engine Text-to-Speech
class SpeechEngine:
    def __init__(self):
        self.pyttsx_engine = pyttsx3.init()
        self.gtts_engine = None

    def speak_pyttsx(self, text):
        try:
            self.pyttsx_engine.say(text)
            self.pyttsx_engine.runAndWait()
        except Exception as e:
            logging.warning(f"PyTTSx Speech Error: {e}")
            self.speak_gtts(text)

    def speak_gtts(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("speech_output.mp3")
            os.system("start speech_output.mp3")  # Platform-specific
        except Exception as e:
            logging.error(f"GTTS Speech Error: {e}")

    def speak(self, text):
        # Prioritize pyttsx3, fallback to gTTS
        self.speak_pyttsx(text)

# Global Speech Engine
speech_engine = SpeechEngine()

# Pytesseract Configuration
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# Detection Parameters
DETECTION_CONFIG = {
    'threshold': 0.5,
    'announcement_interval': 5
}

class EnhancedRecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0):
        super().__init__()
        self.camera = self._initialize_camera(camera_port)
        self.timer = QtCore.QBasicTimer()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_announcement_time = 0

    def _initialize_camera(self, port):
        camera = cv2.VideoCapture(port)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return camera

    def start_recording(self):
        self.timer.start(33, self)  # ~30 FPS

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return
        
        ret, frame = self.camera.read()
        if ret:
            self.image_data.emit(frame)
            self.executor.submit(self.process_frame, frame)

    def process_frame(self, frame):
        try:
            # Object Detection
            results = model(frame, show=False)
            detections = results[0].boxes
            object_count = {}

            for box in detections:
                confidence = box.conf[0]
                if confidence > DETECTION_CONFIG['threshold']:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    object_count[class_name] = object_count.get(class_name, 0) + 1

            # Periodic Announcements
            current_time = time.time()
            if (current_time - self.last_announcement_time > 
                DETECTION_CONFIG['announcement_interval'] and object_count):
                
                count_text = ", ".join([f"{obj}: {count}" for obj, count in object_count.items()])
                speech_text = f"I detected {count_text}"
                
                speech_engine.speak(speech_text)
                self.last_announcement_time = current_time

            # OCR Processing
            img = Image.fromarray(frame)
            text = pytesseract.image_to_string(img, config=tessdata_dir_config)
            
            if text.strip():
                speech_engine.speak(text)

        except Exception as e:
            logging.error(f"Frame Processing Error: {e}")

class VideoDisplayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image = QtGui.QImage()
        self.setFixedSize(640, 480)

    def image_data_slot(self, image_data):
        self.image = self._convert_to_qimage(image_data)
        self.update()

    def _convert_to_qimage(self, image):
        height, width, _ = image.shape
        return QtGui.QImage(
            image.data, 
            width, 
            height, 
            QtGui.QImage.Format_RGB888
        ).rgbSwapped()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)

class MainApplicationWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.video_widget = VideoDisplayWidget()
        self.record_video = EnhancedRecordVideo()

        self.record_video.image_data.connect(self.video_widget.image_data_slot)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_widget)

        start_button = QtWidgets.QPushButton('Start Detection')
        start_button.clicked.connect(self.record_video.start_recording)
        layout.addWidget(start_button)

        self.setLayout(layout)
        self.setWindowTitle('Enhanced Object Detection')

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplicationWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
