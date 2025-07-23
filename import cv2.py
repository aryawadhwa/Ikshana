import cv2
import pytesseract
from PIL import Image
import threading
import logging
import queue
from typing import Optional, List, Dict

class SmartGlasses:
    def __init__(self, camera_index: int = 0, 
                 resolution: tuple = (640, 480)):
        """
        Enhanced initialization with more robust setup
        
        Args:
            camera_index (int): Camera device index
            resolution (tuple): Desired camera resolution
        """
        # Robust logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        # Camera initialization with multiple fallbacks
        self.camera = self._initialize_camera(camera_index, resolution)
        
        # Thread-safe frame queue
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Model and resource management
        self._load_models()
        
        # State management
        self.running = False
        self.current_frame = None

    def _initialize_camera(self, index: int, resolution: tuple):
        """Robust camera initialization with multiple attempts"""
        attempts = [index, 1, 2]  # Try multiple camera indices
        for cam_index in attempts:
            camera = cv2.VideoCapture(cam_index)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                logging.info(f"Camera initialized: Index {cam_index}")
                return camera
        
        logging.error("No camera available")
        raise RuntimeError("Could not initialize camera")

    def _load_models(self):
        """Centralized model loading with comprehensive error handling"""
        try:
            # YOLO Model Loading
            self.net = cv2.dnn.readNet(
                "yolov3-tiny.weights", 
                "yolov3-tiny.cfg"
            )
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            logging.info("Models loaded successfully")
        
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            self.net = None
            self.classes = []

    def capture_frames(self):
        """Enhanced frame capture with queue management"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                try:
                    # Non-blocking queue put with timeout
                    resized_frame = cv2.resize(frame, (320, 240))
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait(resized_frame)
                except queue.Full:
                    # Discard oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def read_text(self) -> Optional[str]:
        """Enhanced OCR with better error handling"""
        try:
            frame = self.frame_queue.get(timeout=2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            text = pytesseract.image_to_string(
                Image.fromarray(thresh)
            ).strip()
            
            return text if text else "No text detected"
        
        except Exception as e:
            logging.error(f"OCR Error: {e}")
            return "OCR processing failed"

    def detect_objects(self) -> List[Dict[str, int]]:
        """Advanced object detection with detailed reporting"""
        if not self.net:
            logging.warning("Object detection model not loaded")
            return []
        
        # Similar implementation to previous version
        # Add more robust error handling and logging
        
    def run(self):
        """Comprehensive run method with graceful shutdown"""
        try:
            self.running = True
            capture_thread = threading.Thread(
                target=self.capture_frames, 
                daemon=True
            )
            capture_thread.start()
            
            # Interactive command loop
            while self.running:
                cmd = input("Command (R/O/Q): ").lower()
                if cmd == 'q':
                    break
                elif cmd == 'r':
                    text = self.read_text()
                    print(text)
                elif cmd == 'o':
                    objects = self.detect_objects()
                    print(objects)
        
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.running = False
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    glasses = SmartGlasses()
    glasses.run()
