def test_all_cameras():
    """Test all available camera devices and return the first working one."""
    # List of potential camera devices to test
    video_devices = [0, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 31]

```
print("\nTesting available camera devices...")
for device in video_devices:
    print(f"\nTrying camera device: /dev/video{device}")
    cap = cv2.VideoCapture(device)
    
    if not cap.isOpened():
        print(f"Could not open device {device}")
        cap.release()
        continue
        
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame from device {device}")
        cap.release()
        continue
        
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Success! Camera {device} works:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Show test frame
    cv2.imshow(f'Camera Test - Device {device}', frame)
    cv2.waitKey(1000)  # Show for 1 second
    cv2.destroyAllWindows()
    
    cap.release()
    return device

return None

```

def initialize_camera(device_id=None):
    """Initialize camera with specific device ID or find first working camera."""
    if device_id is None:
        device_id = test_all_cameras()
        if device_id is None:
            raise RuntimeError("No working camera found!")

```
cap = cv2.VideoCapture(device_id)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open camera device {device_id}")

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

return cap

```

# Update the recognize_text function to use the new camera initialization

def recognize_text():
    """Capture an image and perform text recognition."""
    text_to_speech("Capturing image for text recognition.")
    print("Opening camera for text recognition...")

```
try:
    cap = initialize_camera()
except Exception as e:
    print(f"Camera initialization error: {str(e)}")
    text_to_speech("Camera error occurred.")
    return

ret, frame = cap.read()
if ret:
    print("Captured frame for text recognition")
    cv2.imwrite("text_image.jpg", frame)
    cv2.imshow('Captured Frame', frame)
    cv2.waitKey(1000)  # Show frame for 1 second
    
    text = pytesseract.image_to_string(Image.open("text_image.jpg"))
    print(f"Detected text: {text}")
    text_to_speech(f"Detected text: {text}")
else:
    print("Failed to capture frame for text recognition")
    text_to_speech("Failed to capture image.")

cap.release()
cv2.destroyAllWindows()

```

# Update the detect_objects function to use the new camera initialization

def detect_objects():
    """Perform object detection using YOLO."""
    text_to_speech("Starting object detection.")
    print("Initializing object detection...")

```
# Check if YOLO files exist
if not all(os.path.exists(f) for f in ['yolov4-tiny.weights', 'yolov4-tiny.cfg', 'coco.names']):
    print("Error: YOLO files not found")
    text_to_speech("Object detection files not found.")
    return

try:
    net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split("\n")
    
    print("Opening camera for object detection...")
    cap = initialize_camera()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)

        # Rest of the object detection code remains the same...
        
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error in object detection: {str(e)}")
    text_to_speech("An error occurred during object detection.")

```

# Update main() to include camera testing

def main():
    """Main function to listen for voice commands and act accordingly."""
    print("\n=== Running System Diagnostics ===")
    try:
        working_camera = test_all_cameras()
        if working_camera is not None:
            print(f"\nFound working camera: /dev/video{working_camera}")
        else:
            print("\nWARNING: No working camera found!")
    except Exception as e:
        print(f"\nCamera test error: {str(e)}")

```
# Rest of the main() function remains the same...

```