# camera.py
import cv2
from detect_faces import detect_faces

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Process the frame for face detection
        frame_with_boxes = detect_faces(frame)

        # Encode the frame in JPEG format
        _, jpeg = cv2.imencode('.jpg', frame_with_boxes)
        # _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def __del__(self):
        self.cap.release()
