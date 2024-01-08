# camera.py
import cv2
from detect_faces import detect_faces


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        self.last_face_coords = None

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        processed_frame, face_coords = detect_faces(frame)
        self.last_frame = processed_frame
        self.last_face_coords = face_coords

        _, jpeg = cv2.imencode(".jpg", processed_frame)
        return jpeg.tobytes()

    def get_last_frame_and_face_coords(self):
        return self.last_frame, self.last_face_coords

    def __del__(self):
        self.cap.release()
