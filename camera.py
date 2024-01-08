# camera.py
import cv2
from detect_faces import detect_faces


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        self.last_faces = None

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        processed_frame, faces = detect_faces(frame)
        self.last_frame = processed_frame
        self.last_faces = faces

        _, jpeg = cv2.imencode(".jpg", processed_frame)
        return jpeg.tobytes()

    def get_last_frame_and_face_coords(self):
        return self.last_frame, self.last_faces

    def __del__(self):
        self.cap.release()
