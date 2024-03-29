import cv2
from face_detection.detect_faces import detect_faces


class Camera:
    def __init__(self, classifier):
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        self.last_faces = None
        self.classifier = classifier

    def get_frame(self, return_faces: bool = True):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.last_frame = frame
        processed_frame = frame

        if return_faces:
            processed_frame, faces = detect_faces(frame, self.classifier)
            self.last_frame = processed_frame
            self.last_faces = faces
        

        _, jpeg = cv2.imencode(".jpg", processed_frame)
        return jpeg.tobytes()

    def get_last_frame_and_face_coords(self):
        return self.last_frame, self.last_faces

    def __del__(self):
        self.cap.release()
