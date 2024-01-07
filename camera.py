import cv2

class Camera:
    def __init__(self):
        # Use the correct camera index (usually 0)
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Encode the frame in JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def __del__(self):
        self.cap.release()
