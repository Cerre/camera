from PIL import Image
import cv2
from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(image):
    # Detect faces in the image
    faces = detector.detect_faces(image)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        start_point = (x, y)
        end_point = (x + width, y + height)
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image

