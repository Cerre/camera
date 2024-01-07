from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from fast_mtcnn import FastMTCNN

# If required, create a face detection pipeline using MTCNN:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.6,
    keep_all=True,
    device="cpu"
)
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def classify_face(face):
    img_embedding = resnet(img_cropped.unsqueeze(0))
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    return img_probs


def detect_faces(image):
    # Detect faces in the image
    # faces = mtcnn.detect_faces(image)
    boxes, probs, points = fast_mtcnn(image)
    # print(f"len faces: {len(faces)}")
    # print(faces)

    # Draw bounding boxes around detected faces
    if boxes is not None:
        for (box, point) in zip(boxes, points):
            # print(type(face))
            (x0, y0, x1, y1) = box
            start_point = (int(x0), int(y0))  # Convert to integer and create a tuple for the start point 
            end_point = (int(x1), int(y1))  # Calculate end point and convert to integer
            color = (255, 0, 0)  # Blue color in BGR
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image

