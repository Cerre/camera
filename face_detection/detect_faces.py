from PIL import Image
import cv2
import torch
from torchvision import transforms
from face_detection.fast_mtcnn import FastMTCNN

# If required, create a face detection pipeline using MTCNN:
device = "cuda" if torch.cuda.is_available() else "cpu"
fast_mtcnn = FastMTCNN(
    stride=4, resize=1, margin=14, factor=0.6, keep_all=True, device="cpu"
)


def load_embeddings_from_file(file_path):
    return torch.load(file_path).numpy()


embeddings_exist = False
try:
    your_embeddings = load_embeddings_from_file("clusters/0_clusters.pt")
    friend_embeddings = load_embeddings_from_file("clusters/1_clusters.pt")
    embeddings_exist = True
except:
    print("No embeddings yet")


def detect_faces(image, classifier):
    boxes, _, _ = fast_mtcnn(image)
    detected_faces = []

    if boxes is not None:
        for box in boxes:
            (x0, y0, x1, y1) = box
            face = image[int(y0) : int(y1), int(x0) : int(x1)]

            if face is not None:
                label = classifier.classify(face)
                detected_faces.append((box, label))

            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(
                image,
                label,
                (int(x0), int(y0) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image, detected_faces


def get_embedding(face, resnet):
    # Preprocess the face image
    face = Image.fromarray(face)  # Convert to PIL image
    face_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    face_tensor = face_transform(face).unsqueeze(0)

    # Ensure tensor is on the same device as the model
    face_tensor = face_tensor.to(next(resnet.parameters()).device)

    # Get the embedding
    with torch.no_grad():
        face_embedding = resnet(face_tensor)
        return face_embedding.squeeze().cpu().numpy()