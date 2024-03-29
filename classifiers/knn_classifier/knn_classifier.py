from collections import Counter
import joblib
from classifiers.classifier import Classifier
from face_detection.detect_faces import get_embedding
from utils.names import names
import torch
from facenet_pytorch import InceptionResnetV1


class KNNClassifier(Classifier):
    def __init__(self, threshold=0.7, num_neighbors=5):
        self.knn_model = joblib.load("models/knn_model.pkl")
        self.threshold = threshold
        self.num_neighbors = num_neighbors
        self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval()
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda")


    def classify(self, face):
        self.new_embedding = get_embedding(face, self.embedding_model)
        new_embedding_reshaped = self.new_embedding.reshape(1, -1)
        distances, neighbors_indices = self.knn_model.kneighbors(
            new_embedding_reshaped,
            n_neighbors=int(self.num_neighbors),
            return_distance=True,
        )
        valid_neighbors = [
            index
            for index, distance in zip(neighbors_indices[0], distances[0])
            if distance < self.threshold
        ]
        # Check if there are valid neighbors
        if not valid_neighbors:
            return "Unknown"
        valid_neighbors = self.knn_model._y[valid_neighbors]
        # Retrieve labels for valid neighbors
        neighbor_labels = [
            self.knn_model.classes_[index]
            for index in valid_neighbors
            if index < len(self.knn_model.classes_)
        ]
        # Determine the majority label among the valid neighbors
        label_counts = Counter(neighbor_labels)
        majority_label, _ = label_counts.most_common(1)[0]

        return names[majority_label]
