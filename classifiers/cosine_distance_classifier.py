from classifiers.classifier import Classifier
import os
from scipy.spatial.distance import cosine
import torch
from utils.names import names


class CosineSimilarityClassifier(Classifier):
    def __init__(self, threshold=0.5):
        self.embeddings_dict = self.load_embeddings_from_directory("clusters")
        self.threshold = threshold

    def classify(self, embedding):
        max_similarity = 0
        label = "Unknown"

        for person_label, embeddings in self.embeddings_dict.items():
            for embedding in embeddings:
                similarity = self.cosine_similarity(self.new_embedding, embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    label = person_label

        if max_similarity < self.threshold:
            label = "Unknown"

        return label

    def cosine_similarity(self, embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    def load_embeddings_from_directory(self, directory):
        embeddings = {}
        for file in os.listdir(directory):
            if file.endswith(".pt"):
                label = file.replace("_clusters.pt", "")
                embeddings[names[label]] = torch.load(
                    os.path.join(directory, file)
                ).numpy()
        return embeddings
