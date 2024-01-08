import os
import torch
from sklearn.cluster import KMeans


def load_embeddings(file_path):
    return torch.load(file_path).numpy()


def perform_clustering(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.cluster_centers_


save_dir = "embeddings"
cluster_dir = "clusters"
os.makedirs(cluster_dir, exist_ok=True)


# Process each embedding file
for embedding_file in os.listdir(save_dir):
    if embedding_file.endswith(".pt"):
        embeddings = load_embeddings(os.path.join(save_dir, embedding_file))
        cluster_centers = perform_clustering(embeddings, n_clusters=5)

        # Save cluster centers
        class_name = embedding_file.replace("_embeddings.pt", "")
        cluster_save_path = os.path.join(cluster_dir, f"{class_name}_clusters.pt")
        torch.save(torch.tensor(cluster_centers), cluster_save_path)
        print(f"Cluster centers for {class_name} saved.")
