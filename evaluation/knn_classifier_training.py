import joblib
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_embeddings(embeddings_folder, embedding_size):
    embeddings, labels = [], []
    for file in os.listdir(embeddings_folder):
        label = file.replace("_embeddings.pt", "")
        file_path = os.path.join(embeddings_folder, file)
        if file_path.endswith(".pt"):
            embedding_array = torch.load(file_path).numpy()

            # Calculate the number of embeddings in the file
            num_embeddings = embedding_array.size // embedding_size

            # Reshape and append embeddings
            reshaped_embeddings = embedding_array.reshape(
                num_embeddings, embedding_size
            )
            for embedding in reshaped_embeddings:
                embeddings.append(embedding)
                labels.append(label)

    return np.array(embeddings), labels


def main():
    script_dir = os.path.dirname(__file__)

    # Construct the path to the embeddings folder
    embeddings_path = os.path.join(script_dir, '..', "embeddings")
    embedding_size = 512  # Replace with the actual size of your individual embeddings
    X, y = load_embeddings(embeddings_path, embedding_size)

    # Split the data - 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, "knn_model.pkl")

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    tsne = TSNE(n_components=2, random_state=0)
    X_reduced = tsne.fit_transform(X_test)

    # Create a color map for your test labels
    unique_labels_test = np.unique(y_test)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels_test)))
    label_to_color = dict(zip(unique_labels_test, colors))

    # Plot each class with a separate color
    plt.figure(figsize=(12, 8))
    for label, color in label_to_color.items():
        indices = [i for i, l in enumerate(y_test) if l == label]
        plt.scatter(
            X_reduced[indices, 0],
            X_reduced[indices, 1],
            color=color,
            label=label,
            alpha=0.7,
        )

    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.title("t-SNE Visualization of Face Embeddings with KNN Classification")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


