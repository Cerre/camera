import os
from sklearn.model_selection import train_test_split

from knn_classifier_training import load_embeddings

embedding_size = 512  # Replace with the actual size of your individual embeddings
script_dir = os.path.dirname(__file__)

# Construct the path to the embeddings folder
embeddings_path = os.path.join(script_dir, '..', "embeddings")
X, y = load_embeddings(embeddings_path, embedding_size)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)  # Assuming X, y are your data and labels

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neighbors_settings = range(1, 26)  # Example range, adjust based on your dataset
accuracy_scores = []

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

# Find the best performing number of neighbors
best_n_neighbors = neighbors_settings[accuracy_scores.index(max(accuracy_scores))]
print("Best number of neighbors:", best_n_neighbors)


import matplotlib.pyplot as plt

plt.plot(neighbors_settings, accuracy_scores)
plt.xlabel("Number of Neighbors (n_neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different Numbers of Neighbors")
plt.show()
