import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib



import torch

def load_embeddings_and_labels(base_dir):
    embeddings = []
    labels = []
    class_labels = os.listdir(base_dir)

    for label in class_labels:
        emb_dir = os.path.join(base_dir, label)
        for emb_file in os.listdir(emb_dir):
            emb_path = os.path.join(emb_dir, emb_file)
            # Load the embedding using PyTorch
            embedding = torch.load(emb_path)
            embeddings.append(embedding.numpy())  # Convert to numpy array if required
            labels.append(label)

    return np.array(embeddings), np.array(labels)




def load_images_and_labels(base_dir):
    data = []
    labels = []
    class_labels = os.listdir(base_dir)

    for label in class_labels:
        img_dir = os.path.join(base_dir, label)
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path)
            img = img.resize((64, 64))  # Resize for consistency
            img_array = np.array(img).flatten()  # Convert to 1D array
            
            # Normalize the image data to 0-1
            img_array = img_array / 255.0
            
            data.append(img_array)
            labels.append(label)

    return np.array(data), np.array(labels)





def evaluate_models_with_cv(models, X, y, cv=5):
    model_scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv)
        model_scores[name] = np.mean(cv_scores)
        print(f"{name}: Average CV Score = {model_scores[name]}")
    return model_scores



def run_models(X,y, data):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }


    model_scores = evaluate_models_with_cv(models, X, y)
    best_model_name = max(model_scores, key=model_scores.get)
    print(f"Best Model using {data}: {best_model_name}")

    best_model = models[best_model_name]
    best_model.fit(X, y)
    joblib.dump(best_model, f'{best_model_name}_best_model_{data}.pkl')
    print(f"Saved {best_model_name} model to file.")




def main():
    
    # X, y = load_images_and_labels('data')  # 'data' is your base directory
    # run_models(X, y, data="images")
    X, y = load_embeddings_and_labels('embeddings')
    run_models(X, y, data="embeddings")


if __name__=='__main__':
    main()