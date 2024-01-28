import os
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Initialize the InceptionResnetV1 model for embedding generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(file_path):
    img = Image.open(file_path)
    img_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),  # Resize the image to the size expected by the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = resnet(img_tensor)
        return img_embedding.squeeze().cpu().numpy()

def process_directory(directory, class_name, save_dir):
    class_save_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_save_dir, exist_ok=True)

    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        embedding = get_embedding(img_path)
        if embedding is not None:
            # Save each embedding to a separate file
            embedding_filename = os.path.splitext(img_file)[0] + "_embedding.pt"
            save_path = os.path.join(class_save_dir, embedding_filename)
            torch.save(torch.tensor(embedding), save_path)
            print(f"Processed and saved embedding for {img_file}")

# Directories
data_dir = "data"
save_dir = "embeddings"
os.makedirs(save_dir, exist_ok=True)

# Process each class directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        process_directory(class_dir, class_name, save_dir)
