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
            transforms.Resize(
                (160, 160)
            ),  # Resize the image to the size expected by the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = resnet(img_tensor)
        return img_embedding.squeeze().cpu().numpy()


def process_directory(directory, class_name, save_dir):
    embeddings = []
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        embedding = get_embedding(img_path)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Processed {img_file}")
    # Save embeddings to a file
    save_path = os.path.join(save_dir, f"{class_name}_embeddings.pt")
    torch.save(torch.tensor(embeddings), save_path)


# Directories
data_dir = "data"
save_dir = "embeddings"
os.makedirs(save_dir, exist_ok=True)

# Process each class directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        process_directory(class_dir, class_name, save_dir)
