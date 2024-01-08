
# Face Recognition Project

This face recognition project involves collecting images, generating embeddings, performing clustering, and running a real-time face recognition application. Follow the steps below to set up and run the project.

## Setup

### Creating a Virtual Environment

Creating a virtual environment is recommended to manage dependencies.

```bash
python -m venv venv
```

### Activating the Virtual Environment

Activate the virtual environment. The activation command varies depending on your operating system.

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```

- On Unix or MacOS:
  ```bash
  source venv/bin/activate
  ```

### Installing Required Packages

Install all required Python packages.

```bash
pip install -r requirements.txt
```

## Running the Application

### Starting the Application

Run the main application script.

```bash
python main.py
```

### Collecting Images

Use the application to collect images. When you run `main.py`, it starts a server that you can interact with through a web interface.

- Navigate to the web interface and use the 'Save Image' button to collect images for the first person.
- The images are saved in a specified directory.

### Changing Directory for the Second Person

After collecting enough images for the first person, stop the application. Change the directory name in the code where images are being saved to start collecting images for the second person.

### Restarting and Collecting More Images

Restart the application and collect images for the second person using the same 'Save Image' button.

## Generating Embeddings

Once you have collected enough images for both individuals:

### Running `create_embeddings.py`

This script will process the collected images and generate embeddings.

```bash
python create_embeddings.py
```

## Performing Clustering

After generating embeddings:

### Running `k_means_clustering.py`

This script applies K-means clustering to the embeddings.

```bash
python k_means_clustering.py
```

## Final Steps

Finally, run `main.py` again.

```bash
python main.py
```

Now, the application should be able to recognize faces in real-time and display names next to the detected faces based on the clusters.

## Notes

- Ensure all scripts (`main.py`, `create_embeddings.py`, `k_means_clustering.py`) are in the project directory.
- Adjust any file paths or configurations in the scripts as necessary for your setup.
