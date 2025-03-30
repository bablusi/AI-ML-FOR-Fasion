import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

# Define the dataset folder
dataset_folder = r'D:\fasion\Data'

# Check if the dataset folder exists
if not os.path.exists(dataset_folder):
    raise FileNotFoundError(f"❌ Folder not found: {dataset_folder}")

# Initialize the ChromaDB persistent client to store image vectors
chroma_client = chromadb.PersistentClient(path="Vector_Database")

# Initialize the image loader and embedding function
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

# Create or get the collection for storing image vectors
image_vdb = chroma_client.get_or_create_collection(
    name="image", 
    embedding_function=CLIP, 
    data_loader=image_loader
)

# Initialize lists to store image IDs and file paths
ids = []
uris = []

# Iterate over each image file in the dataset folder
try:
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        # Check if the file is a PNG image
        if filename.endswith('.png'):
            file_path = os.path.join(dataset_folder, filename)
            
            # Add the image ID and file path to the lists
            ids.append(str(i))
            uris.append(file_path)

    # Add the images to the vector database
    if ids and uris:
        image_vdb.add(ids=ids, uris=uris)
        print("✅ Images have been successfully stored to the Vector database.")
    else:
        print("⚠️ No PNG images found in the folder.")
        
except Exception as e:
    print(f"❌ An error occurred: {e}")
