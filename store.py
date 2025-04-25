import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os

# Define the dataset folder
dataset_folder = r'D:\fasion\Data'

# Check if the dataset folder exists
if not os.path.exists(dataset_folder):
    raise FileNotFoundError(f"❌ Folder not found: {dataset_folder}")

# Initialize the ChromaDB persistent client to store image vectors
chroma_client = chromadb.PersistentClient(path="Vector_Database")

# Initialize the embedding function
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get the collection for storing image vectors
image_vdb = chroma_client.get_or_create_collection(
    name="image", 
    embedding_function=embedding_function
)

# Initialize lists to store image IDs, documents (file paths), and metadata
ids = []
documents = []
metadatas = []

# Iterate over each image file in the dataset folder
try:
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        # Check if the file is a PNG image
        if filename.endswith('.png'):
            file_path = os.path.join(dataset_folder, filename)
            
            # Add the image ID and file path to the lists
            ids.append(str(i))
            documents.append(file_path)
            
            # Add metadata with a default or derived "topic" value
            # For example, set topic to "fashion" or parse from filename if needed
            metadatas.append({"topic": "fashion"})

    # Add the images to the vector database with metadata
    if ids and documents:
        image_vdb.add(ids=ids, documents=documents, metadatas=metadatas)
        print("✅ Images have been successfully stored to the Vector database with metadata.")
    else:
        print("⚠️ No PNG images found in the folder.")
        
except Exception as e:
    print(f"❌ An error occurred: {e}")
