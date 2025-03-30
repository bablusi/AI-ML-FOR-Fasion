from datasets import load_dataset
import os 

# Load the Fashionpedia dataset from the Huggingface Datasets library
dataset = load_dataset("detection-datasets/fashionpedia")

# Define the folder where images will be saved
dataset_folder = r'D:\fasion\Data'

# Check if "Data" is a file; if so, remove it
if os.path.exists(dataset_folder) and not os.path.isdir(dataset_folder):
    os.remove(dataset_folder)  # Remove file if exists

# Create the folder if it doesn't exist
os.makedirs(dataset_folder, exist_ok=True)

# Function to save images from the dataset to the specified folder
def save_images(dataset, dataset_folder, num_images=1000):
    for i in range(num_images):
        try:
            # Extract the image from the dataset
            image = dataset['train'][i]['image']
            
            # Save the image in PNG format with a sequential filename
            image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))
        except Exception as e:
            print(f"Error saving image {i+1}: {e}")

# Call the function to save the first 1000 images
save_images(dataset, dataset_folder, num_images=1000)

print(f"Saved the first 1000 images to {dataset_folder}")
