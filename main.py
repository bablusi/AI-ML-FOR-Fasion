import streamlit as st
import numpy as np
import PIL.Image
import io
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings
import json
from datetime import datetime
from tensorflow import keras
from fpdf import FPDF

warnings.filterwarnings("ignore")

load_dotenv()
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)

HISTORY_FILE = "styling_history.json"
WARDROBE_DIR = "wardrobe_images"

# Function to save styling history
def save_to_history(entry):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            history = json.load(file)
    else:
        history = []

    history.append(entry)
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Function to load styling history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Function to clear styling history
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# Function to download recommendations as PDF
def download_recommendations():
    history = load_history()
    if not history:
        st.error("No recommendations to download.")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for entry in history:
        pdf.cell(200, 10, f"Timestamp: {entry['timestamp']}", ln=True)
        pdf.cell(200, 10, f"Query: {entry['query']}", ln=True)
        for rec in entry['recommendations']:
            pdf.multi_cell(0, 10, f"{rec['image']} - {rec['advice']}")
        pdf.ln(10)

    pdf.output("recommendations.pdf")
    st.success("Recommendations downloaded as PDF.")

# Function to tag style
def tag_style(tags):
    return ", ".join(tags)

# Function to open an image
def open_image(img_data):
    if isinstance(img_data, str):
        response = requests.get(img_data)
        img = PIL.Image.open(io.BytesIO(response.content))
    elif isinstance(img_data, np.ndarray):
        img = PIL.Image.fromarray(img_data.astype('uint8'))
    elif isinstance(img_data, list):
        try:
            img_data = np.array(img_data, dtype='uint8')
            img = PIL.Image.fromarray(img_data)
        except Exception as e:
            st.error(f"Error converting list to array: {e}")
            raise ValueError("Unsupported image data format")
    else:
        raise ValueError("Unsupported image data format")
    return img

# Function for wardrobe scanning
def wardrobe_scanning():
    st.subheader("Wardrobe Scanning")
    uploaded_files = st.file_uploader("Upload your wardrobe images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            image = np.array(PIL.Image.open(file))
            st.image(image, caption=f"Scanned: {file.name}")

# Function for mood board management
def add_to_mood_board(image):
    if not os.path.exists(WARDROBE_DIR):
        os.makedirs(WARDROBE_DIR)
    image.save(os.path.join(WARDROBE_DIR, f"mood_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    st.success("Added to Mood Board!")

# Streamlit app
st.title("AI the Fashion Styling Assistant")

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
query = st.text_input("Enter styling query:")
tags = st.multiselect("Select tags:", ["Casual", "Formal", "Summer", "Winter", "Party", "Business"])

# Sidebar for displaying history
st.sidebar.subheader("Styling History")
history = load_history()
if history:
    for entry in history[::-1]:
        st.sidebar.write(f"**{entry['timestamp']} - {entry['query']}**")
        for rec in entry["recommendations"]:
            st.sidebar.write(f"- {rec['advice'][:100]}...")

# Show history if requested
if st.button("Show History"):
    history = load_history()
    st.write(history)

# Clear history
if st.button("Clear History"):
    clear_history()
    st.success("History cleared successfully.")

# Download recommendations
if st.button("Download Recommendations"):
    download_recommendations()

# Social sharing
if st.button("Share on Social Media"):
    st.success("Shared on social media!")

# E-commerce linking
if st.button("Shop Similar Styles"):
    st.write("[Visit our e-commerce site](https://www.example.com)")

# Generate recommendations
if st.button("Generate Styling Ideas"):
    chroma_client = chromadb.PersistentClient(path="Vector_database")
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query if query else "Image Upload",
        "tags": tags,
        "recommendations": []
    }

    if uploaded_file is not None:
        uploaded_image = np.array(PIL.Image.open(uploaded_file))
        retrieved_imgs = image_vdb.query(query_images=[uploaded_image], include=['data'], n_results=3)

        for i, img_data in enumerate(retrieved_imgs['data'][0]):
            img = open_image(img_data)
            st.image(img, caption=f"Image {i+1}")
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = model.generate_content(["Generate styling advice.", img])
            advice = response.text
            st.write(f"Styling Advice: {advice}")
            history_entry["recommendations"].append({"image": f"Image {i+1}", "advice": advice})

            add_to_mood_board(img)

        save_to_history(history_entry)

# Mood board feature
if st.button("Add to Mood Board"):
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        add_to_mood_board(image)

# Visualize mood board
if st.button("View Mood Board"):
    st.image([os.path.join(WARDROBE_DIR, img) for img in os.listdir(WARDROBE_DIR)])

# Wardrobe scanning
if st.button("Scan Wardrobe"):
    wardrobe_scanning()

# Gamification example
st.sidebar.write("🎖️ Your Style Points: 100")
st.sidebar.write("Level up by using more features!")

# Model customization
st.sidebar.subheader("Model Customization")
selected_model = st.sidebar.selectbox("Choose a model:", ["Model A", "Model B", "Model C"])
