import io
import os
import warnings

import requests
import streamlit as st
from PIL import Image

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Plant Leaves | Health Classification",
    page_icon="🌿",
)

# App Title
st.title("🌿 Plant Leaves | Health Classification")

# File uploader for image input
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg"])

# Placeholder for displaying the result
result_placeholder = st.empty()


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """
    Converts a PIL Image to bytes.

    Args:
        image (Image.Image): The image to convert.
        format (str): The format to save the image in (e.g., "JPEG").

    Returns:
        bytes: The byte representation of the image.
    """
    byte_stream = io.BytesIO()
    image.save(byte_stream, format=format)
    byte_stream.seek(0)  # Move to the beginning of the byte stream
    return byte_stream.read()


def classify_leaf(image: Image.Image):
    # Perform model inference (example using a hypothetical API)
    image = image_to_bytes(image)
    endpoint = os.getenv("BACKEND_URI", "http://backend:8000")
    response = requests.post(f"{endpoint}/predict", files={"data": image})
    try:
        prediction = response.json()  # Attempt to decode JSON
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON response.")
        print("Response content:", response.text)
        prediction = {"error": "Invalid response from backend."}

    return prediction


# Process the uploaded image
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    # Load the image
    image = Image.open(uploaded_file)

    # Perform inference
    with st.spinner("Classifying..."):
        prediction = classify_leaf(image)

    # Display result
    print(type(prediction))
    status = prediction["status_code"]
    label = prediction["image_label"]
    confidence = prediction["confidence"]
    icon = "✅" if label == "healthy" else "⚠️"
    result_placeholder.markdown(
        f"### {icon} The leaf is **{label.upper()}** with **{confidence * 100:.2f}%** confidence."
    )
else:
    st.info("Please upload an image of a plant leaf to classify its health.")
