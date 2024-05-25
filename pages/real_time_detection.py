import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import pathlib
import os

# Use the appropriate Path class depending on the OS
if os.name == 'nt':  # Windows
    pathlib.PosixPath = pathlib.WindowsPath
else:  # Non-Windows (e.g., Linux, MacOS)
    pathlib.WindowsPath = pathlib.PosixPath

# Importing non-max suppression and scaling functions from utils
from utils.general import non_max_suppression, scale_boxes

# Import centralized model loader
from yolov5_model import load_yolov5_model

# Create YOLOv5 detector instance
yolo_detector = load_yolov5_model(weights='models/best.pt')

# Setting page layout and title
st.set_page_config(page_title="Weapon Detection", layout='wide')

# Adding a title and description to the page
st.title('Weapon Detection App')
st.write('Upload an image to detect weapons.')

# Confidence threshold slider
confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5)

# Loading spinner while model is loading
with st.spinner('Please wait while the model is loading...'):
    st.success('Model loaded successfully!')

# Function to perform object detection
def detect_objects(image):
    image = np.array(image)
    # Perform object detection using the loaded YOLOv5 model
    results = yolo_detector(image, confidence_threshold)
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    for *xyxy, conf, cls in results.xyxy[0]:
        cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
        cv2.putText(image, f'{yolo_detector.names[int(cls)]} {conf:.2f}', (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Streamlit function to upload image and perform detection
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to perform object detection
    if st.button('Detect Objects'):
        # Perform object detection
        with st.spinner('Detecting objects...'):
            results = detect_objects(image)
        
        # Draw bounding boxes on the image
        image_np = np.array(image)
        draw_boxes(image_np, results)

        # Display annotated image
        st.image(image_np, caption='Annotated Image', use_column_width=True)
