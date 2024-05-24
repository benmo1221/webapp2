import streamlit as st
from PIL import Image
import numpy as np
import torch
import sys
import cv2
import pathlib

# Adding a temporary fix for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Importing non-max suppression and scaling functions from utils
from utils.general import non_max_suppression, scale_boxes

# Add YOLOv5 repository to Python path
sys.path.append('yolov5')  # Adjust this path to your yolov5 repository

# Importing centralized model loader
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

# Function to upload image
def upload_image():
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size / (1024 ** 2)
        file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": "{:,.2f} MB".format(size_mb)}
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Valid image file type (PNG or JPEG).')
            image_obj = Image.open(image_file)
            if image_obj.mode == 'RGBA':
                image_obj = image_obj.convert('RGB')
            return {"file": image_file, "details": file_details}
        else:
            st.error('Invalid image file type. Please upload PNG or JPEG.')
            return None

# Main function to process uploaded image
def main():
    object = upload_image()
    if object:
        prediction = False
        image_obj = Image.open(object['file'])
        col1, col2 = st.columns(2)
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
        with col2:
            st.subheader('File Details')
            st.json(object['details'])
            button = st.button('Detect Weapons')
            if button:
                with st.spinner("Detecting weapons. Please wait..."):
                    image_array = np.array(image_obj)
                    img_resized = cv2.resize(image_array, (640, 640))
                    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    with torch.no_grad():
                        results = yolo_detector(img_tensor)
                    pred = non_max_suppression(results, confidence_threshold, 0.45)[0]
                    if pred is not None and len(pred):
                        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], image_array.shape).round()
                        for *xyxy, conf, cls in pred:
                            label = f'{yolo_detector.names[int(cls)]} {conf:.2f}'
                            xyxy = list(map(int, xyxy))
                            cv2.rectangle(image_array, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
                            cv2.putText(image_array, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    pred_img_obj = Image.fromarray(image_array)
                    prediction = True
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5 model")
            st.image(pred_img_obj)

if __name__ == "__main__":
    main()
