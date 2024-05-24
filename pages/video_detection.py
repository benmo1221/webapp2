import streamlit as st
import av
import cv2
import tempfile
import torch
import sys
import pathlib
from utils.general import non_max_suppression, scale_boxes

# Adding a temporary fix for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Add YOLOv5 repository to Python path
sys.path.append('yolov5')  # Adjust this path to your yolov5 repository

# Importing centralized model loader
from yolov5_model import load_yolov5_model  

# Function to process video frames
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        results = yolo_detector(img_tensor)
    pred = non_max_suppression(results, confidence_threshold, 0.45)[0]
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img.shape).round()
        for *xyxy, conf, cls in pred:
            label = f'{yolo_detector.names[int(cls)]} {conf:.2f}'
            xyxy = list(map(int, xyxy))
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Function to process video
def process_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        processed_frame = video_frame_callback(av_frame)
        processed_img = processed_frame.to_ndarray(format="rgb24")
        stframe.image(processed_img, channels="RGB")
    video.release()

# Page title and icon
st.set_page_config(page_title="Real-Time Weapon Detection", layout='wide', page_icon='./images/object.png')
st.title("Real-Time Weapon Detection")

# Description and instructions
st.write('Please upload an MP4 video file to perform real-time weapon detection.')

# File uploader
uploaded_file = st.file_uploader("Upload an MP4 video file", type=["mp4"])

# Model loading message
with st.spinner('Please wait while the model is loading...'):
    # Load YOLOv5 model
    yolo_detector = load_yolov5_model(weights='models/best.pt')
    st.success('Model loaded successfully!')

# Confidence threshold slider
confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.25)

# Check if a file is uploaded
if uploaded_file is not None:
    # Check file type
    if uploaded_file.type == "video/mp4":
        # Process the uploaded video
        process_video(uploaded_file)
    else:
        st.error("Please upload a valid MP4 video file.")
