import streamlit as st
import cv2
import numpy as np
import torch
import pathlib
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

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

# Set up confidence threshold
confidence_threshold = 0.5

# Video Processor Class for Object Detection
class ObjectDetector(VideoProcessorBase):
    def __init__(self):
        super().__init__()

    def recv(self, frame: np.ndarray) -> np.ndarray:
        # Convert BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = yolo_detector.detect(rgb_frame, confidence_threshold)

        # Draw bounding boxes on the frame
        for *xyxy, conf, cls in results.xyxy[0]:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{yolo_detector.names[int(cls)]} {conf:.2f}', (int(xyxy[0]), int(xyxy[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

# Streamlit App
def main():
    st.title("Real-time Object Detection with YOLOv5")

    # Display confidence threshold slider
    global confidence_threshold
    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5)

    # Run object detection using Webrtc
    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=ObjectDetector,
                                 async_processing=True)

if __name__ == "__main__":
    main()
