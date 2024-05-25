import cv2
import torch
from torch import hub
import numpy as np
import streamlit as st
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer

def predict(model, frame):
    """Generate predictions and annotate the predicted frame."""
    resx = model(frame, size = 416).crop(save=False)
    for d in resx:
        box = list(map(int, list(map(np.round, d['box']))))
        label = d['label']
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
        px, py = box[0], box[1]-10
        cv2.putText(frame, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
    
@st.cache(allow_output_mutation=True)
def load_model():
    """Load model from hub with custom weights."""
    model = hub.load("ultralytics/yolov5", 'custom', "models/best.pt")
    model.conf = 0.7
    return model
with st.spinner('Model is being loaded..'):
        model=load_model()

st.title("Object Detector")
run = st.checkbox('Camera')

class VideoProcessor:
    def recv(self, frame):
    # def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = predict(model, img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if run:
    webrtc_streamer(key="objectDetector", video_transformer_factory=VideoProcessor, rtc_configuration={ # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
