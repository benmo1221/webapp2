import streamlit as st
import cv2
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Import the YOLOv5 model loader
from yolov5_model import load_yolov5_model

# Set page layout and title
st.set_page_config(layout="wide")
st.title("Real-time Object Detection with Webcam")

# Function to perform object detection on frames
class ObjectDetector(VideoProcessorBase):
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def recv(self, frame):
        # Perform inference on the frame
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)

        # Draw bounding boxes on the frame
        for det in results.xyxy[0]:  # xyxy format: (x1, y1, x2, y2, conf, cls)
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf >= self.confidence_threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{self.model.names[int(cls)]} ({conf:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Function to get the user's chosen model
def get_user_model():
    # Insert your code here to get the user's chosen model file
    return "path_to_your_model_file.pt"  # Replace this with the actual path to the model file

# Load YOLOv5 model
model_file_path = get_user_model()
if model_file_path and os.path.isfile(model_file_path):
    device_option = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_yolov5_model(weights=model_file_path, device=device_option)
else:
    st.warning("Model file not available! Please add it to the models folder.")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

# WebRTC streaming configuration
webrtc_ctx = webrtc_streamer(key="example",
                             mode=WebRtcMode.SENDRECV,
                             video_processor_factory=lambda: ObjectDetector(model, confidence_threshold),
                             media_stream_constraints={"video": True, "audio": False})

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.model.confidence_threshold = confidence_threshold
