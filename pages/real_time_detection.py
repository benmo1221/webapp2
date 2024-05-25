import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import torch
import cv2
import pathlib
import os
import asyncio
import logging
from urllib.error import HTTPError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Use the appropriate Path class depending on the OS
if os.name == 'nt':  # Windows
    pathlib.PosixPath = pathlib.WindowsPath
else:  # Non-Windows (e.g., Linux, MacOS)
    pathlib.WindowsPath = pathlib.PosixPath

# Directory to monitor for changes to the YOLOv5 model weights
model_directory = "models"

st.title("Real-time YOLOv5 Object Detection")

# Load the YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model(weights='models/best.pt', local_weights='local_model.pt'):
    try:
        # Try to load the model from the local path if it exists
        if os.path.exists(local_weights):
            model = torch.hub.load("ultralytics/yolov5", "custom", path=local_weights, force_reload=True)
            logging.info(f"Model loaded from local path: {local_weights}")
        else:
            # Otherwise, attempt to load it from the remote repository
            model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=True)
            logging.info(f"Model loaded from remote path: {weights}")
        model.eval()
        return model
    except HTTPError as e:
        logging.error(f"Failed to load YOLOv5 model. HTTP Error: {e}")
        st.error(f"Failed to load YOLOv5 model. HTTP Error: {e}")
        st.stop()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

# Initialize the model outside the callback to cache it
model = load_model()

class VideoProcessor(VideoProcessorBase):
    def __init__(self, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def update_confidence_threshold(self, new_threshold):
        self.confidence_threshold = new_threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform inference on the frame
        results = self.model(img)

        # Draw bounding boxes on the frame
        for det in results.xyxy[0]:  # xyxy format: (x1, y1, x2, y2, conf, cls)
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf >= self.confidence_threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{self.model.names[int(cls)]} ({conf:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class ModelChangeHandler(FileSystemEventHandler):
    def __init__(self, model_path):
        self.model_path = model_path

    def on_modified(self, event):
        if event.src_path.endswith('.pt') and event.src_path == self.model_path:
            logging.info("Model weights file modified. Reloading model...")
            model = load_model()
            video_processor.model = model
            logging.info("Model reloaded successfully.")

def main():
    st.title("YOLOv5 Object Detection with Webcam (WebRTC)")

    # Confidence threshold slider
    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.25)

    # Set up model change handler
    model_weights_path = os.path.join(model_directory, 'best.pt')
    event_handler = ModelChangeHandler(model_weights_path)
    observer = Observer()
    observer.schedule(event_handler, model_directory, recursive=False)
    observer.start()

    webrtc_ctx = webrtc_streamer(key="example",
                                 mode=WebRtcMode.SENDRECV,
                                 video_processor_factory=lambda: VideoProcessor(confidence_threshold),
                                 media_stream_constraints={"video": True, "audio": False})

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.update_confidence_threshold(confidence_threshold)

if __name__ == "__main__":
    # Ensure the event loop is properly handled
    if not hasattr(asyncio, 'get_running_loop'):
        asyncio.get_event_loop().run_until_complete(main())
    else:
        main()
