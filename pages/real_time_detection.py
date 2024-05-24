import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import torch
import cv2
import pathlib

# Adding a temporary fix for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.title("Real-time")
# Load the YOLOv5 model
@st.cache_resource()
def load_yolov5_model(weights='models/best.pt'):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=True)
    model.eval()
    return model

# Initialize the model outside the callback to cache it
model = load_yolov5_model()

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

def main():
    st.title("YOLOv5 Object Detection with Webcam (WebRTC)")

    # Confidence threshold slider
    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.25)

    webrtc_ctx = webrtc_streamer(key="example",
                                 mode=WebRtcMode.SENDRECV,
                                 video_processor_factory=lambda: VideoProcessor(confidence_threshold),
                                 media_stream_constraints={"video": True, "audio": False})

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.update_confidence_threshold(confidence_threshold)

if __name__ == "__main__":
    main()
