# yolov5_model.py
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
def load_yolov5_model(weights='models/best.pt'):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    return model

# In each script
from yolov5_model import load_yolov5_model

# Use the model
model = load_yolov5_model(weights='models/best.pt')
