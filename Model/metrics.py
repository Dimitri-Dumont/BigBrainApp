import os
import glob
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../detect/train8/weights/best.pt")
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # Apple silicon gpu
else:
    device = torch.device('cpu')


model = model.to(device)
metrics = model.val(data='../DataSet/data.yaml')

# Print out the evaluation metrics
print(metrics)
