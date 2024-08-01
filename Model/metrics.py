import os
import glob
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best_11.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
metrics = model.val(data='/Users/dimitridumont/code/skool/501/AAI-501/DataSet/data.yaml')

# Print out the evaluation metrics
print(metrics)

# You can also calculate mAP (mean Average Precision)
map_50 = metrics['metrics/mAP_0.5']
map_50_95 = metrics['metrics/mAP_0.5:0.95']

print(f"mAP@0.5: {map_50}")
print(f"mAP@0.5:0.95: {map_50_95}")