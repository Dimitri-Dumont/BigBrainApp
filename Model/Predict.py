import os
import glob
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the latest weights
model = YOLO("../detect/train8/weights/best.pt")
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # Apple silicon gpu
else:
    device = torch.device('cpu')

model = model.to(device)

# Path to the folder containing the images
image_folder = '../DataSet/TumorDetectionYolov8/OD8/Brain Tumor Detection/test/images'
annotation_folder = '../DataSet/TumorDetectionYolov8/OD8/Brain Tumor Detection/test/labels'  # Adjust path if necessary

# Get a list of all image files in the folder
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # Adjust the extension if needed

# Helper function to convert normalized coordinates to absolute pixel coordinates
def convert_yolo_bbox(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2

# Initialize counters
total_images = 0
matched_images = 0
partial_overlap_images = 0

# Process each image in the folder
for image_path in image_paths:
    total_images += 1

    # Load an image using OpenCV
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape

    # Load corresponding annotation file
    base_name = os.path.basename(image_path).replace('.jpg', '.txt')
    annotation_path = os.path.join(annotation_folder, base_name)
    
    # Read ground truth bounding boxes
    with open(annotation_path, 'r') as f:
        ground_truth_boxes = []
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = [float(part) for part in parts[1:]]
            ground_truth_boxes.append(convert_yolo_bbox(bbox, img_width, img_height))

    # Convert image to RGB (YOLO expects images in RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Access the predicted boxes
    predicted_boxes = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:  # Access the bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            predicted_boxes.append((x1, y1, x2, y2))

    # Check for matches and partial overlaps
    overlap_found = False
    partial_overlap_found = False
    for (px1, py1, px2, py2) in predicted_boxes:
        for (gx1, gy1, gx2, gy2) in ground_truth_boxes:
            if (px1 < gx2 and px2 > gx1 and py1 < gy2 and py2 > gy1):  # Check for overlap
                if abs(px1 - gx1) < 10 and abs(py1 - gy1) < 10 and abs(px2 - gx2) < 10 and abs(py2 - gy2) < 10:
                    overlap_found = True
                else:
                    partial_overlap_found = True

    if overlap_found:
        matched_images += 1

    if partial_overlap_found:
        partial_overlap_images += 1

    if overlap_found or partial_overlap_found:
        # Draw ground truth boxes (green) and predicted boxes (red) on the image
        for (gx1, gy1, gx2, gy2) in ground_truth_boxes:
            cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)  # Green box for ground truth

        for (px1, py1, px2, py2) in predicted_boxes:
            if overlap_found:
                cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 255), 2)  # Purple box  Exact match
            else:
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)  # Blue box for partial match

        # Convert image back to BGR for displaying with OpenCV (if necessary)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # # Display the image with bounding boxes using matplotlib
        # plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()        

# Print the result
print(f"Total images processed: {total_images}")
print(f"Number of images with matched tumor locations: {matched_images}")
print(f"Number of images with partial overlaps: {partial_overlap_images}")
