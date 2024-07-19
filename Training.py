import os 
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import glob
from ultralytics import YOLO
from PIL import Image
import numpy as np

import subprocess

train_image_dir = r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\train\images'
train_label_dir=r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\train\labels'
val_image_dir = r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\valid\images'
val_label_dir=r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\valid\labels'

#Check the dataset so that all the images have a corresponding label in the expected format. 

def check_dataset_structure(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if not os.path.exists(os.path.join(label_dir, label_file)):
            print(f"Label file missing for image: {image_file}")
        else:
            with open(os.path.join(label_dir, label_file), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Incorrect label format in file: {label_file}")
    print("Dataset structure is correct.")

#Check your Train data 
check_dataset_structure(train_image_dir, train_label_dir)
#Check your Valid data
check_dataset_structure(val_image_dir, val_label_dir)

def check_dataset_structure1(images_path, labels_path):
    images = sorted(os.listdir(images_path))
    labels = sorted(os.listdir(labels_path))
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Image or label directory is empty.")
    for image, label in zip(images, labels):
        if os.path.splitext(image)[0] != os.path.splitext(label)[0]:
            raise ValueError(f"Image and label file names do not match: {image} and {label}")
    print("Dataset structure is correct.")

# Load data.yaml configuration file 
with open(r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

print(f"Number of classes: {data_config['nc']}")
print(f"Class names: {data_config['names']}")

# Check dataset structure
check_dataset_structure1(data_config['train'], data_config['train'])
check_dataset_structure1(data_config['val'], data_config['val'])

class DatasetClass(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        #Set the path where Images and the labels are located 
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        self.transform = transform

        # Ensure image_paths and label_paths are of the same length
        assert len(self.image_paths) == len(self.label_paths), \
            f"Number of images ({len(self.image_paths)}) and labels ({len(self.label_paths)}) do not match."

        print(f'Number of images: {len(self.image_paths)}')
        print(f'Number of labels: {len(self.label_paths)}')

    def load_image_paths(self, images_path):
        return [os.path.join(images_path, fname) for fname in os.listdir(images_path) if fname.endswith('.jpg')]

    def load_label_paths(self, labels_path):
        return [os.path.join(labels_path, fname) for fname in os.listdir(labels_path) if fname.endswith('.txt')]
    
    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image

    def load_label(self, label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [list(map(float, label.strip().split())) for label in labels]
        labels = torch.tensor(labels, dtype=torch.float32)
        return labels
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Ensure the index is within the valid range
        if idx >= len(self.image_paths) or idx >= len(self.label_paths):
            raise IndexError("Index out of range")
        
        # Load and return a sample from the dataset at the given index
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = self.load_image(image_path)
        labels = self.load_label(label_path)

        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)

        return {'images': image, 'labels': labels}
    
# Ensure that all labels have the same size. 
# Either pad the labels to a fixed size ->   Need to know maximum possible size of labels. 
# Or use a custom collate function to handle the varying sizes.
def custom_collate_function(batch):
    images, labels = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Pad labels
    label_lengths = [len(label) for label in labels]
    max_len = max(label_lengths)
    
    padded_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            padding = torch.zeros(max_len - len(label), dtype=label.dtype)
            padded_label = torch.cat((label, padding), dim=0)
        else:
            padded_label = torch.tensor(label + [0] * (max_len - len(label)), dtype=torch.int64)
        padded_labels.append(padded_label)
    
    # Stack labels
    labels = torch.stack(padded_labels, dim=0)

    return images, labels


# Create dataset and dataloader instances
train_dataset = DatasetClass(train_image_dir, train_label_dir, transform=transforms.ToTensor())
valid_dataset = DatasetClass(val_image_dir,val_label_dir, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_function)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_function)

# Iterate through the data loader to check batch contents
# Ensures correct format and structure of data provided to data loader
for batch in train_loader:
    images, labels = batch['images'], batch['labels']
    print(f'Images: {images.shape}, Labels: {labels.shape}')
    # If problem loading, add more print statements here to inspect labels. 
    break  # Stop after the first batch for inspection

# Training configuration
model_path = 'C:\\Users\\mthiruma\\OneDrive - Siemens Healthineers\\Personal\\2024\\Learning\\University of San Diego\\AAI-501\\Final Project\\yolov8n.yaml'  
epochs = 50
img_size = 640
device = 'cpu'  # Change to 'cuda' if using a GPU

# Initialize YOLOv8 model
model = YOLO(model_path)

# loss_function = nn.CrossEntropyLoss()
# learning_rate = 0.001
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(epochs):
#     for batch in train_loader:
#         images, labels = batch['images'], batch['labels']
#         print(f'Batch images shape: {images.shape}')
#         print(f'Batch labels shape: {labels.shape}')
#         print(f'Batch labels: {labels}')

#         # Feed the batch into the model
#         outputs = model(images)

#          # Compute the loss
#         loss = loss_function(outputs, labels)

#          # Backward pass and optimization step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Optionally, print the loss for tracking progress
#         print(f'Epoch {epoch + 1}, Batch loss: {loss.item()}')


# #YOLO training command
# command = ["yolo", "task=detect", "mode=train", "model=yolov8n.yaml", "data=data.yaml", "epochs=50", "imgsz=640", "device=cpu"]
# result = subprocess.run(command, capture_output=True, text=True)

# print("Standard Output:", result.stdout)
# print("Standard Error:", result.stderr)


# Train the model
model.train(
    data='C:\\Users\\mthiruma\\OneDrive - Siemens Healthineers\\Personal\\2024\\Learning\\University of San Diego\\AAI-501\\Final Project\\data.yaml',
    epochs=epochs,
    imgsz=img_size,
    device=device
)

# Validate the model
results = model.val()
print(results)
