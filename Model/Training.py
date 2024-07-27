import os
import glob
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import yaml

import torch.optim as optim
import torch.nn as nn
from ultralytics import YOLO

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        self.transform = transform

        assert len(self.image_paths) == len(self.label_paths), \
            f"Number of images ({len(self.image_paths)}) and labels ({len(self.label_paths)}) do not match."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [list(map(float, label.strip().split())) for label in labels]
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels

def custom_collate_function(batch):
    images, labels = zip(*batch)

    # Stack images (adding batch dimension)
    images = torch.stack(images, dim=0)

    # Find max number of labels
    max_num_labels = max(label.size(0) for label in labels)

    # Pad labels
    padded_labels = torch.zeros((len(labels), max_num_labels, 5), dtype=torch.float32)
    for i, label in enumerate(labels):
        padded_labels[i, :label.size(0), :] = label

    return images, padded_labels

# Paths
train_image_dir = r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\train\images'
train_label_dir = r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\train\labels'
val_image_dir = r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\valid\images'
val_label_dir = r'C:\Users\mthiruma\OneDrive - Siemens Healthineers\Personal\2024\Learning\University of San Diego\AAI-501\Final Project\archive\TumorDetectionYolov8\OD8\Brain Tumor Detection\valid\labels'

# Transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Datasets and DataLoaders
train_dataset = BrainTumorDataset(train_image_dir, train_label_dir, transform=transform)
valid_dataset = BrainTumorDataset(val_image_dir, val_label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_function)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_function)

# Verify DataLoader
for batch in train_loader:
    images, labels = batch
    print(f'Images: {images.shape}, Labels: {labels.shape}')
    break

# Model initialization 
model_path = 'C:\\Users\\mthiruma\\OneDrive - Siemens Healthineers\\Personal\\2024\\Learning\\University of San Diego\\AAI-501\\Final Project\\yolov8.yaml'
data_path = 'C:\\Users\\mthiruma\\OneDrive - Siemens Healthineers\\Personal\\2024\\Learning\\University of San Diego\\AAI-501\\Final Project\\data.yaml'
# Model initialization using the pre-trained model
pretrained_model_path = 'C:\\Users\\mthiruma\\OneDrive - Siemens Healthineers\\Personal\\2024\\Learning\\University of San Diego\\AAI-501\\Final Project\\yolov8n.pt'  

# Training loop
epochs = 50

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the task explicitly
#task = 'detect'  # or 'segment', 'classify', etc., based on your model
# Initialize YOLOv8 model
#model = YOLO(model_path, task=task)
 
model = YOLO(pretrained_model_path)
model = model.to(device)

# Adjust the model to your number of classes
nc = 3  # Number of classes in your dataset
model.model.yaml['nc'] = nc
model.model.model[-1].nc = nc  # Update the number of classes in the detection layer
#model.model.model[-1].detect.reset_parameters()  # Reset the parameters for the detection layer

# Loss and optimizer
#loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute the loss
        #loss = loss_function(outputs, labels)
        loss = model.loss(outputs, labels)  # Use the model's loss function

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Model validation (if needed)
results = model.val(data=data_path, device=device)
print(results)