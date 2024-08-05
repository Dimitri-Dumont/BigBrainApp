import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from ultralytics import YOLO
from torch.cuda.amp import GradScaler, autocast
import yaml

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        self.transform = transform
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels do not match."
        assert len(self.image_paths) != 0, "No images found."

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
    images = torch.stack(images, dim=0)
    max_num_labels = max(label.size(0) for label in labels)
    padded_labels = torch.zeros((len(labels), max_num_labels, 5), dtype=torch.float32)
    for i, label in enumerate(labels):
        padded_labels[i, :label.size(0), :] = label
    return images, padded_labels


img_dir = "../DataSet/TumorDetectionYolov8/OD8/Brain Tumor Detection/train/images"
label_dir ="../DataSet/TumorDetectionYolov8/OD8/Brain Tumor Detection/train/labels"
pt_path = "../detect/train8/weights/best.pt"
yaml_p = "../DataSet/data.yaml"

# Load configuration from YAML file
with open(yaml_p, 'r') as file:
    data_config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize the images to the required dimensions
    transforms.ToTensor()           # Convert the images to tensors
])

# Dataset and DataLoader
train_dataset = BrainTumorDataset(
    image_dir=img_dir,
    label_dir=label_dir,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_function)

# Model initialization
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")
model = YOLO(pt_path).to(device)
model.model.yaml['nc'] = data_config['nc']
model.model.model[-1].nc = data_config['nc']

# Optimizer and scaler
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
model.train(
    data=yaml_p,
    epochs=epochs,  # Explicitly set the number of epochs to 50
    batch=8,
    imgsz=640
)
    
# Model validation
results = model.val(data=yaml_p, device=device)
print(results)
model.export(format='onnx')

