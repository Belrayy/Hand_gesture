import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import numpy as np # type: ignore
from torch.cuda.amp import GradScaler, autocast   # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class OptimizedCUBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Pre-load all paths and labels into memory
        self.image_paths = []
        with open(os.path.join(root_dir, "CUB", "images.txt"), 'r') as f:
            for line in f:
                self.image_paths.append(line.strip().split()[1])
        
        # Pre-load all labels
        self.labels = []
        with open(os.path.join(root_dir, "CUB", "image_class_labels.txt"), 'r') as f:
            for line in f:
                self.labels.append(int(line.strip().split()[1]) - 1)  # 0-based index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "CUB", "images", self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Hyperparameters
batch_size = 128  
num_workers = 8   
num_epochs = 20
learning_rate = 0.001

# Optimized transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset and dataloader with optimizations
dataset = OptimizedCUBDataset(root_dir="data", transform=train_transform)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Faster data transfer to GPU
    persistent_workers=True  # Maintains workers between epochs
)

# Optimized CNN model
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(OptimizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),  # Increased channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Better than Flatten + Linear
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model with optimizations
model = OptimizedCNN(num_classes=200).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW is better
scaler = GradScaler()  # For mixed precision training
scheduler = torch.optim.lr_scheduler.OneCycleLR(  # Fast convergence
    optimizer, 
    max_lr=learning_rate*10,
    steps_per_epoch=len(dataloader),
    epochs=num_epochs
)

# Training loop with optimizations
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed, Avg Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), 'optimized_bird.pth')
print("Training complete")