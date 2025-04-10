import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import numpy as np # type: ignore
from torch.cuda.amp import GradScaler, autocast   # type: ignore
import pandas as pd # type: ignore
import cv2 # type: ignore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class HandDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, frames_clip=30, resize=(100,100)):
        self.root_dir = str(root_dir)  # Ensure root_dir is string
        # Read CSV ensuring first column is string
        self.annotations = pd.read_csv(csv_file, dtype={0: str})  # Force first column as string
        self.transform = transform
        self.frames_clip = frames_clip
        self.resize = resize

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Ensure video_folder is string
        video_folder = str(self.annotations.iloc[idx, 0])  # Convert to string
        video_path = os.path.join(self.root_dir, video_folder)
        label = int(self.annotations.iloc[idx, 1])  # Ensure label is int

        frames = []
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])

        if len(frame_files) > self.frames_clip:
            indices = np.linspace(0, len(frame_files)-1, num=self.frames_clip, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        for frame_file in frame_files[:self.frames_clip]:
            frame_path = os.path.join(video_path, frame_file)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        while len(frames) < self.frames_clip:
            blank_frame = torch.zeros_like(frames[0]) if len(frames) > 0 else torch.zeros(3, *self.resize)
            frames.append(blank_frame)

        video_tensor = torch.stack(frames)
        return video_tensor, label

batch_size = 8
num_workers = 1
num_epochs = 1
learning_rate = 0.0001
num_classes = 27

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = HandDataset(
    root_dir="data/archive/Train",
    csv_file="data/archive/Train.csv",
    transform=train_transform,
    frames_clip=30,
    resize=(100, 100)
)
        
val_dataset = HandDataset(
    root_dir="data/archive/Validation",
    csv_file="data/archive/Validation.csv",
    transform=val_transform,
    frames_clip=30,
    resize=(100, 100)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
    

class HandCNN(nn.Module):
    def __init__(self, num_classes=num_classes, input_channels=3, frames=30, height=100, width=100):
        super(HandCNN, self).__init__()
        
        # Store input dimensions
        self.input_channels = input_channels
        self.frames = frames
        self.height = height
        self.width = width
        
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Calculate the flattened size
        self.flattened_size = self._calculate_flattened_size()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _calculate_flattened_size(self):
        with torch.no_grad():
            # Create dummy input with proper 5D shape (1 batch, channels, frames, height, width)
            dummy_input = torch.zeros(1, self.input_channels, self.frames, self.height, self.width)
            features = self.features(dummy_input)
            return features.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

model=HandCNN().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

scaler = torch.amp.GradScaler()

for epoch in range(num_epochs):

    model.train()
    train_loss=0.0
    correct_train=0
    total_train=0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to device
        inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs} | '
                  f'Batch: {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f}')
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * correct_train / total_train
    val_loss /= len(val_loader)
    val_acc = 100. * correct_val / total_val
    
    scheduler.step(val_loss)
    
    print(f'\nEpoch: {epoch+1}/{num_epochs} | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, f'checkpoint_epoch_{epoch}.pth')



input_size = torch.randn(1, 3, 30, 100, 100).to(device)
traced_model = torch.jit.trace(model, input_size)
traced_model.save("hand_gesture_model_no_cnn.pt")