import torch  # type: ignore
from torch import nn   # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision import datasets  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image  # type: ignore

if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"
print(f"Using {device} device")


class CUBDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = pd.read_csv(f"{root_dir}/CUB/images.txt", sep=" ", header=None)[1]
        self.labels = pd.read_csv(f"{root_dir}/CUB/image_class_labels.txt", sep=" ", header=None)[1] - 1  
        self.attributes = pd.read_csv(f"{root_dir}/CUB/attributes/image_attribute_labels.txt",sep=" ", header=None,engine="python")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.root_dir}/CUB/images/{self.image_paths[idx]}"
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        attr = self.attributes[idx]
        if self.transform:
            image = self.transform(image)
        return image, {"species": label, "attributes": attr}  
    
batch = 64
num_epochs = 1

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = CUBDataset(root_dir="data", transform=transform)

dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

class CNN(nn.Module):
    def __init__(self,num_classes=200):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = CNN(num_classes=200)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = CNN().to(device)

for epoch in range(num_epochs):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'bird.pth')
