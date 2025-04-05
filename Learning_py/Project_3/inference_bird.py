import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import random
import numpy as np # type: ignore
from torch.cuda.amp import GradScaler, autocast   # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = torch.jit.load('optimized_bird.pt').to(device)
model.eval()

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

def random_image():
    all_files = []
    base_path = "data/CUB/images"
    for root, dirs, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)

    if not all_files:
        return None

    return random.choice(all_files)
    
    
path=random_image()
print(f"random image path: {path}")
pred_class = predict(path)
print(f"Predicted class: {pred_class}")