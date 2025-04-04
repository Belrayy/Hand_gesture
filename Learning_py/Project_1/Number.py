import os
import torch # type: ignore
import torchvision.models as models # type: ignore
from torch import nn # type: ignore
from torchvision import transforms # type: ignore
import PIL.Image # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore

if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"
print(f"Using {device} device")

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model= SimpleNN().to(device)
model.load_state_dict(torch.load('number.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),          
    transforms.Resize((28, 28)),     
    transforms.ToTensor(),           
    transforms.Normalize((0.1307,), (0.3081,)) 
])

def preprocess_image(image_path):
    image = PIL.Image.open(image_path)
    image = transform(image).unsqueeze(0) 
    return image.to(device)

def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

correct = 0
total = 0

for digit_label in range(10):  # 0-9
    digit_dir = os.path.join("testing", str(digit_label))
    for image_name in os.listdir(digit_dir): 
        image_path = os.path.join(digit_dir, image_name)
        predicted = predict(image_path)
        total += 1
        if predicted == digit_label:
            correct += 1

print(f"Accuracy: {100 * correct / total:.2f}%")
print(f"Total images tested: {total}")