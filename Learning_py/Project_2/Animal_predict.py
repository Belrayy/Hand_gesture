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
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3072, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def load_model(model_type):
    if model_type == "1":
        model = SimpleNN().to(device)
        model.load_state_dict(torch.load('animal.pth', map_location=device))
    elif model_type == "2":
        model = CNN().to(device)
        model.load_state_dict(torch.load('animal_cnn.pth', map_location=device))
    else:
        raise ValueError("Invalid model type. Choose '1' or '2'.")
    model.eval()
    return model

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return class_names[predicted.item()], probability[predicted.item()].item()

if __name__ == "__main__":
    model_type = input("Choose model ('simple' or 'cnn'): ").strip().lower()
    model = load_model(model_type)

    image_path = "../../Test/cat.jpg"
    class_name, confidence = predict_image(image_path, model)
    print(f"Predicted: {class_name} ({confidence:.2f}% confidence)")