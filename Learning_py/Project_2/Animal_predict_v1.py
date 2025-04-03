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

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(3072, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = SimpleNN().to(device)
model.load_state_dict(torch.load('animal.pth'))
model.eval()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    # Get results
    predicted_class = class_names[predicted.item()]
    confidence = probability[predicted.item()].item()
    
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    image_path = "../../Test/9ard.jpg"
    class_name, confidence = predict_image(image_path)
    print(f"Predicted class: {class_name} with {confidence:.2f}% confidence")