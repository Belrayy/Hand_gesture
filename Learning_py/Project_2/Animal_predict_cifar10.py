import torch  # type: ignore
from torch import nn   # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision import datasets  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader  # type: ignore

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define the models
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

# Load model
def load_model(model_type):
    if model_type == "1":
        model = SimpleNN().to(device)
        model.load_state_dict(torch.load('animal.pth', map_location=device))
    elif model_type == "2":
        model = CNN().to(device)
        model.load_state_dict(torch.load('animal_cnn.pth', map_location=device))
    else:
        raise ValueError("Invalid model type. Choose 'simple' or 'cnn'.")
    model.eval()
    return model

batch = 64

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Main execution
if __name__ == "__main__":
    model_type = input("Choose model ('simple' or 'cnn'): ").strip().lower()
    model = load_model(model_type)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on CIFAR-10 test images: {100 * correct / total:.2f}%")
    print(f"Total images: {total}, Correct predictions: {correct}")