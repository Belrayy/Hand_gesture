import torch  # type: ignore
from torch import nn   # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision import datasets  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader  # type: ignore

if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 stats
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

animal_classes = {'bird', 'cat', 'deer', 'dog', 'frog', 'horse'}  
non_animal_classes = {'airplane', 'automobile', 'ship', 'truck'}  

input_size = 3072
hidden_size = 128
num_classes = 10
num_epochs = 25
batch = 64
learning_rate = 0.001

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x=x.view(-1, input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

for epoch in range(num_epochs):
    for images, labels in train_loader:  # Now yields batches of tensors
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataset:  # Now properly batched tensors
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'animal.pth')
