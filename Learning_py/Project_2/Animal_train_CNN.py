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

input_size = 3072
hidden_size = 128
num_classes = 10
num_epochs = 50
batch = 64
learning_rate = 0.001
kernel_size = 3
stride = 1
padding = 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64*8*8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

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
    for images, labels in test_loader:  # Now labels are batched (shape [batch_size])
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # Now this works (labels are [batch_size])
        correct += (predicted == labels).sum().item()

# Save the model
torch.save(model.state_dict(), 'animal_cnn.pth')



