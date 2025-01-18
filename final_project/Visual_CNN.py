'''
plaintext
复制代码
Input (CIFAR-10: [Batch, 3, 32, 32])
        |
    Retina
        |-- Conv2D (3 -> 6, kernel=5, stride=1, padding=2)
        |-- MaxPool2D (kernel=2, stride=2)
        V
     Output: [Batch, 6, 16, 16]
        |
      LGN
        |-- Conv2D (6 -> 12, kernel=3, stride=1, padding=1)
        |-- Conv2D (12 -> 12, kernel=3, stride=1, padding=1)
        V
     Output: [Batch, 12, 16, 16]
        |
  Visual Cortex
        |-- Conv2D (12 -> 24, kernel=3, stride=1, padding=1)
        |-- Conv2D (24 -> 48, kernel=3, stride=1, padding=1)
        |-- AdaptiveAvgPool2D (output_size=4x4)
        |-- Flatten
        |-- Fully Connected Layer (48*4*4 -> 10)
        V
  Output: [Batch, 20]
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Retina Module
class Retina(nn.Module):
    def __init__(self):
        super(Retina, self).__init__()
        self.color_filter = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # Simulates P and M cells
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling for feature extraction

    def forward(self, x):
        x = self.pool(F.relu(self.color_filter(x)))
        return x

# LGN Module
class LGN(nn.Module):
    def __init__(self):
        super(LGN, self).__init__()
        self.enhance = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)  # Feature enhancement
        self.integrate = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Spatial integration

    def forward(self, x):
        x = F.relu(self.enhance(x))
        x = F.relu(self.integrate(x))
        return x

# Visual Cortex Module (V1-V4)
class VisualCortex(nn.Module):
    def __init__(self):
        super(VisualCortex, self).__init__()
        self.conv1 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Global feature integration
        self.fc = nn.Linear(48 * 4 * 4, 10)  # Classification layer (for CIFAR-10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the FC layer
        x = self.fc(x)
        return x

# Complete Model
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.retina = Retina()
        self.lgn = LGN()
        self.visual_cortex = VisualCortex()

    def forward(self, x):
        x = self.retina(x)
        x = self.lgn(x)
        x = self.visual_cortex(x)
        return x




# Training setup
def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    learn_rate=0.02
    num_epochs = 15

    model = VisionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("Training Complete")

    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    state_dict = model.state_dict()
    torch.save(state_dict, 'cifar10_cnn_f.pth')


def test():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VisionModel()


    model.load_state_dict(torch.load('cifar10_cnn_model.pth'))
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    #train_model()
    test()
