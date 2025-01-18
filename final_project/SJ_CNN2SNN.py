import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import ann2snn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Retina Module
class Retina(nn.Module):
    def __init__(self):
        super(Retina, self).__init__()
        self.color_filter = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # Simulates P and M cells
        self.re=nn.ReLU()
        self.pool = nn.MaxPool2d((2,2))  # Downsampling for feature extraction

    def forward(self, x):
        x = self.pool(self.re(self.color_filter(x)))
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
        #self.re_1=nn.ReLU()
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        #self.re_2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Global feature integration
        self.fc = nn.Linear(48 * 4 * 4, 10)  # Classification layer (for CIFAR-10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x=self.re_1(x)
        x = F.relu(self.conv2(x))
        #x=self.re_2(x)
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training setup
def train_model():



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


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
        acc = correct / total
    return acc

def test():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VisionModel()


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

T=30


model=VisionModel()
model.load_state_dict(torch.load('cifar10_cnn_model.pth'))
print('Converting using RobustNorm')
acc = val(model, device, test_loader)
print('ANN Validating Accuracy: %.4f' % (acc))

model_converter = ann2snn.Converter(mode='max', dataloader=train_loader)
snn_model = model_converter(model)
print(snn_model)
print('Simulating...')
acc=val(snn_model, device, test_loader, T=T)
print('CNN Validating Accuracy: %.4f' % (acc))
