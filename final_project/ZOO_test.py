import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义 LIF 神经元模型
class LIFNeuron(nn.Module):
    def __init__(self, tau=2.0, threshold=1.0):
        super(LIFNeuron, self).__init__()
        self.tau = tau  # 衰减时间常数
        self.threshold = threshold  # 发放阈值
        self.v = None  # 电压状态初始化

    def forward(self, input_current):
        if self.v is None:
            self.v = torch.zeros_like(input_current)
        self.v = self.v + (input_current - self.v) / self.tau
        spikes = (self.v >= self.threshold).float()
        self.v = self.v * (1 - spikes)
        return spikes

# 定义网络结构
class LIFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LIFNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        spikes = self.lif1(x)
        x = self.fc2(spikes)
        return x

# 零阶优化更新函数
def zero_order_update(model, inputs, targets, criterion, step_size=0.01, noise_scale=0.01):
    losses = []
    parameters = list(model.parameters())

    # 原始损失计算
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())

    # 遍历所有参数
    for param in parameters:
        if param.requires_grad:
            noise = noise_scale * torch.randn_like(param)
            param.data.add_(noise)
            output_positive = model(inputs)
            loss_positive = criterion(output_positive, targets)

            param.data.sub_(2 * noise)
            output_negative = model(inputs)
            loss_negative = criterion(output_negative, targets)

            gradient_estimate = (loss_positive - loss_negative) / (2 * noise_scale)
            param.grad = gradient_estimate * torch.ones_like(param)

            param.data.add_(noise)

    return losses

# 加载 CIFAR-10 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 定义模型和损失函数
input_size = 32 * 32 * 3  # CIFAR-10 输入为 32x32 RGB 图像
hidden_size = 128
num_classes = 10

model = LIFNetwork(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()

# 训练过程
step_size = 0.01
noise_scale = 0.01
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to('cpu'), targets.to('cpu')  # 如果有 GPU，可替换为 .to('cuda')
        one_hot_targets = torch.zeros(targets.size(0), num_classes).scatter_(1, targets.unsqueeze(1), 1)

        # 零阶优化更新
        zero_order_update(model, inputs, one_hot_targets, criterion, step_size, noise_scale)

        # 更新参数
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.add_(-step_size * param.grad)

        # 计算损失
        output = model(inputs)
        loss = criterion(output, targets)
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(trainloader):.4f}")

# 测试过程
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Accuracy on CIFAR-10 test set: {100 * correct / total:.2f}%")
