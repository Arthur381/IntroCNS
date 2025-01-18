import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.clock_driven import surrogate, neuron, functional
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 超参数设置
batch_size = 64
num_epochs = 15
learning_rate = 0.005
time_steps = 20  # 仿真时间步长
t_min, t_max = 0.0, 20.0  # TTFS 编码的时间范围

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# TTFS 编码器
def ttfs_encode(image, time_steps, t_min=0, t_max=20):
    """
    将图像像素值归一化后，基于 Time-to-First-Spike (TTFS) 生成脉冲序列。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 [Batch, Channels, Height, Width]
        time_steps (int): 时间步的数量
        t_min (int): 最小发放时间（默认 0）
        t_max (int): 最大发放时间（默认 20）

    Returns:
        torch.Tensor: 编码后的脉冲序列，形状为 [time_steps, Batch, Channels, Height, Width]
    """
    image = image / 255.0  # 归一化到 [0, 1]
    batch_size, channels, height, width = image.shape

    # 将图像值映射到时间范围 [t_min, t_max]
    spike_times = t_max - image * (t_max - t_min)

    # 初始化脉冲序列
    spikes = torch.zeros((time_steps, batch_size, channels, height, width), device=image.device)

    # 根据发放时间生成脉冲
    for t in range(time_steps):
        spikes[t] = (spike_times <= t).float()

    return spikes

def calculate_energy(inputs):
    """
    计算脉冲输入的总能量（非零值的数量）
    参数:
        inputs: 脉冲输入矩阵 (time_steps, C, H, W)，值为 0 或 1
    返回:
        energy: 输入矩阵中非零值的总数量
    """
    energy = inputs.sum().item()
    return energy


# Simple Repeat
def repeat_encode(image, time_steps):
    """
    将图像像素值归一化后，连续重复 time_steps 次
    """
    image = image / 255.0  # 归一化到 [0, 1]
    repeated_images = image.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)  # 在时间步维度重复
    return repeated_images

def gaussian_encode(image, time_steps):
    """
    将图像像素值归一化后，基于标准高斯分布生成脉冲序列。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 [Batch, Channels, Height, Width]
        time_steps (int): 时间步的数量

    Returns:
        torch.Tensor: 编码后的脉冲序列，形状为 [time_steps, Batch, Channels, Height, Width]
    """
    image = image / 255.0  # 归一化到 [0, 1]
    batch_size, channels, height, width = image.shape

    # 生成从标准高斯分布中采样的随机数
    random_values = torch.randn((time_steps, batch_size, channels, height, width), device=image.device)

    # 以图像值为阈值生成脉冲，基于标准高斯分布
    spikes = (random_values < image.unsqueeze(0)).float()  # 图像值作为发放概率
    return spikes


def poisson_encode(image, time_steps):
    """
    将图像像素值归一化后，基于泊松分布生成脉冲序列。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 [Batch, Channels, Height, Width]
        time_steps (int): 时间步的数量

    Returns:
        torch.Tensor: 编码后的脉冲序列，形状为 [time_steps, Batch, Channels, Height, Width]
    """
    image = image / 255.0  # 归一化到 [0, 1]
    batch_size, channels, height, width = image.shape

    # 生成随机数，与图像值比较来确定脉冲
    random_values = torch.rand((time_steps, batch_size, channels, height, width), device=image.device)
    spikes = (random_values < image.unsqueeze(0)).float()  # 图像值作为发放概率
    return spikes

cnt=[]
# 自定义SNN模型
class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        # 第一部分：Conv -> BN -> LIF -> MaxPool
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.neuron1 = neuron.LIFNode(surrogate_function=surrogate.ATan())

        # 第二部分：Conv -> BN -> LIF
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.neuron2 = neuron.LIFNode(surrogate_function=surrogate.ATan())

        # 第三部分：Conv -> BN -> LIF -> MaxPool
        self.conv3 = nn.Conv2d(12, 48, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.neuron3 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc1 = nn.Linear(48 * 8 * 8, 128)
        self.neuron4 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        # classification layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x:torch.Tensor):
        '''

        '''

        inputs=repeat_encode(x,time_steps)
        cnt.append(calculate_energy(inputs))


        # 仿真过程
        functional.reset_net(self)  # 重置网络状态
        #class_votes = torch.zeros(x.size(0), 10).to(x.device)
        for t in range(time_steps):
            out = F.relu(self.conv1(inputs[t]))
            out = self.bn1(out)
            out = self.neuron1(out)
            out = F.max_pool2d(out, kernel_size=2, stride=2)  # 下采样

            out = F.relu(self.conv2(out))
            out = self.bn2(out)
            out = self.neuron2(out)

            out = F.relu(self.conv3(out))
            out = self.bn3(out)
            out = self.neuron3(out)
            out = F.max_pool2d(out, kernel_size=2, stride=2)  # 下采样

            out = out.view(out.size(0), -1)  # Flatten
            out = self.fc1(out)
            out=self.neuron4(out)
            out = self.fc2(out)
            #print(np.shape(out))
            #class_votes += out  # 累积每个类别的概率
        return out

# 初始化模型、损失函数和优化器
model = SNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'Train_SNN_Sigmoid.pth')

# 测试
def test(time_steps):
    model.load_state_dict(torch.load('Train_SNN_ATan.pth'))
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def test_with_visualization(model, test_loader, time_steps, num_images=5):
    model.eval()
    selected_images, labels = next(iter(test_loader))
    selected_images, labels = selected_images[:num_images], labels[:num_images]
    selected_images = selected_images.to(device)

    # 定义需要记录的层
    layers_to_record = ["conv1", "neuron1", "conv2", "neuron2"]  # 根据模型层次调整
    spike_records = {layer: torch.zeros(time_steps, num_images, 32, 32).to(device) for layer in layers_to_record}

    # 显示输入图片
    fig, axes = plt.subplots(len(layers_to_record) + 1, num_images, figsize=(12, 8))
    fig.suptitle("Test Images and Neuron Activity", fontsize=28)

    # 第一行：输入图片
    for i in range(num_images):
        axes[0, i].imshow(selected_images[i].permute(1, 2, 0).cpu().numpy() / 2 + 0.5)  # CIFAR10 normalize还原
        axes[0, i].axis('off')
        #axes[0, i].set_title(f"Label: {labels[i].item()}")

    # 仿真过程中记录脉冲
    functional.reset_net(model)

    for t in range(time_steps):  # 此处应完整仿真 time_steps 个时间步
        out = (model.conv1(selected_images))
        spike_records["conv1"][t] = out.max(dim=1)[0]  # 记录 conv1 的脉冲情况
        out = model.bn1(out)
        spikes = model.neuron1(out).sign()
        spike_records["neuron1"][t] = spikes.max(dim=1)[0]  # 记录 neuron1 的脉冲情况

        # 模拟后续层
        out = (model.conv2(out))
        spike_records["conv2"][t] = out.max(dim=1)[0]  # 记录 conv2 的脉冲情况
        out = model.bn2(out)
        spikes = model.neuron2(out).sign()
        spike_records["neuron2"][t] = spikes.max(dim=1)[0]  # 记录 neuron2 的脉冲情况

    # 显示脉冲发放情况
    for row, layer in enumerate(layers_to_record, start=1):  # 每层对应一行
        for col in range(num_images):
            spike_sum = spike_records[layer][:, col].sum(dim=0).cpu().detach().numpy()  # 时间维度上的发放情况
            axes[row, col].imshow(spike_sum, cmap='binary')
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(layer, fontsize=12)  # 在第一列标注层名称

    plt.tight_layout()
    plt.show()




# 示例运行
#model.load_state_dict(torch.load('Train_SNN_ATan.pth'))
#test_with_visualization(model, test_loader, time_steps=50)
# 初始化参数


#train()
print(test(20))
print("energy=",np.mean(cnt))