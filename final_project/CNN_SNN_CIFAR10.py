import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import ann2snn
# 定义 LIFNeuron
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


# 定义 SNN 模块（每个子模块单独实现）
class SNNRetina(nn.Module):
    def __init__(self, cnn_state_dict):
        super(SNNRetina, self).__init__()
        self.color_filter = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif_neuron1 = LIFNeuron()

        # 初始化权重
        self.color_filter.weight.data = cnn_state_dict['retina.color_filter.weight']
        self.color_filter.bias.data = cnn_state_dict['retina.color_filter.bias']

    def forward(self, x):
        x = self.color_filter(x)
        x = self.lif_neuron1(x)
        x = self.pool(x)
        return x


class SNNLGN(nn.Module):
    def __init__(self, cnn_state_dict):
        super(SNNLGN, self).__init__()
        self.enhance = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.integrate = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.lif_neuron1 = LIFNeuron()
        self.lif_neuron2 = LIFNeuron()

        # 初始化权重
        self.enhance.weight.data = cnn_state_dict['lgn.enhance.weight']
        self.enhance.bias.data = cnn_state_dict['lgn.enhance.bias']
        self.integrate.weight.data = cnn_state_dict['lgn.integrate.weight']
        self.integrate.bias.data = cnn_state_dict['lgn.integrate.bias']

    def forward(self, x):
        x = self.lif_neuron1(self.enhance(x))
        x = self.lif_neuron2(self.integrate(x))
        return x


class SNNVisualCortex(nn.Module):
    def __init__(self, cnn_state_dict):
        super(SNNVisualCortex, self).__init__()
        self.conv1 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(48 * 4 * 4, 10)
        self.lif_neuron1 = LIFNeuron()
        self.lif_neuron2 = LIFNeuron()

        # 初始化权重
        self.conv1.weight.data = cnn_state_dict['visual_cortex.conv1.weight']
        self.conv1.bias.data = cnn_state_dict['visual_cortex.conv1.bias']
        self.conv2.weight.data = cnn_state_dict['visual_cortex.conv2.weight']
        self.conv2.bias.data = cnn_state_dict['visual_cortex.conv2.bias']
        self.fc.weight.data = cnn_state_dict['visual_cortex.fc.weight']
        self.fc.bias.data = cnn_state_dict['visual_cortex.fc.bias']

    def forward(self, x):
        x = self.lif_neuron1(self.conv1(x))
        x = self.lif_neuron2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SNNVisionModel(nn.Module):
    def __init__(self, cnn_state_dict):
        super(SNNVisionModel, self).__init__()
        self.retina = SNNRetina(cnn_state_dict)
        self.lgn = SNNLGN(cnn_state_dict)
        self.visual_cortex = SNNVisualCortex(cnn_state_dict)

    def forward(self, x, time_steps=10):
        # 分类计数器：记录每个类别的脉冲总和
        class_spike_counts = torch.zeros(x.size(0), 10).to(x.device)  # 每个类别的脉冲计数
        #print(np.shape(class_spike_counts))
        # 进行时间步的迭代
        for _ in range(time_steps):
            out = self.retina(x)  # 经过视网膜处理
            out = self.lgn(out)  # 经过 LGN 处理
            out = self.visual_cortex(out)  # 经过视觉皮层处理
            # 对每个类别的脉冲计数进行累加
            class_spike_counts += out  # 将每个像素的发放次数加总

        # 返回脉冲计数最多的类别
        return class_spike_counts.argmax(dim=1)  # 返回投票最多的类别


# 测试 SNN 模型
def test_snn():
    cnn_state_dict = torch.load('cifar10_cnn_model.pth')
    snn_model = SNNVisionModel(cnn_state_dict)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    snn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = snn_model(images, time_steps=10)  # 使用 time_steps 次分类
            print(predictions)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print(f"SNN Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    test_snn()
