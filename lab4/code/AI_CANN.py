import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from CANN import CANN1D
import numpy as np
from scipy.ndimage import gaussian_filter



# 加载图像分类网络
model = resnet18(pretrained=True)
model.eval()

# 图像预处理
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 生成注意力图（示例）
def generate_attention(image):
    with torch.no_grad():
        features = model(image)
        attention_map = features.mean(dim=1)  # 简单平均激活值
    return attention_map



# 获取 CANN 的刺激
def map_attention_to_cann(attention_map, cann):
    # 将注意力图缩放到 CANN 的特征空间
    attention_map = attention_map / attention_map.max()  # 归一化
    stimulus = cann.A * torch.exp(-0.5 * (cann.x - attention_map.argmax(dim=1)) ** 2 / cann.a ** 2)
    return stimulus

def smooth_saliency_map(saliency_map, sigma=3):
    """
    对显著性图进行高斯平滑处理

    Parameters:
    saliency_map: 2D numpy array, 输入的显著性图
    sigma: float, 高斯核的标准差，控制平滑程度

    Returns:
    2D numpy array, 平滑后的显著性图
    """
    smoothed_map = gaussian_filter(saliency_map, sigma=sigma)
    return smoothed_map




def saliency_to_cann_input(saliency_map, sigma=3):
    """
    将显著性图转化为 CANN 的连续输入

    Parameters:
    saliency_map: 2D numpy array, 输入的显著性图
    sigma: float, 高斯平滑参数

    Returns:
    1D numpy array, 用于 CANN 的连续输入
    """
    # 平滑显著性图
    smoothed_map = smooth_saliency_map(saliency_map, sigma)
    # 取列平均值生成连续输入
    continuous_input = np.mean(smoothed_map, axis=0)
    return continuous_input


def map_to_cann_input(cann, continuous_input, sigma=0.5, A=1.0):
    """
    将连续输入映射到 CANN 的外部输入

    Parameters:
    cann: CANN 模型实例
    continuous_input: 1D numpy array, 连续输入信号
    sigma: float, 高斯平滑参数
    A: float, 刺激强度

    Returns:
    1D numpy array, 用于 CANN 的输入信号
    """
    x_focus = np.argmax(continuous_input)  # 确定显著区域的中心
    x_focus_pos = cann.x[x_focus]  # 获取对应的 CANN 空间位置
    cann_input = A * np.exp(-0.5 * ((cann.x - x_focus_pos) ** 2) / sigma**2)
    return cann_input




image=0
cann=CANN1D(num=512,k=2,J0=1)
map_attention_to_cann()