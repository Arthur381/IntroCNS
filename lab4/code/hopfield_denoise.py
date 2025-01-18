import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def load_handwritten_digits(n_samples=10,if_orth=0):
    """加载手写数字数据集并将其转换为二进制黑白图像"""
    mnist = fetch_openml('mnist_784', version=1)
    data, labels = mnist.data, mnist.target.astype(int)
    
    # 手动设置列名
    data.columns = [f'pixel_{i}' for i in range(784)]  # 784个特征列

    # 随机选取样本
    indices = np.random.choice(len(data), n_samples, replace=False)
    images = data.iloc[indices]  # 使用iloc获取随机选取的样本
    labels = labels[indices]
    
    # 将灰度图像转换为黑白图像（0/1）
    binary_images = (images.values > 127).astype(int)  # 阈值为127
    
    def gram_schmidt_orthogonalization(patterns):
        """
        使用Gram-Schmidt正交化方法对输入的patterns进行正交化。
        输出的每个pattern保持为二进制（0和1）。
        """
        n_patterns, n_features = patterns.shape
        orthogonal_patterns = np.zeros_like(patterns, dtype=float)  # 用于存储正交化的结果

        pass

        return orthogonal_patterns
    

    patterns=[]
    # 正交化
    if(if_orth):
        patterns=gram_schmidt_orthogonalization(binary_images)

    return binary_images,labels


def plot_images(images, title="Images", rows=1, cols=10,test_num=0):
    """可视化多个图像"""
    
    if(not test_num):
        if test_num==0 and cols==1:
            pass
        else:

            fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
            axes = axes.flatten()  # 将多维子图数组展开为一维
            #print(axes)
            for i, ax in enumerate(axes):
                ax.imshow(images[i].reshape(28, 28), cmap='gray')
                ax.axis('off')
    else:
        images_sort=[]
        for i in range(test_num):
            for j in range(3):
                images_sort.append(images[j][i])

        fig, axes = plt.subplots(rows, cols, figsize=(8, 30))
        axes = axes.flatten()  # 将多维子图数组展开为一维
        #print(axes)
        for i, ax in enumerate(axes):
            ax.imshow(images_sort[i].reshape(28, 28), cmap='gray')
            ax.axis('off')

    plt.suptitle(title)
    plt.show()

# 2. Hebbian 学习规则
def train_hopfield_network(patterns,if_prth=0):
    """使用 Hebbian Learning 计算 Hopfield 网络的权重矩阵"""
    n_neurons = patterns.shape[1]  # 每个 pattern 的维度（像素数）


    

    print(np.shape(patterns))
    W = np.zeros((n_neurons, n_neurons))
    
    for pattern in patterns:
        pattern = pattern * 2 - 1  # 将二进制转换为 -1 和 1
        W += np.outer(pattern, pattern)
    
    np.fill_diagonal(W, 0)  # 自连接权重置为 0
    return W / patterns.shape[0]  # 归一化

# 3. 状态更新与能量函数
def update_state(input_pattern, W, steps=10):
    """逐步更新状态，尝试恢复原始模式"""
    n_neurons = len(input_pattern)
    state = input_pattern.copy()
    
    for _ in range(steps):
        for i in range(n_neurons):
            h = np.dot(W[i], state)  # 计算输入加权和
            state[i] = 1 if h >= 0 else -1  # 激活函数为符号函数
    
    return state

def calculate_energy(state, W):
    """计算 Hopfield 网络的能量函数"""
    state = state * 2 - 1  # 将二进制转换为 -1 和 1
    return -0.5 * np.dot(state, np.dot(W, state.T))

# 4. 测试网络
def test_hopfield_network(W, test_pattern, noise_level=0.005):
    noisy_pattern = test_pattern.copy()
    '''add noise'''
    n_noisy = int(noise_level * len(noisy_pattern))

    '''
    I must use different noise and maybe try to strengthen the noise
    '''


    noisy_indices = np.random.choice(len(noisy_pattern), n_noisy, replace=False)
    noisy_pattern[noisy_indices] = 1 - noisy_pattern[noisy_indices]  # 翻转噪声位
    '''get recover pattern'''
    recovered_pattern = update_state(noisy_pattern * 2 - 1, W, steps=10)
    return noisy_pattern, recovered_pattern

# 5. 主程序
if __name__ == "__main__":
    # 加载数据并展示原始样本
    n_samples=1
    patterns,labels = load_handwritten_digits(n_samples=n_samples,if_orth=0)

    
    #plot_images(patterns, title="Original Patterns",rows=1,cols=n_samples)
    
    # 训练 Hopfield 网络
    W = train_hopfield_network(patterns)
    
    test_patterns=[]
    noisy_patterns=[]
    recovered_patterns=[]

    # 测试恢复能力
    test_num=1
    for i in range(test_num):
        test_pattern = patterns[i]
        noisy_pattern, recovered_pattern = test_hopfield_network(W, test_pattern,noise_level=0.8)
        test_patterns.append(test_pattern)
        noisy_patterns.append(noisy_pattern)
        recovered_patterns.append(recovered_pattern)

    
        # 展示测试结果
    plot_images([test_patterns, noisy_patterns, recovered_patterns],
                title=f"Test Pattern | Noisy Pattern | Recovered Pattern for Test {i}", rows=test_num, cols=3,test_num=test_num)
    

