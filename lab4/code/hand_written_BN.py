from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 选择前 10 张图片并二值化（阈值设为 128）
binary_images = (x_train[:10] > 128).astype(int)

# 显示其中一张图片
plt.imshow(binary_images[0], cmap='gray')
plt.show()
