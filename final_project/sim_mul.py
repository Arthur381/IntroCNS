import numpy as np
import matplotlib.pyplot as plt

# 1. 生成拟合数据
x = np.linspace(-1, 1, 100)  # 输入数据
y = 1 / (1 + np.exp(-10 * x))  # Sigmoid 函数

# 2. 定义多项式拟合的阶数
degrees = [3, 5, 8,12]  # 多项式阶数列表
polynomial_fits = {}

# 3. 对每个阶数进行拟合
for degree in degrees:
    coefficients = np.polyfit(x, y, degree)  # 拟合多项式系数
    polynomial_fits[degree] = coefficients  # 存储结果

# 4. 绘制原始函数和拟合曲线
plt.figure(figsize=(10, 6))

# 原始函数
plt.plot(x, y, label="Sigmoid Function", color="blue", linewidth=2)

# 多项式拟合曲线
for degree, coefficients in polynomial_fits.items():
    y_poly = np.polyval(coefficients, x)  # 计算多项式值
    plt.plot(x, y_poly, label=f"Polynomial Degree {degree}", linestyle="--")

# 设置图例和标题
plt.title("Sigmoid Function and Polynomial Approximations", fontsize=14)
plt.xlabel("Input", fontsize=12)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.grid()
plt.show()
