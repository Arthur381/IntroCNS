import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation  # 添加这行导入
from matplotlib.widgets import Button

# Pattern_Set class stays the same
class Pattern_Set:
    def __init__(self, num_neurons, num_patterns, sigma=5):
        self.train_order = ""  
        self.num_neurons = num_neurons
        self.num_patterns = num_patterns
        self.step = num_neurons // num_patterns
        self.whole_set = self.construct_patterns(sigma)
        self.left_idx = 0
        self.train_num = 0

    def choose_training_set_with_order(self, lef_index, type, train_num):
        self.train_order = type
        train = self.whole_set[lef_index:lef_index+train_num]
        self.left_idx = lef_index
        self.train_num = train_num
        if(type == "random"):
            random.shuffle(train)
        return train

    def change_amperage(self, amp=1):
        for pattern in self.whole_set:
            pattern *= amp

    def construct_patterns(self, sigma):
        patterns = []
        for i in range(self.num_patterns):
            center = i * self.step
            pattern = np.exp(-((np.arange(self.num_neurons) - center)**2) / (2 * sigma**2))
            pattern /= np.max(pattern)
            patterns.append(pattern)
        return patterns

class CANN1D(bp.dyn.NeuDyn):
    def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4., z_min=-5*bm.pi, z_max=5*bm.pi, **kwargs):
        super().__init__(size=num, **kwargs)

        self.tau = tau
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0

        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, num)
        self.rho = num / self.z_range
        self.dx = self.z_range / num
        self.num = num

        self.u = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.W = bm.zeros((num, num))
        
        self.integral = bp.odeint(self.derivative)

    def derivative(self, u, t, Irec, Iext):
        du = (-u + Irec + Iext) / self.tau
        return du

    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def orthogonal_hebb_learning(self, patterns):
        """
        Implement orthogonal Hebb learning rule with normalized vectors
        """
        n_patterns = len(patterns)
        n_neurons = len(patterns[0])
        
        # Convert patterns to array
        xi = np.array(patterns)  # ξ in the paper
        eta = np.zeros_like(xi)  # η in the paper
        eta_hat = np.zeros_like(xi)  # ηˆ in the paper
        
        # First pattern
        eta[0] = xi[0]
        eta_hat[0] = eta[0] / np.linalg.norm(eta[0])
        
        # Orthogonalize remaining patterns
        for p in range(1, n_patterns):
            eta[p] = xi[p]
            for mu in range(p):
                eta[p] = eta[p] - np.dot(eta_hat[mu], xi[p]) * eta_hat[mu]
            eta_hat[p] = eta[p] / np.linalg.norm(eta[p])
        
        # Construct weight matrix using normalized vectors
        W = np.zeros((n_neurons, n_neurons))
        for p in range(n_patterns):
            for i in range(n_neurons):
                for j in range(n_neurons):
                    if i == j:
                        W[i,j] += eta_hat[p,i] * eta_hat[p,j] - eta_hat[p,i] * eta_hat[p,i]
                    else:
                        W[i,j] += eta_hat[p,i] * eta_hat[p,j]
                        
        self.W = bm.asarray(W)

    def update(self, x=None):
        _t = bp.share['t']
        u2 = bm.square(self.u)
        r = u2 / (1.0 + self.k * bm.sum(u2))
        Irec = bm.dot(self.W, r)
        self.u[:] = self.integral(self.u, _t, Irec, self.input)
        self.input[:] = 0.

class Pattern_Experiment:
    def __init__(self, model:CANN1D, patterns:Pattern_Set, lef_index=12, type="", train_num=40):
        self.Model = model
        self.patterns = patterns
        train_set = patterns.choose_training_set_with_order(lef_index, type, train_num)
        self.Model.orthogonal_hebb_learning(train_set)

    def run_trace(self, Input, pattern_idx=0, visualize=False):
        # 重置网络状态
        self.Model.u[:] = 0.  # 重置membrane potential为0

        dur1, dur2 = 10., 10.
        I = Input
        Iext, duration = bp.inputs.section_input(values=[I, 0.],
                                        durations=[dur1, dur2],
                                        return_length=True)
        noise_level = 0.1
        noise = bm.random.normal(0., noise_level, (int(duration / bm.get_dt()), len(I)))
        Iext += noise

        runner = bp.DSRunner(self.Model, inputs=['input', Iext, 'iter'], monitors=[('u')])
        runner.run(duration)
        assert runner.mon.u.shape[1] == Iext.shape[1] == self.Model.x.shape[0], "Shape mismatch!"

        if visualize:
            # 创建图形和轴
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 获取数据范围以设置统一的轴范围
            y_min = min(np.min(runner.mon.u), np.min(Iext))
            y_max = max(np.max(runner.mon.u), np.max(Iext))
            y_range = y_max - y_min
            y_min -= 0.1 * y_range  # 添加一些边距
            y_max += 0.1 * y_range

            # 初始化两条线
            line_u, = ax.plot(self.Model.x, runner.mon.u[0], 'b-', label='Network State (u)')
            line_I, = ax.plot(self.Model.x, Iext[0], 'r--', label='Input')
            
            # 设置图形属性
            ax.set_xlim(self.Model.z_min, self.Model.z_max)
            ax.set_ylim(-1, 6)
            ax.set_xlabel('Position')
            ax.set_ylabel('Activity')
            ax.legend()
            ax.grid(True)
            
            frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
            plt.subplots_adjust(bottom=0.2)
            
            # 更新函数
            def update(frame):
                line_u.set_ydata(runner.mon.u[frame])
                line_I.set_ydata(Iext[frame])
                frame_text.set_text(f'Frame: {frame}, Pattern: {pattern_idx}, Time: {frame * bm.get_dt():.1f}ms')
                return line_u, line_I, frame_text
            
            # 创建动画
            ani = animation.FuncAnimation(fig, update, 
                                        frames=len(runner.mon.u), 
                                        interval=20, 
                                        blit=True)
            
            # 添加暂停按钮
            is_paused = False
            
            def pause(event):
                nonlocal is_paused
                if is_paused:
                    ani.event_source.start()
                else:
                    ani.event_source.stop()
                is_paused = not is_paused
            
            ax_pause = plt.axes([0.7, 0.05, 0.1, 0.075])
            btn_pause = Button(ax_pause, 'Pause/Resume')
            btn_pause.on_clicked(pause)
            
            plt.title(f'Pattern {pattern_idx} Dynamics')
            plt.show()

        return runner.mon.u[-1]

    def Testing(self):
        E_list = []
        whole_patterns = self.patterns.whole_set
        for i, input_pattern in enumerate(whole_patterns):
            u = self.run_trace(input_pattern, i)  # No visualization during testing
            normalized_u = u / np.linalg.norm(u)
            E = -0.5 * (normalized_u.T @ self.Model.W @ normalized_u)
            #E = -0.5 * (u.T @ self.Model.W @ u) 
            E_list.append(float(E))
        return E_list

    def plot_show(self, E_list):
        index = list(range(len(E_list)))
        plt.figure(figsize=(10, 6))
        plt.plot(index, E_list, label='Energy', marker='o', linestyle='-')
        plt.xticks(ticks=[i for i in range(0, len(index)+1, 8)], 
                  labels=[str(i) for i in range(0, len(index)+1, 8)])
        plt.xlabel('Pattern Index')
        plt.ylabel('Energy')
        plt.title(f'Orthogonal Hebb Learning\nNeuron num {self.Model.num}, '
                 f'Pattern num {self.patterns.num_patterns}, '
                 f'Train index [{self.patterns.left_idx},{self.patterns.left_idx+self.patterns.train_num}]')
        plt.legend()
        plt.grid(True)
        plt.show()

def Persistent_Activity():
    # 初始化CANN和Patterns
    cann = CANN1D(num=512, k=0.1, J0=4.0)
    Patterns = Pattern_Set(num_neurons=512, num_patterns=160)
    pre_dis = np.max(cann.W) - np.min(cann.W)
    
    # 创建experiment实例
    exp1 = Pattern_Experiment(cann, Patterns, lef_index=50, type="", train_num=60)
    
    # 选择两个pattern进行可视化展示
    pattern_indices = []  
    print("Demonstrating dynamics for selected patterns...")
    for idx in pattern_indices:
        input_pattern = Patterns.whole_set[idx]
        print(f"\nVisualizing Pattern {idx}")
        exp1.run_trace(input_pattern, pattern_idx=idx, visualize=True)
    
    # 运行完整测试并显示能量图
    print("\nRunning full test...")
    E_list = exp1.Testing()
    exp1.plot_show(E_list)
    
    print(f"\nPrevious max distance: {pre_dis}")
    print(f"After training, distance: {float(bm.max(cann.W) - bm.min(cann.W))}")



if __name__ == "__main__":
    Persistent_Activity()
