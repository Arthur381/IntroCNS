import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


class CANN1D_SFA(bp.dyn.NeuDyn):
  def __init__(self, num, m = 0.1, tau=1., tau_v=10., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, **kwargs):
    super(CANN1D_SFA, self).__init__(size=num, **kwargs)

    # 1、初始化参数
    self.tau = tau
    self.tau_v = tau_v #time constant of SFA
    self.k = k
    self.a = a
    self.A = A
    self.J0 = J0
    self.m = m #SFA strength
      
    # 2、初始化特征空间相关参数
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)
    self.rho = num / self.z_range
    self.dx = self.z_range / num

    # 3、初始化变量
    self.u = bm.Variable(bm.zeros(num))
    self.v = bm.Variable(bm.zeros(num)) #SFA current
    self.input = bm.Variable(bm.zeros(num))
    self.conn_mat = self.make_conn(self.x)  # 连接矩阵

    # 4、定义积分函数
    self.integral = bp.odeint(bp.JointEq(self.du, self.dv))

  # 微分方程
  def du(self, u, t, v, Irec, Iext):
    # TODO: 定义u的微分方程
    return (-u + Irec + Iext-v) / self.tau

  def dv(self, v, t, u):
    # TODO: 定义v的微分方程
    return (-v+self.m*u) / self.tau_v

  # 5、将距离转换到[-z_range/2, z_range/2)之间
  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  # 计算连接矩阵
  def make_conn(self, x):
    assert bm.ndim(x) == 1
    d = self.dist(x - x[:, None])  # 距离矩阵
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a) 
    return Jxx

  # 6、获取各个神经元到pos处神经元的输入
  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  # 7、网络更新函数
  def update(self, x=None):
    u2 = bm.square(self.u)
    r = u2 / (1.0 + self.k * bm.sum(u2))
    Irec = bm.dot(self.conn_mat, r)
    u, v = self.integral(self.u, self.v, bp.share['t'],Irec, self.input)
    self.u[:] = bm.where(u>0,u,0)
    self.v[:] = v
    self.input[:] = 0.  # 重置外部电流


def anticipative_tracking(m=10,v_ext=6*1e-3):
    cann_sfa = CANN1D_SFA(num=512, m=m)
    """
    预期追踪函数：模拟并可视化一个连续吸引子神经网络（CANN）模型的预期追踪行为。

    参数：
        m (float)：模型中的一个参数，用于调整神经元的活动。
        v_ext (float)：外部刺激的速度。

    返回：
        无

    注释：
        1. 创建一个包含512个神经元的CANN模型实例。
        2. 定义一个随时间变化的外部刺激，初始持续时间为10ms，随后持续1000ms。
        3. 计算外部刺激的位置，模拟刺激在神经元空间中的移动。
        4. 使用DSRunner运行模拟，记录神经元的活动。
        5. 可视化模拟结果，包括不同时间点的神经元活动和外部刺激。
        6. 生成一个动画，展示神经元活动和外部刺激随时间的变化。
    """
    # 定义随时间变化的外部刺激
    v_ext = v_ext
    dur1, dur2, = 10., 1000.
    num1 = int(dur1 / bm.get_dt())
    num2 = int(dur2 / bm.get_dt())
    position = np.zeros(num1 + num2)
    for i in range(num2):
        pos = position[i+num1-1]+v_ext*bm.dt
        # the periodical boundary
        pos = np.where(pos>np.pi, pos-2*np.pi, pos)
        pos = np.where(pos<-np.pi, pos+2*np.pi, pos)
        # update
        position[i+num1] = pos
    position = position.reshape((-1, 1))
    Iext = cann_sfa.get_stimulus_by_pos(position)

    # TODO：任务1 - 创建DSRunner，运行模拟
    runner = bp.DSRunner(cann_sfa, inputs=['input', Iext, 'iter'], monitors=[('u')])
    runner.run(dur1 + dur2)

    # 可视化
    def plot_response(t, extra_fun=None):
        fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
        ax = fig.add_subplot(gs[0, 0])
        ts = int(t / bm.get_dt())
        I, u = Iext[ts], runner.mon.u[ts]
        ax.plot(cann_sfa.x, I, label='Iext')
        ax.plot(cann_sfa.x, 10*u, linestyle='dashed', label='U')
        ax.set_title(r'$t$' + ' = {} ms'.format(t))
        ax.set_xlabel(r'$x$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    plot_response(t=10.)
    plot_response(t=200.)
    plot_response(t=400.)
    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': runner.mon.u, 'xs': cann_sfa.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': cann_sfa.x, 'legend': 'Iext'}],
        frame_step=5,
        frame_delay=50,
        show=True,
    )
    plt.show()


anticipative_tracking(m=50,v_ext=0.1)

# TODO: 任务2，改变输入移动的快慢
#anticipative_tracking(?????)

#2014