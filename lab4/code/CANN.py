import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation





class CANN1D(bp.dyn.NeuDyn):
  def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4., z_min=-5*bm.pi, z_max=5*bm.pi, **kwargs):
    super().__init__(size=num, **kwargs)

    # 1、初始化参数
    self.tau = tau
    self.k = k
    self.a = a
    self.A = A
    self.J0 = J0

    # 2、初始化特征空间相关参数
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)
    self.rho = num / self.z_range
    self.dx = self.z_range / num

    # 3、初始化变量
    self.u = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))
    self.conn_mat = self.make_conn(self.x)  # 连接矩阵

    # 4、定义积分函数
    self.integral = bp.odeint(self.derivative)

  # 微分方程
  def derivative(self, u, t, Irec, Iext):
    # TODO：定义微分方程？？？ But Why don't we consider dt
    du = (-u + Irec + Iext) / self.tau
    return du

  # 6、将距离转换到[-z_range/2, z_range/2)之间
  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    # TODO：实现一维的环的距离度量？？？
    
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
      '''
      更新网络状态。

      参数：
          x (optional): 外部输入。如果未提供，默认为 None。

      返回：
          None
      '''
      # 获取当前时间
      _t = bp.share['t']
      # 计算平方后的神经元激活值
      u2 = bm.square(self.u)
      # 计算归一化的激活值
      r = u2 / (1.0 + self.k * bm.sum(u2))
      # 计算网络的递归输入
      Irec = bm.dot(self.conn_mat, r)
      print(np.shape(Irec))
      # 更新神经元状态
      self.u[:] = self.integral(self.u, _t, Irec, self.input)
      # 重置外部输入
      self.input[:] = 0.




def Persistent_Activity(k=0.1,J0=0.5):
    # 初始化一个CANN
    cann = CANN1D(num=512, k=k, J0=J0)
    #print(cann.conn_mat)
    # 生成外部刺激，从第2到12ms，持续10ms
    dur1, dur2, dur3 = 2., 8., 18.
    I1 = cann.get_stimulus_by_pos(3.)#返回一个二维矩阵
    I2=cann.get_stimulus_by_pos(-3.)
    Iext, duration = bp.inputs.section_input(values=[0., I1+I2, 0.],
                                             durations=[dur1, dur2, dur3],
                                             return_length=True)
    noise_level = 0.1
    noise = bm.random.normal(0., noise_level, (int(duration / bm.get_dt()), len(I1)))
    Iext += noise

    # TODO：运行数值模拟，监控变量u的历史变化？？
    runner = bp.DSRunner(cann, inputs=['input', Iext, 'iter'], monitors=[('u')])
    runner.run(duration)

    # 可视化
    def plot_response(t):
        fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
        ax = fig.add_subplot(gs[0, 0])
        ts = int(t / bm.get_dt())
        #print(np.shape(Iext),f"u.shape={np.shape(runner.mon.u)}",np.shape(cann.x),f"dt={bm.get_dt()}",f"cann.u={np.shape(cann.u)}")
        I, u = Iext[ts], runner.mon.u[ts]
        ax.plot(cann.x, I, label='Iext')
        ax.plot(cann.x, u, linestyle='dashed', label='U')
        ax.set_title(r'$t$' + ' = {} ms'.format(t))
        ax.set_xlabel(r'$x$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        # plt.savefig(f'CANN_t={t}.pdf', transparent=True, dpi=500)
    #plot_response(t=1.)
    #print(cann.conn_mat)
    plot_response(t=10.)
    plot_response(t=18.)
    plot_response(t=25.)
    assert runner.mon.u.shape[1] == Iext.shape[1] == cann.x.shape[0], "Shape mismatch!"

    save_path="../picture/2simi_2.gif"
    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
        frame_step=1,
        frame_delay=40,
        #save_path=save_path,  # 添加保存路径
        show=True,
    )
    plt.show()
# 创建子图
Persistent_Activity(k=0.1)

