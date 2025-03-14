import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
class CANN1D(bp.dyn.NeuDyn):
    def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4., z_min=-bm.pi, z_max=bm.pi, **kwargs):
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
        du = (-u+Irec+Iext) / self.tau
        return du

    # 6、将距离转换到[-z_range/2, z_range/2)之间
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
        _t = bp.share['t']
        u2 = bm.square(self.u)
        r = u2 / (1.0 + self.k * bm.sum(u2))
        Irec = bm.dot(self.conn_mat, r)
        self.u[:] = self.integral(self.u, _t,Irec, self.input)
        self.input[:] = 0.  # 重置外部电流

def Persistent_Activity(I_positions,k=0.1,J0=1.,plot=False):
    # 初始化一个CANN
    cann = CANN1D(num=512, k=k, J0=J0)

    # 生成外部刺激，从第2到12ms，持续10ms
    dur1, dur2, dur3 = 2., 10., 20.
    I_list=map(cann.get_stimulus_by_pos, I_positions)
    I=sum(I_list)
    Iext, duration = bp.inputs.section_input(values=[0., I, 0.],
                                                durations=[dur1, dur2, dur3],
                                                return_length=True)
    noise_level = 0.1
    noise = bm.random.normal(0., noise_level, (int(duration / bm.get_dt()), len(I)))
    Iext += noise

    runner = bp.DSRunner(cann, inputs=['input', Iext, 'iter'], monitors=['u'])
    runner.run(duration)

    # 可视化
    def plot_response(t):
        fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
        ax = fig.add_subplot(gs[0, 0])
        ts = int(t / bm.get_dt())
        I, u = Iext[ts], runner.mon.u[ts]
        ax.plot(cann.x, I, label='Iext')
        ax.plot(cann.x, u, linestyle='dashed', label='U')
        ax.set_title(r'$t$' + ' = {} ms'.format(t))
        ax.set_xlabel(r'$x$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        # plt.savefig(f'CANN_t={t}.pdf', transparent=True, dpi=500)
    if plot:
        plot_response(t=10.)
        plot_response(t=20.)
        plot_response(t=30.)

        bp.visualize.animate_1D(
            dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                            {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
            frame_step=1,
            frame_delay=40,
            show=True,
        )
        plt.show()
    return Iext, runner.mon.u

#Persistent_Activity([0,bm.pi],k=1,plot=True)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp

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
    return (-u+Irec+Iext-v) / self.tau

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

def anticipative_tracking(I_time, k=8.1, J0=4.,m=10,v_ext=[0,6*1e-3],plot=False, plot_time=None):
    # 初始化一个CANN
    num=512
    cann_sfa = CANN1D_SFA(num=num, m=m,k=k,J0=J0)
    position = np.zeros(int(sum(I_time)/bm.get_dt()))
    time_before=0
    for index in range(len(I_time)):
        time=int(I_time[index]/bm.get_dt())
        v_ext_i=v_ext[index]
        for i in range(time_before,time_before+time):
            pos = position[i-1]+v_ext_i*bm.dt
            # the periodical boundary
            pos = np.where(pos>np.pi, pos-2*np.pi, pos)
            pos = np.where(pos<-np.pi, pos+2*np.pi, pos)
            # update
            position[i] = pos
        time_before+=time
    position = position.reshape((-1, 1))
    Iext = cann_sfa.get_stimulus_by_pos(position)
        
    runner = bp.DSRunner(cann_sfa, inputs=['input', Iext, 'iter'], monitors=['u'])
    runner.run(sum(I_time))

    theta=cann_sfa.x
    # 将角度转换为笛卡尔坐标 (x, y)
    xc = np.cos(theta)
    yc = np.sin(theta)
    x_center = np.sum(runner.mon.u * xc,axis=1) / np.sum(runner.mon.u,axis=1)  # x方向的加权平均
    y_center = np.sum(runner.mon.u * yc,axis=1) / np.sum(runner.mon.u,axis=1)  # y方向的加权平均
    # 从笛卡尔坐标计算质心的角度
    ucenter = np.arctan2(y_center, x_center)
    # 可视化
    def plot_response(t):
        fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
        ax = fig.add_subplot(gs[0, 0])
        ts = int(t / bm.get_dt())
        I, u = Iext[ts], runner.mon.u[ts]

        ax.plot(cann_sfa.x, I, label='Iext')
        ax.plot(cann_sfa.x, u, linestyle='dashed', label='U')
        ax.scatter(ucenter[ts], np.max(u)/2, color='red', zorder=5, label="Center Point")

        ax.set_title(r'$t$' + ' = {} ms'.format(t))
        ax.set_xlabel(r'$x$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        # plt.savefig(f'CANN_t={t}.pdf', transparent=True, dpi=500)
    if plot:
        if plot_time is None:
            # 如果没有提供 plot_time，使用默认时间范围或选择一个合理值
            plot_time = np.linspace(0, sum(I_time), len(I_time))
        list(map(plot_response, plot_time))
        temp=np.zeros((len(ucenter),len(cann_sfa.x)))
        for i, center in enumerate(ucenter):
            indices = ((center + np.pi) / (2 * np.pi) * 511).astype(int)
            indices=np.clip(indices, 0, 511)
            temp[i,indices] = 10  # 根据 ucenter 中的位置修改 temp 中的值


        bp.visualize.animate_1D(
            dynamical_vars=[{'ys': runner.mon.u, 'xs': cann_sfa.x, 'legend': 'u'},
                            {'ys': Iext, 'xs': cann_sfa.x, 'legend': 'Iext'},
                            {'ys': temp, 'xs': cann_sfa.x, 'legend': 'ucenter'}],
            frame_step=1,
            frame_delay=40,
            show=True,
            #save_path=f'move_stimuli,m={m}'

        )
        plt.show()
    return Iext, runner.mon.u, ucenter

""" u_center=[]
for m_o in range(1,201):
    m=m_o/10
    Iext, u = anticipative_tracking([20,100],v_ext=[0,0.06],m=m,plot=False)
    u_center.append(np.mean(u*np.linspace(-bm.pi,bm.pi,512),axis=1))



y = np.linspace(1, 200, 200)  # X 轴坐标
x = np.linspace(1, 1200, 1200)  # Y 轴坐标
X, Y = np.meshgrid(x, y)     # 网格化 X 和 Y
Z = np.array(u_center)  # Z 值（函数值）


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

fig.colorbar(surf, shrink=0.5, aspect=10)

ax.set_title("3D Surface Plot")
ax.set_xlabel("Time")
ax.set_ylabel("m")
ax.set_zlabel("u_center")

plt.show() """


#travelling
Iext, u, ucenter = anticipative_tracking([2,100],v_ext=[0,0.06],m=70,k=0.1,plot=True,plot_time=[10])
plt.plot(ucenter[500:900])
plt.show()
Iext, u, ucenter = anticipative_tracking([2,100],v_ext=[0,0.06],m=20,k=0.1,plot=True,plot_time=[10])
plt.plot(ucenter[500:900])
plt.show()
#smooth
#anticipative_tracking([2,1000],v_ext=[0,0.06],m=10,k=8.1,plot=True,plot_time=[10,100,150])
#oscilattory
#anticipative_tracking([2,1000],v_ext=[0,0.06],m=0.5,k=0.1,plot=True,plot_time=[10,100,150])