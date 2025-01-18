import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import random
import sys



class Pattern_Set:
  def __init__(self,num_neurons, num_patterns,sigma=5):

    self.train_order="" # left to right or random
    self.num_neurons=num_neurons
    self.num_patterns=num_patterns
    self.step=num_neurons//num_patterns
    self.whole_set=self.construct_patterns(sigma)
    self.left_idx=0
    self.train_num=0

  def choose_training_set_with_order(self,lef_index,type,train_num):
      self.train_order=type
      train=self.whole_set[lef_index:lef_index+train_num]
      self.left_idx=lef_index
      self.train_num=train_num
      if(type=="ramdom"):
          random.shuffle(train)
      
      return train 

  def change_amperage(self,amp=1):
      for pattern in self.whole_set:
          pattern*=amp

  def construct_patterns(self,sigma):
      patterns = []
      for i in range(self.num_patterns):
          # 高斯中心位置
          center = i * self.step
          pattern = np.exp(-((np.arange(self.num_neurons) - center)**2) / (2 * sigma**2))
          pattern /= np.max(pattern)
          patterns.append(pattern)

      return patterns



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
    self.num=num
    # 3、初始化变量
    self.u = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))
    self.W = self.make_conn(self.x)  # 连接矩阵

    # 4、定义积分函数
    self.integral = bp.odeint(self.derivative)

  # 微分方程
  def derivative(self, u, t, Irec, Iext):
    du = (-u + Irec + Iext) / self.tau
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
      # 获取当前时间
      _t = bp.share['t']
      # 计算平方后的神经元激活值
      u2 = bm.square(self.u)
      # 计算归一化的激活值
      r = u2 / (1.0 + self.k * bm.sum(u2))
      # 计算网络的递归输入,是一个向量
      Irec = bm.dot(self.W, r)
      # 更新神经元状态
      self.u[:] = self.integral(self.u, _t, Irec, self.input)
      # 重置外部输入
      self.input[:] = 0.
    
  def Hebb_Rule_Update(self,voltage,lr):
      u2 = bm.square(voltage)
       # 归一化发放率
      r = u2 / (1.0 + self.k * bm.sum(u2))
    # 计算权重更新 (外积计算 r_i * r_j)
      delta_W = lr * bm.outer(r, r)
    # 更新权重矩阵
      self.W += delta_W


class Pattern_Experiment:

    def __init__(self,model:CANN1D,patterns:Pattern_Set,lef_index=12,type="",train_num=40,lr=0.01):
        self.Model=model
        self.patterns=patterns
        self.train_set=patterns.choose_training_set_with_order(lef_index,type,train_num)
        self.lr=lr

    def run_trace(self,Input):
        dur1, dur2= 2., 10.
        I = Input#返回一个二维矩阵
        Iext, duration = bp.inputs.section_input(values=[I, 0.],
                                                durations=[dur1, dur2],
                                                return_length=True)
        noise_level = 0.1
        noise = bm.random.normal(0., noise_level, (int(duration / bm.get_dt()), len(I)))
        Iext += noise

        runner = bp.DSRunner(self.Model, inputs=['input', Iext, 'iter'], monitors=[('u')])
        runner.run(duration)
        assert runner.mon.u.shape[1] == Iext.shape[1] == self.Model.x.shape[0], "Shape mismatch!"
        u=runner.mon.u[-1]
        return u


    def Training(self):
      # 生成外部刺激，从第2到12ms，持续10ms
      for input_pattern in self.train_set:
        u=self.run_trace(input_pattern)
        self.Model.Hebb_Rule_Update(u,self.lr)


    def Testing(self):
        E_list=list()
        whole_patterns=self.patterns.whole_set
        for input_pattern in whole_patterns:
            u=self.run_trace(input_pattern)
            normalized_u = u / np.linalg.norm(u)
            E=-0.5*(normalized_u.T@self.Model.W@normalized_u)
            E_list.append(E)
        return E_list
    
    def plot_show(self,E_list):
        index = list(range(len(E_list)))

        # 绘制能量与索引的关系图
        plt.figure(figsize=(10, 6))
        plt.plot(index, E_list, label='Energy', marker='o', linestyle='-')
        
        # 设置x轴刻度每8个显示一次
        plt.xticks(ticks=[i for i in range(0, len(index)+1, 8)], labels=[str(i) for i in range(0, len(index)+1, 8)])
        
        plt.xlabel('Index')
        plt.ylabel('Energy')
        plt.title(f'Hebb Learning Rate {self.lr}\nNeuron num {self.Model.num}, Pattern num {self.patterns.num_patterns}, Train index [{self.patterns.left_idx},{self.patterns.left_idx+self.patterns.train_num}]')
        plt.legend()
        plt.grid(True)
        plt.show()

class Visualize_Experiment:
    def __init__(self,model:CANN1D,lr=0.01,learn_pos=0,learn_iter=1):
        '''
        iter: total number of ext input
        lr: learning rate
        learn_pos: the postion to apply Hebb Rule and the first input at
        
        '''
        self.voltage_list=[]
        self.Iext_list=[]
        self.iteration_times=learn_iter
        self.lr=lr
        self.learn_pos=learn_pos
        self.learn_iter=learn_iter
        self.Model=model
        self.learn()

    def learn(self):
        for i in range(self.learn_iter):
            dur1, dur2, dur3 = 2., 5., 5.
            I1 = self.Model.get_stimulus_by_pos(self.learn_pos)#返回一个二维矩阵
            Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                                    durations=[dur1, dur2, dur3],
                                                    return_length=True)
            noise_level = 0.1
            noise = bm.random.normal(0., noise_level, (int(duration / bm.get_dt()), len(I1)))
            Iext += noise
            self.Iext_list.append(Iext)
            runner = bp.DSRunner(self.Model, inputs=['input', Iext, 'iter'], monitors=[('u')])
            runner.run(duration)
            u=runner.mon.u[-1]
            print(f"Learning Time {i}, voltage is {max(u)}")
            self.voltage_list.append(runner.mon.u)
            self.Model.Hebb_Rule_Update(u,self.lr)

    def test(self,pos):
        self.iteration_times+=1
        dur2, dur3 =  2., 5.
        I1 = self.Model.get_stimulus_by_pos(pos)#返回一个二维矩阵
        Iext, duration = bp.inputs.section_input(values=[ I1, 0.],
                                                durations=[ dur2, dur3],
                                                return_length=True)
        noise_level = 0.1
        noise = bm.random.normal(0., noise_level, (int(duration / bm.get_dt()), len(I1)))
        Iext += noise
        self.Iext_list.append(Iext)
        runner = bp.DSRunner(self.Model, inputs=['input', Iext, 'iter'], monitors=[('u')])
        runner.run(duration)
        self.voltage_list.append(runner.mon.u)

    def dy_plot(self,save_path=None):
        Iext=[item for sublist in self.Iext_list[-3:] for item in sublist]
        u=[item for sublist in self.voltage_list[-3:] for item in sublist]
        bp.visualize.animate_1D(
              dynamical_vars=[{'ys': u, 'xs': self.Model.x, 'legend': 'u'},
                              {'ys': Iext, 'xs': self.Model.x, 'legend': 'Iext'}],
              frame_step=1,
              frame_delay=10,
              save_path=save_path,  # 添加保存路径
              show=True,
          )

def Persistent_Activity():
    # 初始化一个CANN
    cann = CANN1D(num=512, k=0.1, J0=0.5)
    Patterns=Pattern_Set(num_neurons=512,num_patterns=160)
    pre_dis=np.max(cann.W)-np.min(cann.W)
    '''Pattern experiment'''
    '''
    exp1=Pattern_Experiment(cann,Patterns,lef_index=50,type="",train_num=60,lr=0.01)
    exp1.Training()
    E_list=exp1.Testing()
    exp1.plot_show(E_list)
    print(f"previous max distance is {pre_dis}",f"after training, distances is {np.max(cann.W)-np.min(cann.W)}")
    '''
    
    
    '''Visualize different location experiment'''
    vis_experiment=Visualize_Experiment(cann,learn_iter=8,lr=0.3)
    vis_experiment.test(2)
    vis_experiment.dy_plot(save_path="animation_5_0.3.gif")
    sys.exit()  # 程序退出
Persistent_Activity()
