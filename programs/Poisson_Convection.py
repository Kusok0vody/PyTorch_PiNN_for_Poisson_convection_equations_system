import numpy as np 
import os
import torch
import time

from torch.autograd import Variable
from texttable import Texttable
from copy import deepcopy, copy

import programs.conditions as cnd
import programs.misc as misc
from programs.NN import *


class Poisson_Convection:
    def __init__(self,
                 w: np.float64,
                 mu0: np.float64,
                 cmax: np.float64,
                 v_in: np.float64,
                 chi: np.float64,
                 transitional_times: list,
                 size: list,
                 c_cond: list,
                 p_cond: list,
                 PDE_points: int,
                 BC_points: int,
                 IC_points: int,
                 beta: np.float64 = -2.5):

        # CPU/GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        # Дебаг для производной
        torch.autograd.set_detect_anomaly(False)
        
        # Создание модели
        self.model = Net(input_size=3,
                         neurons_arr=[48,48,48,48,48],
                         output_size=3,
                         depth=4,
                         act=Sin)

        # Итерационные переменные
        self.Adam_epochs = 2000
        self.losses      = []
        self.epoch       = 0
        self.k           = 10

        # Переменные числа точек
        self.PDE_points = PDE_points
        self.BC_points  = BC_points
        self.BC_points2 = self.BC_points**2
        self.IC_points  = IC_points
        
        # Вспомогаительные тензоры 0 и 1
        self.zeros = torch.FloatTensor([0]).to(self.device)
        self.ones  = torch.FloatTensor([1]).to(self.device)

        # Параметры для обучения
        self.criterion = torch.nn.MSELoss()
        self.weights   = [1,1,1,1,1,1]
        self.optimizer = self.model.set_optimizer('Adam')

        # Параметры для Cirriculum Learning
        self.CL_epochs = []
        self.CL        = False
        self.const     = 0
        
        # Физические Константты
        self.w    = w
        self.mu0  = mu0
        self.beta = beta
        self.cmax = cmax
        self.v_in = v_in
        
        # Граничные условия
        self.c_cond = deepcopy(c_cond)
        self.p_cond = deepcopy(p_cond)
        
        # Геометрия
        self.chi  = chi
        self.size = deepcopy(size)
        self.times = np.concatenate(([self.size[-2]], transitional_times, [self.size[-1]]))
        self.a    = (self.size[3] - self.chi)/2
        self.b    = (self.size[3] + self.chi)/2

        # Создание сеток координат, массивов НУ и ГУ
        self.makeIBC()
        self.PDE_coords()

    def full_load(self, path_nn:str, path_data:str):
        """
        Loads NN and other parameters (temporarily only losses) from file
        """
        self.model.load_state_dict(torch.load(path_nn,map_location=torch.device('cpu')))
        data = np.load(path_data)
        self.losses = data[0].tolist()


    def full_save(self, path_nn:str, path_data:str):
        """
        Saves NN and other parameters (temporarily only losses) from file
        """
        torch.save(self.model.state_dict(), path_nn)
        data = [self.losses]
        np.save(path_data, data)


    def makeIBC(self):
        """
        Makes initial and boundary conditions
        """

        # Тензоры точек "разрыва"
        y_change = torch.FloatTensor([self.a, self.b]).to(self.device)
        t_change = torch.FloatTensor(self.times[1:-1]).to(self.device)
        
        # Создание массивов для начального условия
        x = torch.linspace(self.size[0], self.size[1], self.IC_points).to(self.device)
        y = torch.cat((torch.linspace(self.size[2], self.size[3], self.IC_points-len(self.times[1:-1])).to(self.device), y_change)).sort()[0]
    
        XYT = torch.stack(torch.meshgrid(x, y, self.zeros, indexing='ij')).reshape(3, -1)
        self.x_IC = Variable(XYT[0], requires_grad=True)
        self.y_IC = Variable(XYT[1], requires_grad=True)
        self.t_IC = Variable(XYT[2], requires_grad=True)

        left_cond = self.x_IC==0
        psi_cond  = misc.psi(self.y_IC, self.chi)
        self.c_IC = torch.where(left_cond+psi_cond==2*True, self.c_cond[0], 0.)
        
        # Создание массивов для граничных условий
        x = torch.linspace(self.size[0], self.size[1], self.BC_points).to(self.device)
        y = torch.cat((torch.linspace(self.size[2], self.size[3], self.BC_points-2).to(self.device), y_change)).sort()[0]
        t = torch.cat((torch.linspace(self.size[4], self.size[5], self.BC_points-len(self.times[1:-1])).to(self.device), t_change)).sort()[0]

        # Интервал перфорации
        psi = misc.psi_th(y, self.a, self.b)
        psi = torch.stack(torch.meshgrid(psi, torch.zeros(self.BC_points), indexing='ij')).reshape(2, -1).T[:,0]

        # ГУ для концентрации
        zeros = torch.zeros(4*len(psi)).reshape(4, len(psi))
        c_condition = cnd.form_boundaries([x, y, t], zeros, self.ones, self.zeros)
        self.c = c_condition[:,0]
        self.x = c_condition[:,1]
        self.y = c_condition[:,2]
        self.t = c_condition[:,3]
        
        self.p = cnd.form_boundaries([x, y, t], zeros, self.ones, self.zeros)[:,0]

        left_side_cond = self.x==self.size[0]
        psi_cond = misc.psi(self.y, self.chi)
        for i in range(len(self.times)-1):
            time_start_cond = torch.where(self.t>=self.times[i], 1., 0.)
            time_end_cond = torch.where(self.t<=self.times[i+1], 1., 0.)
            self.c = torch.where(time_start_cond + 
                                 time_end_cond +
                                 left_side_cond +
                                 psi_cond == 4*True,
                                 self.c_cond[i], self.c)
            
            self.p = torch.where(time_start_cond + 
                                 time_end_cond +
                                 left_side_cond +
                                 psi_cond == 4*True,
                                 self.p_cond[i+1], self.p) 
        right_side_cond = self.x==self.size[1]
        self.p = torch.where(right_side_cond+psi_cond==2*True, self.p_cond[0], self.p)
        
        # self.c = Variable(self.c, requires_grad=True)
        # self.p = Variable(self.p, requires_grad=True)

        self.x_tb = Variable(self.x[:2*self.BC_points2], requires_grad=True)
        self.y_tb = Variable(self.y[:2*self.BC_points2], requires_grad=True)
        self.t_tb = Variable(self.t[:2*self.BC_points2], requires_grad=True)

        self.x_lr = Variable(self.x[2*self.BC_points2:], requires_grad=True)
        self.y_lr = Variable(self.y[2*self.BC_points2:], requires_grad=True)
        self.t_lr = Variable(self.t[2*self.BC_points2:], requires_grad=True)


    def PDE_coords(self):
        # Тензоры точек "разрыва"
        y_change = torch.FloatTensor([self.a, self.b]).to(self.device)
        t_change = torch.FloatTensor(self.times[1:-1]).to(self.device)
        # Создание координат для ДУЧП
        # x_collocation = torch.linspace(self.size[0], self.size[1], self.PDE_points).to(self.device)
        # y_collocation = torch.cat((torch.linspace(self.size[2], self.size[3], self.PDE_points-2).to(self.device), y_change)).sort()[0]
        # t_collocation = torch.cat((torch.linspace(self.size[4], self.size[5], self.PDE_points-len(self.times[1:-1])).to(self.device), t_change)).sort()[0]

        x_collocation = torch.cat((self.zeros, torch.FloatTensor(self.PDE_points-2).to(self.device).uniform_(self.size[0], self.size[1]), self.ones)).sort()[0]
        y_collocation = torch.cat((self.zeros, torch.FloatTensor(self.PDE_points-4).to(self.device).uniform_(self.size[2], self.size[3]), self.ones, y_change)).sort()[0]
        t_collocation = torch.cat((self.zeros, torch.FloatTensor(self.PDE_points-2-len(self.times[1:-1])).to(self.device).uniform_(self.size[2], self.size[3]),
                                   self.ones, t_change)).sort()[0]
        
        XYT = torch.stack(torch.meshgrid(x_collocation, y_collocation, t_collocation, indexing='ij')).reshape(3, -1)
        self.X = Variable(XYT[0], requires_grad=True)
        self.Y = Variable(XYT[1], requires_grad=True)
        self.T = Variable(XYT[2], requires_grad=True)


    def update(self):
        self.makeIBC()
        self.PDE_coords()

    
    def transform(self, net):
        a = self.cmax * torch.sigmoid(500*(net[:,0].clone() - self.cmax))
        net[:,0] *= torch.sigmoid(500*net[:,0]) + torch.sigmoid(500*(self.cmax - net[:,0])) - 1
        net[:,0] += a
        net[:,0] *= 0.9988
        net[:,0] += 1/1700

        
    def PDELoss(self):
        """
        Calculates the loss from PDE
        """
        u = self.model([self.X,self.Y,self.T], self.transform)
        
        c  = u[:,0]
        px = u[:,1]
        py = u[:,2]
        
        mu =  (1.0 - c / self.cmax).pow(self.beta)

        u_x = -self.w**2 * px / (12. * self.mu0 * mu)
        u_y = -self.w**2 * py / (12. * self.mu0 * mu)

        c_x = u_x * self.model.derivative(c, self.X)
        c_y = u_y * self.model.derivative(c, self.Y)
        c_t =       self.model.derivative(c, self.T)

        u_xx = self.model.derivative(u_x, self.X)
        u_yy = self.model.derivative(u_y, self.Y)

        p_xy = self.model.derivative(px, self.Y)
        p_yx = self.model.derivative(py, self.X)

        u_ = u_xx + u_yy
        c_ = c_t + c_x + c_y
        
        loss_p   = self.criterion(u_, torch.zeros_like(u_))
        loss_c   = self.criterion(c_, torch.zeros_like(c_))
        loss_corr = self.weights[2] * self.criterion(p_xy-p_yx, torch.zeros_like(p_xy))

        loss = self.weights[0] * loss_c + self.weights[1] * loss_p
        
        return loss, loss_corr
    
    def loss_function(self):
        """
        Closure function; calculates all losses (IC, BC, PDE)
        """
        start = self.k * time.time()
        self.optimizer.zero_grad()

        # Начальное условие
        predictions_IC = self.model([self.x_IC, self.y_IC, self.t_IC])[:,0]
        loss_IC = self.weights[3] * self.criterion(predictions_IC, self.c_IC)
        if torch.isnan(loss_IC)==True:
            raise ValueError("nan value reached")
        
        # Граничные условия
        prediction_top_bottom = self.model([self.x_tb, self.y_tb, self.t_tb], self.transform)
        prediction_left_right = self.model([self.x_lr, self.y_lr, self.t_lr], self.transform)

        prediction_c_lr = prediction_left_right[:,0]
        prediction_px   = prediction_left_right[:,1]
        prediction_py   = prediction_top_bottom[:,2]

        
        prediction_c_tb = self.model.derivative(prediction_top_bottom[:,0], self.y_tb)
        c_x = self.model.derivative(prediction_c_lr, self.x_lr)
        
        prediction_c_lr[:self.BC_points2] *= misc.psi_th(self.y_lr[:self.BC_points2], self.a, self.b)
        prediction_c_lr[:self.BC_points2] += (1 - misc.psi_th(self.y_lr[:self.BC_points2], self.a, self.b)) * c_x[:self.BC_points2]
        # prediction_c_lr[self.BC_points2:] = c_x[self.BC_points2:]
        
        loss_BC = (self.weights[4] * (self.criterion(prediction_py[:self.BC_points2],   self.p[:self.BC_points2])                     +
                                      self.criterion(prediction_py[self.BC_points2:],   self.p[self.BC_points2:2*self.BC_points2])    +
                                      self.criterion(prediction_px[:self.BC_points2],   self.p[2*self.BC_points2:3*self.BC_points2])  +
                                      self.criterion(prediction_px[self.BC_points2:],   self.p[3*self.BC_points2:]))                  +
                   
                   self.weights[5] * (self.criterion(prediction_c_tb[:self.BC_points2], self.c[:self.BC_points2])                     +
                                      self.criterion(prediction_c_tb[self.BC_points2:], self.c[self.BC_points2:2*self.BC_points2])    +
                                      self.criterion(prediction_c_lr[:self.BC_points2], self.c[2*self.BC_points2:3*self.BC_points2])) )
                                      # self.criterion(prediction_c_lr[self.BC_points2:], self.c[3*self.BC_points2:]))                  )

        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE
        loss_PDE, loss_corr = self.PDELoss()
        if torch.isnan(loss_PDE)==True:
            raise ValueError("nan value reached")

        output_hook = OutputHook()
        self.model.sin.register_forward_hook(output_hook)

        l1_lambda = 0.5
        l2_lambda = 0.01
        l1_penalty = 0.
        l2_penalty = 0.
        for output in output_hook:
            l1_penalty += torch.norm(output, 1)
            l2_penalty += torch.norm(output, 2)
        l1_penalty *= l1_lambda
        l2_penalty *= l2_lambda

        loss = (loss_PDE   +
                loss_BC    +
                loss_IC    +
                loss_corr  + 
                l1_penalty +
                l2_penalty )

        loss.backward()
        output_hook.clear()
        self.losses.append(loss.item())
        end = self.k * time.time()

        if self.CL==False:
            if self.epoch % self.k == 0:
                self.print_tab.add_rows([['|', f'{self.epoch}\t',         '|',
                                        f'{round(loss_PDE.item(),  6)}\t','|',
                                        f'{round(loss_corr.item(), 6)}\t','|',
                                        f'{round(loss_IC.item(),   6)}\t','|',
                                        f'{round(loss_BC.item(),   6)}\t','|',
                                        f'{round(self.losses[-1],  6)}\t','|',
                                        f'{round(end - start, 6)}',       '|']])
                print(self.print_tab.draw())
        else:
            if self.epoch % self.k == 0:
                self.print_tab.add_rows([['|', f'{self.epoch}\t',         '|',
                                        f'{round(loss_PDE.item(),  6)}\t','|',
                                        f'{round(loss_corr.item(), 6)}\t','|',
                                        f'{round(loss_IC.item(),   6)}\t','|',
                                        f'{round(loss_BC.item(),   6)}\t','|',
                                        f'{round(self.losses[-1],  6)}\t','|',
                                        f'{self.const}\t',                '|',
                                        f'{round(end - start, 6)}',       '|']])  
                print(self.print_tab.draw())
        self.epoch += 1

        return loss


    def train(self):
        """
        The main function of Net training
        """
        self.print_tab = Texttable()
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,10,1,15,1,15,1,15,1,15,1,15,1,10,1])
        self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','p corr loss','|','IC loss','|','BC loss','|','Summary loss','|','time','|']])
        print(self.print_tab.draw())
        self.model.train()

        self.optimizer = self.model.set_optimizer('NAdam')

        if self.epoch <= self.Adam_epochs+1:
            for _ in range(self.epoch, self.Adam_epochs+1):
                self.optimizer.step(self.loss_function)
        
        self.optimizer = self.model.set_optimizer('LBFGS')
        self.optimizer.step(self.loss_function)

    
    def CL_train(self, constants:dict):
        """
        Function for Cirriculum Learning for Net
        
        So far only for flow rates
        """
        self.print_tab = Texttable()
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,10,1,20,1,20,1,20,1,20,1,5,1])
        self.model.train()
        self.CL=True

        if constants.get("v_in")!=None:
            try:
                os.mkdir(f'data/CL_v_in,{constants["v_in"]}')
            except FileExistsError:
                None
            self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|','v_x','|']])
            print(self.print_tab.draw())
            ijk=0
            for param in constants["v_in"]:
                self.v_in = param
                self.const = param
                self.makeIBC()

                self.epoch=0
                self.optimizer = self.model.set_optimizer('NAdam')
                for _ in range(self.epoch, self.Adam_epochs+1):
                    self.optimizer.step(self.loss_function)
                
                # self.optimizer = self.model.set_optimizer('LBFGS')
                # self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_v_in,{constants["v_in"]}/{param}', f'data/CL_v_in,{constants["v_in"]}/{param}_data')
                ijk+=1
        elif constants.get("w")!=None:
            try:
                # os.mkdir(f'data/CL_w,{constants["w"]}')
                os.mkdir('data/megaserv')
            except FileExistsError:
                None
            self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|','w','|']])
            print(self.print_tab.draw())
            ijk=0
            for param in constants["w"]:
                self.w = param
                self.const = param
                self.p_cond[0][2] = -12 * self.mu0 * self.v_in / param**2 * (1 - self.c1/self.cmax)**(self.beta)
                if torch.abs(self.p_cond[0][3].max())!=0 or torch.abs(self.p_cond[0][3].min())!=0:
                    self.p_cond[0][3] = -12 * self.mu0 * self.v_in / param**2
                self.makeIBC()
                # print (self.p_cond[0][2].min(),self.p_cond[0][3].min())

                self.epoch=0
                self.optimizer = self.model.set_optimizer('NAdam')
                for _ in range(self.CL_epochs[ijk]):
                    self.optimizer.step(self.loss_function)
                
                self.optimizer = self.model.set_optimizer('LBFGS')
                self.optimizer.step(self.loss_function)

                # self.full_save(f'data/CL_w,{constants["w"]}/{param}', f'data/CL_w,{constants["w"]}/{param}_data')
                self.full_save(f'data/megaserv/{param}', f'data/megaserv/{param}_data')
                ijk+=1
        elif constants.get("c1")!=None:
            try:
                os.mkdir(f'data/CL_c1,{constants["c1"]}')
            except FileExistsError:
                None
            self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|','c1','|']])
            print(self.print_tab.draw())
            ijk=0
            for param in constants["c1"]:
                self.c1 = param
                self.const = param
                self.c_cond[0][2] = self.c1
                self.p_cond[0][2] = -12 * self.mu0 * self.v_in / self.w**2 * (1 - param/self.cmax)**(self.beta)
                self.makeIBC()

                self.epoch=0
                self.optimizer = self.model.set_optimizer('NAdam')
                for _ in range(self.CL_epochs[ijk]):
                    self.optimizer.step(self.loss_function)
                
                self.optimizer = self.model.set_optimizer('LBFGS')
                self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_c1,{constants["c1"]}/{param}', f'data/CL_c1,{constants["c1"]}/{param}_data')
                ijk+=1
        self.CL=False
    

    def eval_(self):
        self.model.eval()
