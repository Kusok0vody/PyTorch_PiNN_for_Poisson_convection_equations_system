import os
import torch
import time
import json
import io

from torch.autograd import Variable
from texttable import Texttable
from copy import copy

import programs.conditions as cnd
import programs.misc as misc
import programs.objects as obj
from programs.NN import *


class Poisson_Convection:
    def __init__(self, data : dict
                #  'w'                  : float,
                #  'mu0'                : float,
                #  'cmax'               : float,
                #  'u_in'               : float,
                #  'chi'                : float,
                #  'transitional_times' : list,
                #  'size'               : list,
                #  'c_cond'             : list,
                #  'p_cond'             : list,
                #  'PDE_points'         : int,
                #  'BC_points'          : int,
                #  'IC_points'          : int,
                #  'beta'               : float = -2.5,
                #  'NN_params'          : dict = None}
                ):

        # CPU/GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        # Дебаг для производной
        torch.autograd.set_detect_anomaly(False)
        
        min_data = {'w'          : float,
                    'mu0'        : float,
                    'cmax'       : float,
                    'u_in'       : float,
                    'chi'        : float,
                    'times'      : list,
                    'size'       : list,
                    'c_cond'     : list,
                    'PDE_points' : int,
                    'IC_points'  : int,
                    'BC_points'  : int,
        }
        
        for i in min_data.keys():
            if not(i in data):
                raise ValueError(f'Value \"{i}\" not found')
            if type(min_data[i])==int and not(type(data[i])==int):
                raise ValueError(f'Value {data[i]} in {i} must be integer')

        self.update_data(data)

        # Создание модели
        self.model = Net(input_size  = self.NN_params.get('input_size'),
                         neurons_arr = self.NN_params.get('neurons_arr'),
                         output_size = self.NN_params.get('output_size'),
                         depth       = self.NN_params.get('depth'),
                         act         = obj.Sin)

        self.optimizer = self.model.set_optimizer('NAdam')
        
        # Создание сеток координат, массивов НУ и ГУ
        self.update()
    
    def update_data(self, data:dict):
        """
        Updates variables based on the supplied dictionary.
        """
        # Итерационные переменные
        self.Adam_epochs = data.setdefault('Adam_epochs', 2000)
        self.epoch       = data.setdefault('epoch', 0)
        self.k           = data.setdefault('k', 10)

        # Массивы ошибок
        self.losses = data.setdefault('losses', [])
        self.PDE    = data.setdefault('PDE', [])
        self.BC     = data.setdefault('BC', [])
        self.IC     = data.setdefault('IC', [])

        # Вспомогаительные тензоры 0 и 1
        self.zeros = torch.FloatTensor([0]).to(self.device)
        self.ones  = torch.FloatTensor([1]).to(self.device)

        # Параметры для обучения
        self.criterion = torch.nn.MSELoss()
        self.weights   = data.setdefault('weights', [1,1,1,1,1,1])

        # Параметры для Cirriculum Learning
        self.CL_epochs = []
        self.CL        = False
        self.const     = 0
        
        # Физические Константты
        self.mu0  = data.get('mu0')
        self.beta = data.setdefault('beta', -2.5)
        self.cmax = data.get('cmax')
        self.u_in = data.get('u_in')

        # Геометрия
        self.chi    = data.get('chi')
        self.size   = data.get('size')
        self.a      = data.setdefault('a', (self.size[3] - self.chi)/2)
        self.b      = data.setdefault('b', (self.size[3] + self.chi)/2)
        self.random = data.setdefault('random', [False, False, True])

        # Параметры НУ
        self.IC_points = data.get('IC_points')
        self.IC_type   = data.setdefault('IC_type', 'zero')
        self.IC_const  = data.setdefault('IC_const', 0)
        self.band_val  = data.setdefault('band_val', 0.6)
        self.bandshape = data.setdefault('IC_bandshape', [0.2,0.4])

        # Параметры ГУ
        self.BC_points  = data.get('BC_points')
        self.BC_points2 = self.BC_points**2
        self.c_cond     = data.get('c_cond')
        self.times      = data.get('times') 
        self.check_vel  = data.setdefault('check_vel', False)

        # Параметры ДУЧП
        self.PDE_points = data.get('PDE_points')
        self.PDE_x_mult = data.setdefault('PDE_x_mult', 2)
        self.PDE_x_int  = data.setdefault('PDE_x_int', 1/4)
        
        # Координаты точек "разрыва"
        self.y_change = torch.FloatTensor([self.a, self.b]).to(self.device)
        self.t_change = torch.FloatTensor(self.times[1:-1]).to(self.device)

        # Параметры ширины трещины
        self.w  = data.get('w')
        self.w1 = data.setdefault('w1', 0.2)
        self.w2 = data.setdefault('w2', 2)
        self.w3 = data.setdefault('w3', 2)
        self.w_func = obj.Width(data.setdefault('w_func', 'const'), self.w, self.w1, self.w2, self.w3)

        # Параметры нейросети
        self.NN_params = {}
        if data.get('NN_params')!=None:
            self.NN_params = data.get('NN_params')
        else:
            self.NN_params['input_size']  = 3
            self.NN_params['neurons_arr'] = [48,48,48,48,48]
            self.NN_params['output_size'] = 3
            self.NN_params['depth']       = 4

    
    def load(self, path):
        with open(path+'.json') as data_file:
            data = json.load(data_file)
        self.update_data(data)
        self.update()
        self.model.load_state_dict(torch.load(path, weights_only=True))

    
    def save(self, path):
        data = {'Adam_epochs' : self.Adam_epochs,
                'epoch'       : self.epoch,
                'k'           : self.k,
                'weights'     : self.weights,
                'mu0'         : self.mu0,
                'beta'        : self.beta,
                'cmax'        : self.cmax,
                'u_in'        : self.u_in,
                'chi'         : self.chi,
                'size'        : self.size,
                'times'       : self.times,
                'a'           : self.a,
                'b'           : self.b,
                'random'      : self.random,
                'IC_points'   : self.IC_points,
                'IC_type'     : self.IC_type,
                'IC_const'    : self.IC_const,
                'band_val'    : self.band_val,
                'bandshape'   : self.bandshape,
                'BC_points'   : self.BC_points,
                'c_cond'      : self.c_cond,
                'check_vel'   : self.check_vel,
                'PDE_points'  : self.PDE_points,
                'PDE_x_mult'  : self.PDE_x_mult,
                'PDE_x_int'   : self.PDE_x_int,
                'w'           : self.w,
                'w1'          : self.w1,
                'w2'          : self.w2,
                'w3'          : self.w3,
                'w_func'      : self.w_func.name,
                'NN_params'   : self.NN_params,
                'losses'      : self.losses,
                'PDE'         : self.PDE,
                'BC'          : self.BC,
                'IC'          : self.IC
               }
            
        with io.open(path+'.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(str(str_))
        torch.save(self.model.state_dict(), path)

    def IC_coords(self):
        if self.random[0]:
            x = torch.cat((self.zeros, self.ones,
                           torch.FloatTensor(self.IC_points-2).to(self.device).uniform_(self.size[0], self.size[1]))).sort()[0]
            y = torch.cat((self.zeros,  self.ones, self.y_change,
                           torch.FloatTensor(self.IC_points-2-2).to(self.device).uniform_(self.size[2], self.size[3]))).sort()[0]
        else:
            x = torch.linspace(self.size[0], self.size[1], self.IC_points).to(self.device)
            y = torch.cat((torch.linspace(self.size[2], self.size[3], self.IC_points-2).to(self.device), self.y_change)).sort()[0]
            
        XYT = torch.stack(torch.meshgrid(x, y, self.zeros, indexing='ij')).reshape(3, -1)
        self.x_IC = Variable(XYT[0], requires_grad=True)
        self.y_IC = Variable(XYT[1], requires_grad=True)
        self.t_IC = Variable(XYT[2], requires_grad=True)

        if self.IC_type=='zero':
            self.c_IC = self.IC_const * torch.ones_like(self.x_IC)
        elif self.IC_type=='band':
            left_side  = torch.where(self.x_IC>=self.bandshape[0], 1, 0)
            right_side = torch.where(self.x_IC<=self.bandshape[1], 1, 0)
            self.c_IC  = torch.where(left_side+right_side==2, self.band_val, self.IC_const)
        else:
            raise NameError(f'The IC type \"{self.IC_type}\" is not defined')
        left_cond = self.x_IC==0
        psi_cond  = misc.psi(self.y_IC, self.chi)
        self.c_IC = torch.where(left_cond+psi_cond==2*True, self.c_cond[0], self.c_IC) 

    def BC_coords(self):
        # Создание массивов для граничных условий
        if self.random[1]:
            x = torch.cat((self.zeros, self.ones,
                           torch.FloatTensor(self.BC_points-2).to(self.device).uniform_(self.size[0], self.size[1]))).sort()[0]
            y = torch.cat((self.zeros, self.ones, self.y_change,
                           torch.FloatTensor(self.BC_points-2-2).to(self.device).uniform_(self.size[2], self.size[3]))).sort()[0]
            t = torch.cat((self.zeros, self.ones, self.t_change,
                           torch.FloatTensor(self.BC_points-2-len(self.times[1:-1])).to(self.device).uniform_(self.size[4], self.size[5]))).sort()[0]
        else:
            x = torch.linspace(self.size[0], self.size[1], self.BC_points).to(self.device)
            y = torch.cat((torch.linspace(self.size[2], self.size[3], self.BC_points-2).to(self.device), self.y_change)).sort()[0]
            t = torch.cat((torch.linspace(self.size[4], self.size[5], self.BC_points-len(self.times[1:-1])).to(self.device), self.t_change)).sort()[0]
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

        left_side_cond = torch.where(self.x==self.size[0], 1, 0)
        psi_cond = misc.psi(self.y, self.chi)
        times = [0] + self.times + [1]
        for i in range(len(times)-1):
            time_start_cond = torch.where(self.t>=times[i], 1., 0.)
            time_end_cond = torch.where(self.t<=times[i+1], 1., 0.)
            self.c = torch.where(time_start_cond + 
                                 time_end_cond   +
                                 left_side_cond  +
                                 psi_cond     == 4,
                                 self.c_cond[i], self.c)
            
            self.p = torch.where(time_start_cond + 
                                 time_end_cond   +
                                 left_side_cond  +
                                 psi_cond     == 4,
                                 misc.viscosity(self.mu0, self.c, self.cmax, self.beta), self.p) 
        right_side_cond = torch.where(self.x==self.size[1], 1, 0)
        self.p  = torch.where(right_side_cond+psi_cond==2, self.mu0, self.p)

        self.p *= -12 * self.u_in / self.w_func(self.x, self.y)**2

        self.x_tb = Variable(self.x[:2*self.BC_points2], requires_grad=True)
        self.y_tb = Variable(self.y[:2*self.BC_points2], requires_grad=True)
        self.t_tb = Variable(self.t[:2*self.BC_points2], requires_grad=True)

        self.x_lr = Variable(self.x[2*self.BC_points2:], requires_grad=True)
        self.y_lr = Variable(self.y[2*self.BC_points2:], requires_grad=True)
        self.t_lr = Variable(self.t[2*self.BC_points2:], requires_grad=True)


    def PDE_coords(self):
        if self.random[2]:
            x_collocation = torch.cat((self.zeros,
                                       torch.FloatTensor(int(self.PDE_x_mult*self.PDE_points*(1-self.PDE_x_int))-1).to(self.device).uniform_(self.size[0],
                                                                                                                                             self.size[1]/2),
                                       torch.FloatTensor(int(self.PDE_x_mult*self.PDE_points*self.PDE_x_int)-1).to(self.device).uniform_(self.size[1]/2,
                                                                                                                                                self.size[1]), 
                                       self.ones)).sort()[0]
            y_collocation = torch.cat((self.zeros, self.ones, self.y_change,
                                       torch.FloatTensor(self.PDE_points-4).to(self.device).uniform_(self.size[2], self.size[3]))).sort()[0]
            t_collocation = torch.cat((self.zeros, self.ones, self.t_change,
                                       torch.FloatTensor(self.PDE_points-2-len(self.times[1:-1])).to(self.device).uniform_(self.size[2], self.size[3]))).sort()[0]
        else:
            x_collocation = torch.linspace(self.size[0], self.size[1], self.PDE_points).to(self.device)
            y_collocation = torch.cat((torch.linspace(self.size[2], self.size[3], self.PDE_points-2).to(self.device), self.y_change)).sort()[0]
            t_collocation= torch.cat((torch.linspace(self.size[4], self.size[5], self.PDE_points-len(self.times[1:-1])).to(self.device), self.t_change)).sort()[0]    

        XYT = torch.stack(torch.meshgrid(x_collocation, y_collocation, t_collocation, indexing='ij')).reshape(3, -1)
        self.X = Variable(XYT[0], requires_grad=True)
        self.Y = Variable(XYT[1], requires_grad=True)
        self.T = Variable(XYT[2], requires_grad=True)


    def update(self):
        self.IC_coords()
        self.BC_coords()
        self.PDE_coords()

    
    def transform(self, net, coords):
        x, y, t = coords

        # Ограничение концентрации от 0 до cmax
        a = self.cmax * torch.sigmoid(500*(net[:,0].clone() - self.cmax))
        net[:,0] *= torch.sigmoid(500*net[:,0]) + torch.sigmoid(500*(self.cmax - net[:,0])) - 1
        net[:,0] += a
        net[:,0] *= 0.9988
        net[:,0] += 1/1700
        # Hard IC
        # net[:,0] *= 2 * torch.sigmoid(250*t) - 1
        # net[:,0] += self.c_cond[0] * 4 * (1 - torch.sigmoid(250*t)) * (1 - torch.sigmoid(250*x)) * misc.psi_th(y, self.a, self.b)

        
    def PDELoss(self):
        """
        Calculates the loss from PDE
        """
        u = self.model([self.X,self.Y,self.T], self.transform)
        
        c  = u[:,0]
        px = u[:,1]
        py = u[:,2]
        
        mu =  (1.0 - c / self.cmax).pow(self.beta)

        u_x = -self.w_func(self.X, self.Y)**2 * px / (12. * self.mu0 * mu)
        u_y = -self.w_func(self.X, self.Y)**2 * py / (12. * self.mu0 * mu)

        c_x = u_x * misc.derivative(c, self.X)
        c_y = u_y * misc.derivative(c, self.Y)
        c_t =       misc.derivative(c, self.T)

        u_xx = misc.derivative(u_x, self.X)
        u_yy = misc.derivative(u_y, self.Y)

        p_xy = misc.derivative(px, self.Y)
        p_yx = misc.derivative(py, self.X)

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

        prediction_c_tb = misc.derivative(prediction_top_bottom[:,0], self.y_tb)
        c_x = misc.derivative(prediction_c_lr, self.x_lr)

        #left
        prediction_c_lr[:self.BC_points2] *= misc.psi_th(self.y_lr[:self.BC_points2], self.a, self.b)
        prediction_c_lr[:self.BC_points2] += (1 - misc.psi_th(self.y_lr[:self.BC_points2], self.a, self.b)) * c_x[:self.BC_points2]
        # right
        # prediction_c_lr[self.BC_points2:int(self.BC_points2 * (1+self.a))] = c_x[self.BC_points2:int(self.BC_points2 * (1+self.a))]
        # prediction_c_lr[int(self.BC_points2 * (1+self.b)):] = c_x[int(self.BC_points2 * (1+self.b)):]
        prediction_c_lr[self.BC_points2:] = c_x[self.BC_points2:]
        
        loss_BC = (self.weights[4] * (self.criterion(prediction_py[:self.BC_points2],   self.p[:self.BC_points2])                     +
                                      self.criterion(prediction_py[self.BC_points2:],   self.p[self.BC_points2:2*self.BC_points2])    +
                                      self.criterion(prediction_px[:self.BC_points2],   self.p[2*self.BC_points2:3*self.BC_points2])  +
                                      self.criterion(prediction_px[self.BC_points2:],   self.p[3*self.BC_points2:]))                  +
                   
                   self.weights[5] * (self.criterion(prediction_c_tb[:self.BC_points2], self.c[:self.BC_points2])                     +
                                      self.criterion(prediction_c_tb[self.BC_points2:], self.c[self.BC_points2:2*self.BC_points2])    +
                                      self.criterion(prediction_c_lr[:self.BC_points2], self.c[2*self.BC_points2:3*self.BC_points2])  +
                                      self.criterion(prediction_c_lr[self.BC_points2:], self.c[3*self.BC_points2:]))                  )
        
        # loss_BC += self.weights[5] * (self.criterion(prediction_c_lr[self.BC_points2:int(self.BC_points2 * (1+self.a))],
                                                     # self.c[3*self.BC_points2:int(self.BC_points2 * (3+self.a))]))
        # loss_BC += self.weights[5] * (self.criterion(prediction_c_lr[int(self.BC_points2 * (1+self.b)):],
                                                     # self.c[int(self.BC_points2 * (3+self.b)):]))
        
        if self.check_vel:
            u = (prediction_px * self.w_func(self.x_lr, self.y_lr)**2 / -12. /
                 misc.viscosity(self.mu0, prediction_c_lr.clone(), self.cmax, self.beta))
            loss_BC += self.criterion(u, self.u_in * misc.psi(self.y_lr, self.chi))
            
        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE
        loss_PDE, loss_corr = self.PDELoss()
        if torch.isnan(loss_PDE)==True:
            raise ValueError("nan value reached")

        output_hook = obj.OutputHook()
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

        if constants.get("u_in")!=None:
            try:
                os.mkdir(f'data/CL_u_in,{constants["u_in"]}')
            except FileExistsError:
                None
            self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|','v_x','|']])
            print(self.print_tab.draw())
            ijk=0
            for param in constants["u_in"]:
                self.u_in = param
                self.const = param
                self.makeIBC()

                self.epoch=0
                self.optimizer = self.model.set_optimizer('NAdam')
                for _ in range(self.epoch, self.Adam_epochs+1):
                    self.optimizer.step(self.loss_function)
                
                # self.optimizer = self.model.set_optimizer('LBFGS')
                # self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_u_in,{constants["u_in"]}/{param}', f'data/CL_u_in,{constants["u_in"]}/{param}_data')
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
                self.p_cond[0][2] = -12 * self.mu0 * self.u_in / param**2 * (1 - self.c1/self.cmax)**(self.beta)
                if torch.abs(self.p_cond[0][3].max())!=0 or torch.abs(self.p_cond[0][3].min())!=0:
                    self.p_cond[0][3] = -12 * self.mu0 * self.u_in / param**2
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
                self.p_cond[0][2] = -12 * self.mu0 * self.u_in / self.w**2 * (1 - param/self.cmax)**(self.beta)
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
