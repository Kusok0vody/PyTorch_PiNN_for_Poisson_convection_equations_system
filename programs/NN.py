import numpy as np 
import os
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad

from texttable import Texttable
from collections import OrderedDict

import programs.conditions as cnd


class Sin(nn.Module):
    """
    sin activation function for Neural Network
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        sin = torch.sin(input)
        return sin


class Net(nn.Module):
    """
    The main Class for all Neural Networks
    """
    def __init__(
        self,
        input_size:int,
        neurons_arr:list[int],
        output_size:int,
        depth:int,
        act,):

        super(Net, self).__init__()
        
        layers = [('input', torch.nn.Linear(input_size, neurons_arr[0]))]
        layers.append(('input_activation', act()))
        for i in range(depth): 
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(neurons_arr[i], neurons_arr[i+1]))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(neurons_arr[-1], output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)


    def forward(self, inputs:list):
        inputs_united = inputs[0].reshape(-1, 1)
        for i in range(1, len(inputs)):
            inputs_united = torch.cat([inputs_united, inputs[i].reshape(-1, 1)], axis=1)
        outputs = self.layers(inputs_united)
        return outputs


    def set_optimizer(self, optimizer_type:str):
        """
        Sets the optimizer for Neural Network
        
        Types: Adam, LBFGS
        """
        if optimizer_type=='Adam':
            return torch.optim.Adam(self.parameters())
        if optimizer_type=='LBFGS':
            return torch.optim.LBFGS(self.parameters(),
                                     lr=0.001, 
                                     max_iter=50000, 
                                     max_eval=50000, 
                                     history_size=50,
                                     tolerance_grad=1e-12, 
                                     tolerance_change=0.5 * np.finfo(float).eps,
                                     line_search_fn="strong_wolfe")


    @staticmethod
    def derivative(dx, x, order=1):
        """
        Calculates the derivative of a given array
        """
        for _ in range(order):
            dx = grad(outputs=dx, inputs=x, grad_outputs = torch.ones_like(dx), create_graph=True, retain_graph=True, allow_unused=True)[0]

        return dx
    

    def load(self, path:str):
        """
        Loads NN from file
        """
        self.model.load_state_dict(torch.load(path))


    def save(self, path:str):
        """
        Saves NN to file
        """
        torch.save(self.model.state_dict(), path)
    

class Poisson:
    def __init__(self,
                 size:list,
                 cond:list,
                 collocation:int,
                 ranges:list):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net(input_size=2,
                         neurons_arr=[32,16,16,8,8],
                         output_size=1,
                         depth=4,
                         act=Sin).to(self.device)
        # Technical Variables
        self.Adam_epochs = 5000
        self.losses=[]
        self.epoch = 0
        self.BC_len = ranges[0].shape[0]
        self.zeros = np.zeros(self.BC_len)
        self.ones = np.ones(self.BC_len)
        self.print_tab = Texttable()
        self.criterion = torch.nn.MSELoss()
        self.weights = [1,1,1]
        self.optimizer = self.model.set_optimizer('Adam')
        self.collocation = collocation

        # Constants and conditions
        self.size = size
        self.ranges = ranges
        self.cond = cond
        self.p_cond = cond
        self.makeIBC()

        x_collocation = torch.linspace(self.size[0], self.size[1], self.collocation).to(self.device)
        y_collocation = torch.linspace(self.size[2], self.size[3], self.collocation).to(self.device)

        self.XY = torch.stack(torch.meshgrid(x_collocation, y_collocation)).reshape(2, -1).T
        self.XY.requires_grad = True
        self.X = Variable(self.XY[:,0], requires_grad=True).to(self.device)
        self.Y = Variable(self.XY[:,1], requires_grad=True).to(self.device)
    

    def full_load(self, path_nn:str, path_data:str):
        """
        Loads NN and other parameters (temporarily only losses) from file
        """
        self.model.load_state_dict(torch.load(path_nn))
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
        Makes boundary conditions
        """
        condition = cnd.form_boundaries(self.ranges, self.p_cond[0], self.ones, self.zeros)
        self.f, XY = cnd.form_condition_arrays(condition)
        self.x = XY[0]
        self.y = XY[1]


    def check_BC(self, cond, u, u_tb, u_lr, x, y):
        """
        Makes the Neumann condition on those boundaries where it is required
        """
        u_x = self.model.derivative(u_lr, x)
        u_y = self.model.derivative(u_tb, y)

        if cond[0]==1:
            u[:self.BC_len] = u_y[:self.BC_len]
        if cond[1]==1:
            u[self.BC_len:2*self.BC_len] = u_y[self.BC_len:]
        if cond[2]==1:
            u[2*self.BC_len:3*self.BC_len] = u_x[:self.BC_len]
        if cond[3]==1:
            u[3*self.BC_len:] = u_x[self.BC_len:]

        return u


    def PDELoss(self):
        """
        Calculates the loss from PDE
        """
        u = self.model([self.X, self.Y])

        p_x = self.model.derivative(u, self.X)
        p_y = self.model.derivative(u, self.Y)

        p_xx = self.model.derivative(p_x, self.X)
        p_yy = self.model.derivative(p_y, self.Y)

        p = p_xx + p_yy

        loss = self.weights[0] * self.criterion(torch.zeros_like(p), p)

        return loss
    

    def loss_function(self):
        """
        Closure function; calculates all losses (IC, BC, PDE)
        """
        self.optimizer.zero_grad()

        # Boundary conditions
        pt_x_tb = Variable(self.x[:2*self.BC_len], requires_grad=True).to(self.device)
        pt_y_tb = Variable(self.y[:2*self.BC_len], requires_grad=True).to(self.device)

        pt_x_lr = Variable(self.x[2*self.BC_len:], requires_grad=True).to(self.device)
        pt_y_lr = Variable(self.y[2*self.BC_len:], requires_grad=True).to(self.device)

        pt_p = Variable(self.f, requires_grad=True).to(self.device)

        prediction_top_bottom = self.model([pt_x_tb, pt_y_tb])
        prediction_left_right = self.model([pt_x_lr, pt_y_lr])

        prediction_BC = torch.cat((prediction_top_bottom, prediction_left_right))

        prediction_p = self.check_BC(self.cond[1], prediction_BC.reshape(-1), prediction_top_bottom, prediction_left_right, pt_x_lr, pt_y_tb)

        loss_BC = self.weights[1] * self.criterion(pt_p, prediction_p)
        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE
        loss_PDE = self.PDELoss()
        if torch.isnan(loss_PDE)==True:
            raise ValueError("nan value reached")

        loss = loss_PDE + loss_BC
        loss.backward()
        
        self.losses.append(loss.item())

        if self.epoch % 10 == 0:
            self.print_tab.add_rows([['|',f'{self.epoch}\t','|',
                                      f'{loss_PDE}\t','|',
                                      f'{loss_BC}\t','|',
                                      f'{self.losses[-1]}\t','|']])
            print(self.print_tab.draw())
        self.epoch += 1

        return loss


    def train(self):
        """
        The main function of Net training
        """
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,15,1,25,1,25,1,25,1])
        self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','BC loss','|','Summary loss','|']])
        print(self.print_tab.draw())
        self.model.train()

        self.optimizer = self.model.set_optimizer('Adam')

        if self.epoch <= self.Adam_epochs+1:
            for _ in range(self.epoch, self.Adam_epochs+1):
                self.optimizer.step(self.loss_function)
        
        self.optimizer = self.model.set_optimizer('LBFGS')
        self.optimizer.step(self.loss_function)

class Poisson_Convection:
    def __init__(self,
                 w:np.float64,
                 mu0:np.float64,
                 cmax:np.float64,
                 v_in:np.float64,
                 chi:np.float64,
                 size:list,
                 c_cond:list,
                 p_cond:list,
                 collocation:int,
                 ranges:list):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = Net(input_size=3,
                         neurons_arr=[32,16,16,8,8],
                         output_size=2,
                         depth=4,
                         act=Sin).to(self.device)
        
        # Technical Variables
        self.Adam_epochs = 5000
        self.losses=[]
        self.epoch = 0
        self.const = 0
        self.BC_len = ranges[0].shape[0]
        self.zeros = np.zeros(self.BC_len)
        self.ones = np.ones(self.BC_len)
        self.print_tab = Texttable()
        self.criterion = torch.nn.MSELoss()
        self.weights = [1,1,1,1]
        self.CL = False
        self.optimizer = self.model.set_optimizer('Adam')
        self.collocation = collocation

        # Constants and conditions
        self.w = w
        self.mu0 = mu0
        self.cmax = cmax
        self.v_in = v_in
        self.chi = chi
        self.size = size
        self.ranges = ranges
        self.c_cond = c_cond
        self.p_cond = p_cond
        self.makeIBC()

        x_collocation = torch.linspace(self.size[0], self.size[1], self.collocation).to(self.device)
        y_collocation = torch.linspace(self.size[2], self.size[3], self.collocation).to(self.device)
        t_collocation = torch.linspace(self.size[4], self.size[5], self.collocation).to(self.device)

        self.XYT = torch.stack(torch.meshgrid(x_collocation, y_collocation, t_collocation)).reshape(3, -1).T
        self.XYT.requires_grad = True
        self.X = Variable(self.XYT[:,0], requires_grad=True).to(self.device)
        self.Y = Variable(self.XYT[:,1], requires_grad=True).to(self.device)
        self.T = Variable(self.XYT[:,2], requires_grad=True).to(self.device)
    

    def full_load(self, path_nn:str, path_data:str):
        """
        Loads NN and other parameters (temporarily only losses) from file
        """
        self.model.load_state_dict(torch.load(path_nn))
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
        self.IC_x = torch.linspace(self.size[0], self.size[1], int(self.BC_len/1000)).to(self.device)
        self.IC_y = torch.linspace(self.size[2], self.size[3], int(self.BC_len/1000)).to(self.device)
        self.IC_t = torch.zeros(int(self.BC_len/1000)).to(self.device)

        IC_XYT = torch.stack(torch.meshgrid(self.IC_x, self.IC_y, self.IC_t)).reshape(3, -1).T

        self.IC_x = Variable(IC_XYT[:,0], requires_grad=True).to(self.device)
        self.IC_y = Variable(IC_XYT[:,1], requires_grad=True).to(self.device)
        self.IC_t = Variable(IC_XYT[:,2], requires_grad=True).to(self.device)

        c_initial = torch.zeros(self.IC_x.shape).to(self.device)
        # for i in range(len(self.IC_x)):
        #     if self.IC_x[i]==1 and torch.abs(self.IC_y[i] - torch.max(self.IC_y) / 2) <= self.chi / 2:
        #         c_initial[i] = np.max(self.c_cond[0][3]) 
        self.IC_c = c_initial

        # BC for c
        c_condition = cnd.form_boundaries(self.ranges, self.c_cond[0], self.ones, self.zeros)
        self.c_f, XYT = cnd.form_condition_arrays(c_condition)
        self.x = XYT[0]
        self.y = XYT[1]
        self.t = XYT[2]

        # BC for p
        p_condition = cnd.form_boundaries(self.ranges, self.p_cond[0], self.ones, self.zeros)
        self.p_f, _ = cnd.form_condition_arrays(p_condition)


    def check_BC(self, cond, u, u_tb, u_lr, x, y):
        """
        Makes the Neumann condition on those boundaries where it is required
        """
        u_x = self.model.derivative(u_lr, x)
        u_y = self.model.derivative(u_tb, y)

        if cond[0]==1:
            u[:self.BC_len] = u_y[:self.BC_len]
        if cond[1]==1:
            u[self.BC_len:2*self.BC_len] = u_y[self.BC_len:]
        if cond[2]==1:
            u[2*self.BC_len:3*self.BC_len] = u_x[:self.BC_len]
        if cond[3]==1:
            u[3*self.BC_len:] = u_x[self.BC_len:]

        return u


    def PDELoss(self):
        """
        Calculates the loss from PDE
        """
        u = self.model([self.X, self.Y, self.T])
        u[:,1] = torch.clamp(u[:,1].clone(), min=0, max=self.cmax-0.0000001)

        p_x = self.model.derivative(u[:,0], self.X)
        p_y = self.model.derivative(u[:,0], self.Y)

        mu = self.mu0 * (1 - u[:,1] / self.cmax).pow(-2.5)

        v_x = -self.w**2 * p_x / (12 * mu)
        v_y = -self.w**2 * p_y / (12 * mu)

        c_x = self.model.derivative(u[:,1]*v_x, self.X)
        c_y = self.model.derivative(u[:,1]*v_y, self.Y)
        c_t = self.model.derivative(u[:,1],     self.T)

        p_xx = self.model.derivative(p_x / mu, self.X)
        p_yy = self.model.derivative(p_y / mu, self.Y)

        c = c_t + c_x + c_y
        p = p_xx + p_yy

        loss_p = self.criterion(torch.zeros_like(p), p)
        loss_c = self.criterion(torch.zeros_like(c), c)

        loss = self.weights[0] * loss_p + self.weights[1] * loss_c

        return loss
    

    def loss_function(self):
        """
        Closure function; calculates all losses (IC, BC, PDE)
        """
        self.optimizer.zero_grad()

        # Initial conditions
        pt_x_IC = Variable(self.IC_x, requires_grad=True).to(self.device)
        pt_y_IC = Variable(self.IC_y, requires_grad=True).to(self.device)
        pt_t_IC = Variable(self.IC_t, requires_grad=True).to(self.device)
        pt_c_IC = Variable(self.IC_c, requires_grad=True).to(self.device).reshape(-1)

        predictions_IC = self.model([pt_x_IC, pt_y_IC, pt_t_IC])[:,1]

        loss_IC = self.weights[2] * self.criterion(predictions_IC, pt_c_IC)
        if torch.isnan(loss_IC)==True:
            raise ValueError("nan value reached")

        # Boundary conditions
        pt_x_tb = Variable(self.x[:2*self.BC_len], requires_grad=True).to(self.device)
        pt_y_tb = Variable(self.y[:2*self.BC_len], requires_grad=True).to(self.device)
        pt_t_tb = Variable(self.t[:2*self.BC_len], requires_grad=True).to(self.device)

        pt_x_lr = Variable(self.x[2*self.BC_len:], requires_grad=True).to(self.device)
        pt_y_lr = Variable(self.y[2*self.BC_len:], requires_grad=True).to(self.device)
        pt_t_lr = Variable(self.t[2*self.BC_len:], requires_grad=True).to(self.device)

        pt_c = Variable(self.c_f, requires_grad=True).to(self.device)
        pt_p = Variable(self.p_f, requires_grad=True).to(self.device)

        prediction_top_bottom = self.model([pt_x_tb, pt_y_tb, pt_t_tb])
        prediction_left_right = self.model([pt_x_lr, pt_y_lr, pt_t_lr])

        prediction_BC = torch.cat((prediction_top_bottom,prediction_left_right))

        prediction_p = self.check_BC(self.p_cond[1], prediction_BC[:,0], prediction_top_bottom[:,0], prediction_left_right[:,0], pt_x_lr, pt_y_tb)
        prediction_c = self.check_BC(self.c_cond[1], prediction_BC[:,1], prediction_top_bottom[:,1], prediction_left_right[:,1], pt_x_lr, pt_y_tb)

        loss_BC = self.weights[3] * (self.criterion(pt_p, prediction_p) + self.criterion(pt_c, prediction_c))
        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE
        loss_PDE = self.PDELoss()
        if torch.isnan(loss_PDE)==True:
            raise ValueError("nan value reached")

        loss = loss_PDE + loss_IC + loss_BC
        loss.backward()
        
        self.losses.append(loss.item())
        
        if self.CL==False:
            if self.epoch % 10 == 0:
                self.print_tab.add_rows([['|',f'{self.epoch}\t','|',
                                        f'{loss_PDE}\t','|',
                                        f'{loss_IC}\t','|',
                                        f'{loss_BC}\t','|',
                                        f'{self.losses[-1]}\t','|']])
                print(self.print_tab.draw())
        else:
            if self.epoch % 10 == 0:
                self.print_tab.add_rows([['|',f'{self.epoch}\t','|',
                                        f'{loss_PDE}\t','|',
                                        f'{loss_IC}\t','|',
                                        f'{loss_BC}\t','|',
                                        f'{self.losses[-1]}\t','|',
                                        f'{self.const}\t','|']])
                print(self.print_tab.draw())
        self.epoch += 1

        return loss


    def train(self):
        """
        The main function of Net training
        """
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,15,1,25,1,25,1,25,1,25,1])
        self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|']])
        print(self.print_tab.draw())
        self.model.train()

        self.optimizer = self.model.set_optimizer('Adam')

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
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,15,1,25,1,25,1,25,1,25,1,15,1])
        self.model.train()
        self.CL=True

        if constants.get("v_in")!=None:
            os.mkdir(f'data/CL_v_in,{constants["v_in"]}')
            self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|','v_x','|']])
            print(self.print_tab.draw())
            for param in constants["v_in"]:
                self.v_in = param
                self.const = param
                self.makeIBC()

                self.epoch=0
                self.optimizer = self.model.set_optimizer('Adam')
                if self.epoch <= self.Adam_epochs+1:
                    for _ in range(self.epoch, self.Adam_epochs+1):
                        self.optimizer.step(self.loss_function)
                
                self.optimizer = self.model.set_optimizer('LBFGS')
                self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_v_in,{constants["v_in"]}/{param}', f'data/CL_v_in,{constants["v_in"]}/{param}_data')
        self.CL=False
    

    def eval_(self):
        self.model.eval()
