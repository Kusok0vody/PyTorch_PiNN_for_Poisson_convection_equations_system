import numpy as np 
import os
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad

from texttable import Texttable
from collections import OrderedDict
from copy import deepcopy

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

class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, neural, ideal):
        rmse = torch.sqrt(torch.nn.MSELoss(neural, ideal))
        return rmse

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
            return torch.optim.NAdam(self.parameters(), weight_decay=1e-5)
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
            dx = grad(outputs=dx, inputs=x, grad_outputs = torch.ones_like(dx), create_graph=True, retain_graph=True)[0]

        return dx
    

    def load_model(self, path:str):
        """
        Loads NN from file
        """
        self.load_state_dict(torch.load(path))


    def save_model(self, path:str):
        """
        Saves NN to file
        """
        torch.save(self.state_dict(), path)
    

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
        self.cond_points = ranges[0].shape[0]
        self.zeros = torch.zeros(self.cond_points)
        self.ones = torch.ones(self.cond_points)
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
            u[:self.cond_points] = u_y[:self.cond_points]
        if cond[1]==1:
            u[self.cond_points:2*self.cond_points] = u_y[self.cond_points:]
        if cond[2]==1:
            u[2*self.cond_points:3*self.cond_points] = u_x[:self.cond_points]
        if cond[3]==1:
            u[3*self.cond_points:] = u_x[self.cond_points:]

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
        pt_x_tb = Variable(self.x[:2*self.cond_points], requires_grad=True).to(self.device)
        pt_y_tb = Variable(self.y[:2*self.cond_points], requires_grad=True).to(self.device)

        pt_x_lr = Variable(self.x[2*self.cond_points:], requires_grad=True).to(self.device)
        pt_y_lr = Variable(self.y[2*self.cond_points:], requires_grad=True).to(self.device)

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
                 c_in:np.float64,
                 chi:np.float64,
                 size:list,
                 c_cond:list,
                 p_cond:list,
                 collocation:int,
                 cond_points:int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = Net(input_size=3,
                         neurons_arr=[32,16,16,8,8],
                         output_size=2,
                         depth=4,
                         act=Sin).to(self.device)
        
        # Technical Variables
        self.Adam_epochs = 10000
        self.losses=[]
        self.epoch = 0
        self.const = 0
        self.cond_points = cond_points
        self.cond_p = self.cond_points**3
        self.zeros = torch.Tensor([0]).to(self.device)
        self.ones = torch.Tensor([1]).to(self.device)
        self.print_tab = Texttable()
        self.criterion = torch.nn.MSELoss()
        self.weights = [1,1,1,1]
        self.CL_epochs = []
        self.CL = False
        self.optimizer = self.model.set_optimizer('Adam')
        self.collocation = collocation

        self.ppde = []
        self.bc = []
        self.ic = []

        # Constants and conditions
        self.w = w
        self.mu0 = mu0
        self.cmax = cmax
        self.v_in = v_in
        self.c_in = c_in
        self.chi = chi
        self.size = deepcopy(size)
        self.c_cond = deepcopy(c_cond)
        self.p_cond = deepcopy(p_cond)

        self.makeIBC()

        # Coords for PDE
        x_collocation = torch.linspace(self.size[0], self.size[1], self.collocation).to(self.device)
        y_collocation = torch.linspace(self.size[2], self.size[3], self.collocation).to(self.device)
        t_collocation = torch.linspace(self.size[4]+0.01, self.size[5], self.collocation).to(self.device)

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

        x = torch.linspace(self.size[0]+0.01, self.size[1]-0.01, self.cond_points).to(self.device)
        y = torch.linspace(self.size[2]+0.01, self.size[3]-0.01, self.cond_points).to(self.device)
        t = torch.linspace(self.size[4], self.size[5], self.cond_points).to(self.device)
        XYT = torch.stack(torch.meshgrid(x, y, t)).reshape(3, -1).T

        self.IC_x = XYT[:,0]
        self.IC_y = XYT[:,1]
        self.t = XYT[:,2]
        self.IC_t = torch.zeros_like(XYT[:,2]).to(self.device)
        self.IC_c = torch.zeros_like(self.IC_t).to(self.device)
        
        x = torch.linspace(self.size[0], self.size[1], self.cond_points**2).to(self.device)
        y = torch.linspace(self.size[2], self.size[3], self.cond_points**2).to(self.device)
        t = torch.linspace(self.size[4], self.size[5], self.cond_points).to(self.device)
        psi = torch.where(torch.abs(y - torch.max(y) / 2) <= self.chi / 2, 1., 0.).to(self.device)
        psi = torch.stack(torch.meshgrid(psi, torch.Tensor([0 for _ in range(32)]).to(self.device))).reshape(2, -1).T[:,0]

        # BC for c
        for i in range(len(self.c_cond[0])):
            if self.c_cond[2][i]:
                self.c_cond[0][i] = self.c_cond[0][i] * psi
            else:
                self.c_cond[0][i] = self.c_cond[0][i] * torch.ones_like(self.IC_x)

        c_condition = cnd.form_boundaries([x, y, t], self.c_cond[0], self.ones, self.zeros)
        self.c_f, XYT = cnd.form_condition_arrays(c_condition)

        # BC for p
        for i in range(len(self.p_cond[0])):
            if self.p_cond[2][i]:
                self.p_cond[0][i] = self.p_cond[0][i] * psi
            else:
                self.p_cond[0][i] = self.p_cond[0][i] * torch.ones_like(self.IC_x)

        p_condition = cnd.form_boundaries([x, y, t], self.p_cond[0], self.ones, self.zeros)
        self.p_f, _ = cnd.form_condition_arrays(p_condition)

        self.x = XYT[0]
        self.y = XYT[1]
        self.t = XYT[2]


    def make_XYT(self):

        # Coords for PDE
        x_collocation = torch.linspace(self.size[0], self.size[1], self.collocation).to(self.device)
        y_collocation = torch.linspace(self.size[2], self.size[3], self.collocation).to(self.device)
        t_collocation = torch.linspace(self.size[4], self.size[5], self.collocation).to(self.device)

        self.XYT = torch.stack(torch.meshgrid(x_collocation, y_collocation, t_collocation)).reshape(3, -1).T
        # self.XYT.requires_grad = True
        self.X = Variable(self.XYT[:,0], requires_grad=True).to(self.device)
        self.Y = Variable(self.XYT[:,1], requires_grad=True).to(self.device)
        self.T = Variable(self.XYT[:,2], requires_grad=True).to(self.device)

        #IC coords (without boundaries) + cond:
        x = torch.linspace(self.size[0]+0.01, self.size[1]-0.01, self.cond_points).to(self.device)
        y = torch.linspace(self.size[2]+0.01, self.size[3]-0.01, self.cond_points).to(self.device)
        XY = torch.stack(torch.meshgrid(x, y)).reshape(2, -1).T

        self.IC_x = XY[:,0]
        self.IC_y = XY[:,1]
        self.IC_t = torch.zeros_like(XY[:,0]).to(self.device)
        self.IC_c = torch.zeros_like(self.IC_t).to(self.device)
        # self.IC_c += torch.where(self.IC_x >= 0.4, 0.5, 0.).to(self.device)
        # self.IC_c += torch.where(self.IC_x >= 0.6, -0.5, 0.).to(self.device)

        #BC coords
        self.raw_x = torch.linspace(self.size[0], self.size[1], self.cond_points).to(self.device)
        # self.raw_xx = torch.linspace(self.size[0], self.size[1], self.cond_points).to(self.device)
        self.raw_y = torch.linspace(self.size[2], self.size[3], self.cond_points).to(self.device)
        self.raw_t = torch.linspace(self.size[4], self.size[5], self.cond_points).to(self.device)
        self.x, self.y, self.t = cnd.formBC_coords(self.raw_x, self.raw_y, self.raw_t)

        self.psi = torch.where(torch.abs(self.y - torch.max(self.y) / 2) <= self.chi / 2, 1/self.chi, 0)[2*self.cond_p:3*self.cond_p]
        #self.psi = torch.stack(torch.meshgrid(self.psi, torch.zeros(32).to(self.device))).reshape(2, -1).T[:,0]
        self.c_psi = self.psi * self.chi


    def make_BC_conditions(self, p, c, x, y):

        # BC for c
        c_cond = [0, 0, 0, 0]
        c_cond[2] = self.c_in * self.c_psi

        p_y = self.model.derivative(p, y)
        p_yy = self.model.derivative(p_y, y)
        p_x = self.model.derivative(p, x)
        p_xx = self.model.derivative(p_x, x)
        
        const_top    = 0*self.cmax / (2.5 * c[:self.cond_p]              * (1 - c[:self.cond_p] / self.cmax).pow(-3.5)              - self.cmax)
        const_bottom = 0*self.cmax / (2.5 * c[self.cond_p:2*self.cond_p] * (1 - c[self.cond_p:2*self.cond_p] / self.cmax).pow(-3.5) - self.cmax)
        const_right  = 0*self.cmax / (2.5 * c[self.cond_p*3:]            * (1 - c[self.cond_p*3:] / self.cmax).pow(-3.5)            - self.cmax)

        c_cond[0] = c[:self.cond_p]              * p_yy[:self.cond_p] / p_y[:self.cond_p] * const_top
        c_cond[1] = c[self.cond_p:2*self.cond_p] * p_yy[self.cond_p:] / p_y[self.cond_p:] * const_bottom
        c_cond[3] = c[3*self.cond_p:]            * p_xx[self.cond_p:] / p_x[self.cond_p:] * const_right

        self.c_condition = torch.cat(c_cond)

        # BC for p
        p_cond = [0, 0, 0, 0]
        cond =  -12 * self.mu0 * self.v_in / self.w**2 #*(1 - self.c_in / self.cmax)**(-2.5)

        p_cond[0] = torch.zeros(self.cond_p).to(self.device)
        p_cond[1] = torch.zeros(self.cond_p).to(self.device)
        p_cond[2] = cond * self.psi
        p_cond[3] = torch.zeros(self.cond_p).to(self.device)

        self.p_condition = torch.cat(p_cond)

    def check_BC(self, cond, u, u_tb, u_lr, x, y):
        """
        Makes the Neumann condition on those boundaries where it is required
        """
        u_x = self.model.derivative(u_lr, x)
        u_y = self.model.derivative(u_tb, y)
        

        if cond[0]==1:
            u[:self.cond_p] = u_y[:self.cond_p]
        if cond[1]==1:
            u[self.cond_p:2*self.cond_p] = u_y[self.cond_p:]
        if cond[2]==1:
            u[2*self.cond_p:3*self.cond_p] = u_x[:self.cond_p]
        if cond[3]==1:
            u[3*self.cond_p:] = u_x[self.cond_p:]

        return u


    def PDELoss(self):
        """
        Calculates the loss from PDE
        """
        u = self.model([self.X, self.Y, self.T])
        # if torch.max(u[:,1])>=self.cmax or torch.min(u[:,1])<0:
            # u = self.model([self.X, self.Y, self.T])
        u[:,0] = torch.abs(torch.clamp(u[:,0].clone(), max=self.cmax-0.001))

        p_x = self.model.derivative(u[:,1], self.X)
        p_y = self.model.derivative(u[:,1], self.Y)

        mu12 = 12. * self.mu0 * (1. - u[:,0] / self.cmax).pow(-2.5)

        v_x = -self.w**2 * p_x / mu12
        v_y = -self.w**2 * p_y / mu12

        c_x = self.model.derivative(u[:,0]*v_x*self.w, self.X)
        c_y = self.model.derivative(u[:,0]*v_y*self.w, self.Y)
        c_t = self.model.derivative(u[:,0]*1. *self.w, self.T)

        p_xx = self.model.derivative(p_x * self.w**3 / mu12, self.X)
        p_yy = self.model.derivative(p_y * self.w**3 / mu12, self.Y)

        p = p_xx + p_yy
        c = c_t + c_x + c_y
        
        loss_p = self.criterion(p, torch.zeros_like(p))
        loss_c = self.criterion(c, torch.zeros_like(c))

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

        predictions_IC = self.model([pt_x_IC, pt_y_IC, pt_t_IC])[:,0]

        loss_IC = self.weights[2] * self.criterion(predictions_IC, pt_c_IC)
        self.ic.append(loss_IC.item())
        if torch.isnan(loss_IC)==True:
            raise ValueError("nan value reached")

        # Boundary conditions
        pt_x_tb = Variable(self.x[:2*self.cond_p], requires_grad=True).to(self.device)
        pt_y_tb = Variable(self.y[:2*self.cond_p], requires_grad=True).to(self.device)
        pt_t_tb = Variable(self.t[:2*self.cond_p], requires_grad=True).to(self.device)

        pt_x_lr = Variable(self.x[2*self.cond_p:], requires_grad=True).to(self.device)
        pt_y_lr = Variable(self.y[2*self.cond_p:], requires_grad=True).to(self.device)
        pt_t_lr = Variable(self.t[2*self.cond_p:], requires_grad=True).to(self.device)

        prediction_top_bottom = self.model([pt_x_tb, pt_y_tb, pt_t_tb])
        # prediction_top_bottom[:,1] = torch.clamp(torch.abs(prediction_top_bottom[:,1].clone()), min=0., max=self.cmax-0.001)
        # prediction_top_bottom[:,0] = prediction_top_bottom[:,0] * -self.w * (1 - prediction_top_bottom[:,1] / self.cmax).pow(2.5) / 12 / self.mu0
        prediction_left_right = self.model([pt_x_lr, pt_y_lr, pt_t_lr])
        # prediction_left_right[:,1] = torch.clamp(torch.abs(prediction_left_right[:,1].clone()), min=0., max=self.cmax-0.001)
        # prediction_left_right[:,0][self.cond_p:] = prediction_left_right[:,0][self.cond_p:] * -self.w * (1 - prediction_left_right[:,1][self.cond_p:] / self.cmax).pow(2.5) / 12 / self.mu0

        prediction_BC = torch.cat((prediction_top_bottom, prediction_left_right))

        # self.make_BC_conditions(prediction_BC[:,0], prediction_BC[:,1], pt_x_lr, pt_y_tb)

        pt_c = Variable(self.c_f, requires_grad=True).to(self.device)
        pt_p = Variable(self.p_f, requires_grad=True).to(self.device)

        prediction_p = self.check_BC(self.p_cond[1], prediction_BC[:,1], prediction_top_bottom[:,1], prediction_left_right[:,1], pt_x_lr, pt_y_tb)
        prediction_c = self.check_BC(self.c_cond[1], prediction_BC[:,0], prediction_top_bottom[:,0], prediction_left_right[:,0], pt_x_lr, pt_y_tb)

        loss_BC = self.weights[3] * torch.sqrt(self.criterion(prediction_p, pt_p)) + self.weights[4] * torch.sqrt(self.criterion(prediction_c, pt_c))
        self.bc.append(loss_BC.item())
        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE
        loss_PDE = self.PDELoss()
        self.ppde.append(loss_PDE.item())
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
                self.optimizer = self.model.set_optimizer('Adam')
                for _ in range(self.epoch, self.Adam_epochs+1):
                    self.optimizer.step(self.loss_function)
                
                self.optimizer = self.model.set_optimizer('LBFGS')
                self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_v_in,{constants["v_in"]}/{param}', f'data/CL_v_in,{constants["v_in"]}/{param}_data')
                ijk+=1
        elif constants.get("w")!=None:
            try:
                os.mkdir(f'data/CL_w,{constants["w"]}')
            except FileExistsError:
                None
            self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|','w','|']])
            print(self.print_tab.draw())
            ijk=0
            for param in constants["w"]:
                self.w = param
                self.const = param
                self.p_cond[0][2] = -12 * self.mu0 * self.v_in / param**2 * (1 - self.c_in/self.cmax)**(-2.5)
                if torch.abs(self.p_cond[0][3].max())!=0 or torch.abs(self.p_cond[0][3].min())!=0:
                    self.p_cond[0][3] = -12 * self.mu0 * self.v_in / param**2 * (1 - self.c_in/self.cmax)**(-2.5)
                self.makeIBC()

                self.epoch=0
                self.optimizer = self.model.set_optimizer('Adam')
                for _ in range(self.CL_epochs[ijk]):
                    self.optimizer.step(self.loss_function)
                
                self.optimizer = self.model.set_optimizer('LBFGS')
                self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_w,{constants["w"]}/{param}', f'data/CL_w,{constants["w"]}/{param}_data')
                ijk+=1
        self.CL=False
    

    def eval_(self):
        self.model.eval()
