import numpy as np 
import torch

from torch.autograd import Variable

from texttable import Texttable
from copy import deepcopy

import programs.conditions as cnd
from programs.NN import *


class Poisson:
    def __init__(self,
                 size:list,
                 cond:list,
                 collocation:int,
                 cond_points:int,
                 chi:np.float64=1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net(input_size=2,
                         neurons_arr=[32,16,16,8,8],
                         output_size=1,
                         depth=4,
                         act=torch.nn.Tanh).to(self.device)
        
        # Technical Variables
        self.Adam_epochs = 5000
        self.losses=[]
        self.epoch = 0
        self.cond_points = cond_points
        self.cond_p = self.cond_points**2
        self.zeros = torch.Tensor([0]).to(self.device)
        self.ones = torch.Tensor([1]).to(self.device)
        self.print_tab = Texttable()
        self.criterion = torch.nn.MSELoss()
        self.weights = [1,1,1]
        self.optimizer = self.model.set_optimizer('NAdam')
        self.collocation = collocation

        self.PDE = []
        self.BC = []
        self.IC = []

        # Constants and conditions
        self.size = deepcopy(size)
        self.cond = deepcopy(cond)
        self.chi = chi

        self.makeIBC()


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
        x = torch.linspace(self.size[0], self.size[1], self.cond_points**2).to(self.device)
        y = torch.linspace(self.size[2], self.size[3], self.cond_points**2).to(self.device)

        psi = torch.where(torch.abs(y - torch.max(y) / 2) <= self.chi / 2, 1., 0.).to(self.device)

        for i in range(len(self.cond[0])):
            if self.cond[2][i]:
                self.cond[0][i] = self.cond[0][i] * psi
            else:
                self.cond[0][i] = self.cond[0][i] * torch.ones_like(psi)

        condition = cnd.form_boundaries([x, y], self.cond[0], self.ones, self.zeros)
        self.p_f, XY = cnd.form_condition_arrays(condition)

        self.x = XY[0]
        self.y = XY[1]

        # Coords for PDE
        x_collocation = torch.linspace(self.size[0]+0.01, self.size[1]-0.01, self.collocation).to(self.device)
        y_collocation = torch.linspace(self.size[2]+0.01, self.size[3]-0.01, self.collocation).to(self.device)

        self.XY = torch.stack(torch.meshgrid(x_collocation, y_collocation)).reshape(2, -1).T
        self.XY.requires_grad = True
        self.X = Variable(self.XY[:,0], requires_grad=True).to(self.device)
        self.Y = Variable(self.XY[:,1], requires_grad=True).to(self.device)


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
        u = self.model([self.X, self.Y])

        p_xx = self.model.derivative(u, self.X, order=2)
        p_yy = self.model.derivative(u, self.Y, order=2)

        p = p_xx + p_yy

        loss = self.weights[0] * self.criterion(p, torch.zeros_like(p))

        return loss
    

    def loss_function(self):
        """
        Closure function; calculates all losses (IC, BC, PDE)
        """
        self.optimizer.zero_grad()

        # Boundary conditions
        pt_x_tb = Variable(self.x[:2*self.cond_p], requires_grad=True).to(self.device)
        pt_y_tb = Variable(self.y[:2*self.cond_p], requires_grad=True).to(self.device)

        pt_x_lr = Variable(self.x[2*self.cond_p:], requires_grad=True).to(self.device)
        pt_y_lr = Variable(self.y[2*self.cond_p:], requires_grad=True).to(self.device)

        pt_p = Variable(self.p_f, requires_grad=True).to(self.device)

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

        self.optimizer = self.model.set_optimizer('NAdam')

        if self.epoch <= self.Adam_epochs+1:
            for _ in range(self.epoch, self.Adam_epochs+1):
                self.optimizer.step(self.loss_function)
        
        self.optimizer = self.model.set_optimizer('LBFGS')
        self.optimizer.step(self.loss_function)