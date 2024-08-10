import numpy as np 
import os
import torch

from torch.autograd import Variable
from texttable import Texttable
from copy import deepcopy

import programs.conditions as cnd
from programs.NN import *


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

        self.PDE = []
        self.BC = []
        self.IC = []

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

        # Coords for PDE
        x_collocation = torch.linspace(self.size[0]+0.01, self.size[1]-0.01, self.collocation).to(self.device)
        y_collocation = torch.linspace(self.size[2]+0.01, self.size[3]-0.01, self.collocation).to(self.device)
        t_collocation = torch.linspace(self.size[4], self.size[5], self.collocation).to(self.device)

        self.XYT = torch.stack(torch.meshgrid(x_collocation, y_collocation, t_collocation)).reshape(3, -1).T
        self.XYT.requires_grad = True
        self.X = Variable(self.XYT[:,0], requires_grad=True).to(self.device)
        self.Y = Variable(self.XYT[:,1], requires_grad=True).to(self.device)
        self.T = Variable(self.XYT[:,2], requires_grad=True).to(self.device)


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
        u[:,0] = torch.clamp(u[:,0].clone(), max=self.cmax-0.001)

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
        self.IC.append(loss_IC.item())
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
        prediction_left_right = self.model([pt_x_lr, pt_y_lr, pt_t_lr])
        prediction_BC = torch.cat((prediction_top_bottom, prediction_left_right))

        pt_c = Variable(self.c_f, requires_grad=True).to(self.device)
        pt_p = Variable(self.p_f, requires_grad=True).to(self.device)

        prediction_p = self.check_BC(self.p_cond[1], prediction_BC[:,1], prediction_top_bottom[:,1], prediction_left_right[:,1], pt_x_lr, pt_y_tb)
        prediction_c = self.check_BC(self.c_cond[1], prediction_BC[:,0], prediction_top_bottom[:,0], prediction_left_right[:,0], pt_x_lr, pt_y_tb)

        loss_BC = self.weights[3] * torch.sqrt(self.criterion(prediction_p, pt_p)) + self.weights[4] * torch.sqrt(self.criterion(prediction_c, pt_c))
        self.BC.append(loss_BC.item())
        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE
        loss_PDE = self.PDELoss()
        self.PDE.append(loss_PDE.item())
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
                self.optimizer = self.model.set_optimizer('NAdam')
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
                self.optimizer = self.model.set_optimizer('NAdam')
                for _ in range(self.CL_epochs[ijk]):
                    self.optimizer.step(self.loss_function)
                
                self.optimizer = self.model.set_optimizer('LBFGS')
                self.optimizer.step(self.loss_function)

                self.full_save(f'data/CL_w,{constants["w"]}/{param}', f'data/CL_w,{constants["w"]}/{param}_data')
                ijk+=1
        self.CL=False
    

    def eval_(self):
        self.model.eval()
