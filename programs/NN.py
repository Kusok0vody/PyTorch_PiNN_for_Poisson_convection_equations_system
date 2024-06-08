import numpy as np 

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad
from torch import FloatTensor
from texttable import Texttable


class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.neurons_per_layer = 20
        self.fc1 = nn.Linear(3, self.neurons_per_layer)
        self.fc2 = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
        self.fc3 = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
        self.fc4 = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
        self.fc5 = nn.Linear(self.neurons_per_layer, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.sin = torch.sin

        # Объявление констант
        self.w = 1
        self.mu0 = 1
        self.cmax = 1
        self.c_BC_x = self.c_BC_y = self.c_BC_t = self.c_BC_f = None
        self.p_BC_x = self.p_BC_y = self.p_BC_t = self.p_BC_f = None
        self.IC_x = self.IC_y = self.IC_t = self.IC_f = None
        self.collocation = 1000
        self.weights = [1,1,1]
        self.optim = 'Adam'
        self.boundaries = []
        self.device = device

        self.loss_PDE = self.loss_BC = self.loss_IC = 0
        self.losses=[0]
        self.epoch = 0
        self.print_tab = Texttable()
        self.criterion = torch.nn.MSELoss()


    def forward(self, x, y, t):
        inputs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), t.reshape(-1, 1)], axis=1)
        output = self.sin(self.fc1(inputs))
        output = self.sin(self.fc2(output))
        output = self.sin(self.fc3(output))
        output = self.sin(self.fc4(output))
        output = self.fc5(output) 
        return output


    def load(self,path):
        self.load_state_dict(torch.load(path))


    def save(self, path):
        torch.save(self.state_dict(), path)
    

    def full_load(self, path_nn, path_data):
        self.load_state_dict(torch.load(path_nn))
        data = np.load(path_data)
        # self.w = data[0]
        # self.mu0 = data[1]
        # self.cmax = data[2]
        # self.c_BC_x, self.c_BC_y, self.c_BC_t, self.c_BC_f = data[3], data[4], data[5], data[6] 
        # self.p_BC_x, self.p_BC_y, self.p_BC_t, self.p_BC_f = data[7], data[8], data[9], data[10] 
        # self.IC_x = self.IC_y = self.IC_t = self.IC_f = data[11], data[12], data[13], data[14] 
        # self.collocation = data[15]
        # self.loss_PDE = self.loss_BC = self.loss_IC = 0
        self.losses = data[0]


    def full_save(self, path_nn, path_data):
        torch.save(self.state_dict(), path_nn)
        # data = [self.w, self.mu0, self.cmax,
        #                     self.c_BC_x.cpu().numpy(), self.c_BC_y.cpu().numpy(), self.c_BC_t.cpu().numpy(), self.c_BC_f.cpu().numpy(), 
        #                     self.p_BC_x.cpu().numpy(), self.p_BC_y.cpu().numpy(), self.p_BC_t.cpu().numpy(), self.p_BC_f.cpu().numpy(),
        #                     self.IC_x.cpu().numpy(), self.IC_y.cpu().numpy(), self.IC_t.cpu().numpy(), self.IC_f.cpu().numpy(),
        #                     self.collocation, np.array(self.losses)]
        data = [self.losses]
        np.save(path_data, data)


    def optimizer(self, optimizer_type):
        if optimizer_type=='Adam':
            return torch.optim.Adam(self.parameters())
        if optimizer_type=='LBFGS':
            return torch.optim.LBFGS(self.parameters(),
                                     lr=15.0, 
                                     max_iter=50, 
                                     max_eval=50, 
                                     history_size=20,
                                     tolerance_grad=1e-7, 
                                     tolerance_change=1.0 * np.finfo(float).eps,
                                     line_search_fn="strong_wolfe")


    def PDELoss(self, x, y, t):
        u = self(x,y,t)
        u[:,1] = torch.clamp(u[:,1].clone(), min=0, max=self.cmax-0.0000001)

        p_x = grad(u[:,0], x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u[:,0]))[0]
        p_y = grad(u[:,0], y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u[:,0]))[0]

        mu = self.mu0 * (1 - u[:,1] / self.cmax).pow(-2.5)

        v_x = -self.w**2 * p_x / (12 * mu)
        v_y = -self.w**2 * p_y / (12 * mu)

        c_x = grad(u[:,1]*v_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u[:,1]*v_x))[0]
        c_y = grad(u[:,1]*v_y, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u[:,1]*v_y))[0]
        c_t = grad(u[:,1], t, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u[:,1]))[0]
        
        p_xx = grad(p_x / mu, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(p_x / mu))[0]
        p_yy = grad(p_y / mu, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(p_y / mu))[0]

        c = c_t + c_x + c_y
        p = p_xx + p_yy
        
        # c = c.view(len(c), -1)
        # p = p.view(len(p), -1)

        loss_p = self.criterion(torch.zeros_like(p), p)
        loss_c = self.criterion(torch.zeros_like(c), c)

        loss = loss_p + loss_c

        return loss
    

    def loss_func(self):
        
        self.optimizer(self.optim).zero_grad()
                
        # Точки коллокации
        x_collocation = FloatTensor(self.collocation,).uniform_(self.boundaries[0], self.boundaries[1]).to(self.device)
        y_collocation = FloatTensor(self.collocation,).uniform_(self.boundaries[2], self.boundaries[3]).to(self.device)
        t_collocation = FloatTensor(self.collocation,).uniform_(self.boundaries[4], self.boundaries[5]).to(self.device)

        x_collocation = Variable(x_collocation, requires_grad=True).to(self.device)
        y_collocation = Variable(y_collocation, requires_grad=True).to(self.device)
        t_collocation = Variable(t_collocation, requires_grad=True).to(self.device)

        # Начальные условия

        pt_x_ic = Variable(self.IC_x, requires_grad=True).to(self.device)
        pt_y_ic = Variable(self.IC_y, requires_grad=True).to(self.device)
        pt_t_ic = Variable(self.IC_t, requires_grad=True).to(self.device)

        pt_p_IC = Variable(self.IC_f[:,0], requires_grad=True).to(self.device)
        pt_c_IC = Variable(self.IC_f[:,1], requires_grad=True).to(self.device)

        predictions_IC = self(pt_x_ic, pt_y_ic, pt_t_ic)

        self.loss_IC = self.criterion(predictions_IC[:,0], pt_p_IC) + self.criterion(predictions_IC[:,1], pt_c_IC)
        if torch.isnan(self.loss_IC)==True:
            raise ValueError("Достигнуто значение nan")

        # ГУ концентрации (Дирихле)
        pt_c_x = Variable(self.c_BC_x, requires_grad=True).to(self.device)
        pt_c_y = Variable(self.c_BC_y, requires_grad=True).to(self.device)
        pt_c_t = Variable(self.c_BC_t, requires_grad=True).to(self.device)
        pt_c_f = Variable(self.c_BC_f, requires_grad=True).to(self.device)

        prediction_c = self(pt_c_x, pt_c_y, pt_c_t)[:,1]

        # ГУ давления (Нейман)
        pt_p_x = Variable(self.p_BC_x, requires_grad=True).to(self.device)
        pt_p_y = Variable(self.p_BC_y, requires_grad=True).to(self.device)
        pt_p_t = Variable(self.p_BC_t, requires_grad=True).to(self.device)
        pt_p_f = Variable(self.p_BC_f, requires_grad=True).to(self.device)

        prediction_p = self(pt_p_x, pt_p_y, pt_p_t)[:,0]
        prediction_p = grad(prediction_p, pt_p_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(prediction_p))[0]

        self.loss_BC = self.criterion(pt_c_f, prediction_c) + self.criterion(pt_p_f,prediction_p)
        if torch.isnan(self.loss_BC)==True:
            raise ValueError("Достигнуто значение nan")

        # PDE loss
        self.loss_PDE = self.PDELoss(x_collocation, y_collocation, t_collocation)
        if torch.isnan(self.loss_PDE)==True:
            raise ValueError("Достигнуто значение nan")

        loss = self.weights[0] * self.loss_PDE + self.weights[1] * self.loss_IC + self.weights[2] * self.loss_BC
        # self.losses.append(loss.item())
        loss.backward()
        return loss


    def train(self, epochs, max_loss):
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,15,1,25,1,25,1,25,1,25,1])
        self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','IC loss','|','BC loss','|','Summary loss','|']])
        print(self.print_tab.draw())
        
        optimizer = self.optimizer(self.optim)
        
        for epoch in range(epochs+1):
            self.epoch = epoch
            if self.epoch % 1 == 0:
                self.print_tab.add_rows([['|',f'{self.epoch}\t','|',
                                    f'{self.weights[0] * self.loss_PDE}\t','|',
                                    f'{self.weights[1] * self.loss_IC}\t','|',
                                    f'{self.weights[2] * self.loss_BC}\t','|',
                                    f'{self.losses[-1]}\t','|']])
                print(self.print_tab.draw())
            g = optimizer.step(self.loss_func)
            self.losses.append(g.item())
            if self.losses[-1] < max_loss:
                break

    