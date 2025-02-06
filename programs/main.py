import torch
import time
import json
import io
import os

from torch.autograd import Variable
from texttable import Texttable
from datetime import datetime

import programs.conditions as cnd # type: ignore
import programs.misc as misc # type: ignore
import programs.objects as obj # type: ignore
from programs.NN import * # type: ignore


class Poisson_Convection:
    def __init__(self, data : dict
                ):

        # CPU/GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device='cpu'
        torch.set_default_device(self.device)

        # Derivative debug
        torch.autograd.set_detect_anomaly(False)
        
        min_data = {'w'          : float,
                    'mu0'        : float,
                    'cmax'       : float,
                    'u_in'       : float,
                    'chi'        : float,
                    'times'      : list,
                    'size'       : list,
                    'c_cond'     : list,
                    'N_PDE' : int,
                    'N_IC'  : int,
                    'N_BC'  : int,
        }
        
        for i in min_data.keys():
            if not(i in data):
                raise ValueError(f'Value \"{i}\" not found')
            if type(min_data[i])==int and not(type(data[i])==int):
                raise ValueError(f'Value {data[i]} in {i} must be integer')

        self.update_data(data)

        # Make a model
        self.model = Net(input_size  = self.NN_params.get('input_size'), # type: ignore
                         neurons_arr = self.NN_params.get('neurons_arr'),
                         output_size = self.NN_params.get('output_size'),
                         depth       = self.NN_params.get('depth'),
                         act         = self.NN_params.get('act')
                        ).to(self.device)
        
        # Make first arrays of IC, BC
        self.make_distributed_points()
    
    def update_data(self, data:dict):
        # Training parameters
        self.crit_func   = data.setdefault('critfunc', torch.nn.MSELoss())
        self.Adam_epochs = data.setdefault('Adam_epochs', 1000)
        self.epoch       = data.setdefault('epoch', 0)
        self.k           = data.setdefault('k', 10)
        self.max_epoch   = data.setdefault('max_epoch', 50000)
        self.max_iter    = data.setdefault('max_iter',  2)
        self.save_after  = data.setdefault('save_after',  True)
        self.path        = data.setdefault('path',  '')
        self.start       = time.time()
        self.end         = time.time()

        # Weights
        self.weights      = data.setdefault('weights', [1,1,1,1,1,1,1,1])
        self.time_weights = data.setdefault('time_weights', False)
        self.relobralo    = data.setdefault('relobralo', False)
        self.rel_alpha    = data.setdefault('rel_alpha', 0.99)
        self.rel_T        = data.setdefault('rel_T', 0.1)
        self.rel_rho      = data.setdefault('rel_rho', 0.999)
        self.prev_losses  = torch.ones(4).to(self.device)

        # Data loss
        self.dataloss = data.setdefault('dataloss', False)
        if self.dataloss:
            self.x_data = Variable(torch.Tensor(data.setdefault('coordinates_data', [-1]).to(self.device)))
            self.y_data = Variable(torch.Tensor(data.setdefault('coordinates_data', [-1]).to(self.device)))
            self.t_data = Variable(torch.Tensor(data.setdefault('coordinates_data', [-1]).to(self.device)))
            self.calculated_data = torch.Tensor(data.setdefault('calculated_data',  [-1]).to(self.device))
        
        # Loss arrays
        self.losses = data.setdefault('losses', [])
        self.PDE    = data.setdefault('PDE',    [])
        self.BC     = data.setdefault('BC',     [])
        self.IC     = data.setdefault('IC',     [])
        self.corr   = data.setdefault('corr',   [])
        self.data   = data.setdefault('data',   [])

        # Auxiliary tensors 0 and 1
        self.zeros = torch.FloatTensor([0]).to(self.device)
        self.ones  = torch.FloatTensor([1]).to(self.device)
        
        # Physical constants
        self.mu0  = data.get('mu0')
        self.beta = data.setdefault('beta', -2.5)
        self.cmax = data.get('cmax')
        self.u_in = data.get('u_in')

        # Geometry
        self.chi  = data.get('chi')
        self.size = data.get('size')
        self.a    = data.setdefault('a', (self.size[3] - self.chi)/2)
        self.b    = data.setdefault('b', (self.size[3] + self.chi)/2)

        # Initial condition parameters
        self.N_IC = data.get('N_IC')
        self.IC_type   = data.setdefault('IC_type', 'zero')
        self.IC_const  = data.setdefault('IC_const', 0)
        self.band_val  = data.setdefault('band_val', 0.6)
        self.bandshape = data.setdefault('IC_bandshape', [0.2,0.4])

        # Boundary condition parameters
        self.N_BC   = data.get('N_BC')
        self.N_BC2  = self.N_BC**2
        self.c_cond = data.get('c_cond')
        self.times  = data.get('times') 

        # PDE parameters
        self.N_PDE = data.get('N_PDE')
        
        # Coordinates of the "break" points
        self.y_change = torch.FloatTensor([self.a, self.b]).to(self.device)
        self.t_change = torch.FloatTensor(self.times[1:-1]).to(self.device)

        # Crack width parameters
        self._w  = data.get('w')
        self._w1 = data.setdefault('w1', 0.2)
        self._w2 = data.setdefault('w2', 2)
        self._w3 = data.setdefault('w3', 2)
        self._w4 = data.setdefault('w4', 2)
        self._w_name = data.setdefault('w_func', 'const')
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)
        
        # Neural network parameters
        self.act_dict = {'Sin': obj.Sin(), 'Cos': obj.Cos(), 'Ricker':obj.MexicanHat(), 'Morlet':obj.Morlet(), None:obj.Sin()}
        self.NN_params = dict()
        self.NN_params['input_size']  = data.get('NN_params', dict[1:1]).setdefault('input_size',  3)
        self.NN_params['output_size'] = data.get('NN_params', dict[1:1]).setdefault('output_size', 3)
        self.NN_params['neurons_arr'] = data.get('NN_params', dict[1:1]).setdefault('neurons_arr', [48,48,48,48,48])
        self.NN_params['depth']       = data.get('NN_params', dict[1:1]).setdefault('depth', 4)
        self.NN_params['act']         = self.act_dict.get(data.setdefault('NN_params', dict[1:1]).setdefault('act', None))


    def transform(self, net, coords): 
        eps = 0.001
        a = (self.cmax-eps) * torch.sigmoid(500*(net[:,0].clone() - self.cmax))
        net[:,0] *= torch.sigmoid(500*net[:,0]) + torch.sigmoid(500*(self.cmax - net[:,0])) - 1
        net[:,0] += a


    def compute_PDE(self,x,y,t):
        
        prediction_PDE = self.model([x,y,t], self.transform)
        
        c  = prediction_PDE[:,0]
        px = prediction_PDE[:,1]
        py = prediction_PDE[:,2]
        
        w_  = self.w_func(x, y)
        mu =  (1.0 - c / self.cmax).pow(self.beta)

        u_x = -w_**2 * px / (12. * self.mu0 * mu)
        u_y = -w_**2 * py / (12. * self.mu0 * mu)

        c_x = misc.derivative(u_x * c * w_, x)
        c_y = misc.derivative(u_y * c * w_, y)
        c_t = misc.derivative(      c * w_, t)

        u_xx = misc.derivative(w_ * u_x, x)
        u_yy = misc.derivative(w_ * u_y, y)
        
        p_xy = misc.derivative(px, y)
        p_yx = misc.derivative(py, x)

        u_ = (u_xx + u_yy) / w_.max()**3
        c_ = (c_t + c_x + c_y) / w_.max()**3
        
        return c_, u_, p_xy-p_yx


    def loss_function(self):
    
        self.optimizer.zero_grad()
        
        # Initial condition
        if self.weights[3]!=0:
            predictions_IC = self.model([self.x_IC, self.y_IC, self.t_IC], self.transform)[:,0]
            loss_IC = self.weights[3] * self.criterion(predictions_IC, self.c_IC)
            self.IC.append(loss_IC.item())
        
        # Boundary conditions
        prediction_BC = self.model([self.x_BC, self.y_BC, self.t_BC], self.transform)

        prediction_c  = (prediction_BC[:,0][2*self.N_BC2:3*self.N_BC2] *
                         misc.psi(self.y_BC[2*self.N_BC2:3*self.N_BC2], self.chi, self.size[3]))
        loss_BC = self.weights[4] * self.criterion(prediction_c[:self.N_BC2], self.c[2*self.N_BC2:3*self.N_BC2])

        if self.weights[5]!=0:    
            prediction_px = prediction_BC[:,1][2*self.N_BC2:]
            prediction_py = prediction_BC[:,2][:2*self.N_BC2]
            loss_BC += self.weights[5] * (self.criterion(prediction_py[:self.N_BC2], self.p[:self.N_BC2], 3, self.t_BC[:self.N_BC2])                                       +
                                          self.criterion(prediction_py[self.N_BC2:], self.p[self.N_BC2:2*self.N_BC2], 3, self.t_BC[self.N_BC2:2*self.N_BC2])     +
                                          self.criterion(prediction_px[:self.N_BC2], self.p[2*self.N_BC2:3*self.N_BC2], 3, self.t_BC[2*self.N_BC2:3*self.N_BC2]) +
                                          self.criterion(prediction_px[self.N_BC2:], self.p[3*self.N_BC2:], 3, self.t_BC[3*self.N_BC2:])                                   )  

        if self.weights[6]!=0 or self.weights[7]!=0:
            prediction_ux = (prediction_BC[:,1] * -self.w_func(self.x_BC,self.y_BC)**2
                             / 12 / misc.viscosity(self.mu0, prediction_BC[:,0], self.cmax, self.beta))[2*self.N_BC2:]
            prediction_uy = (prediction_BC[:,2] * -self.w_func(self.x_BC,self.y_BC)**2
                             / 12 / misc.viscosity(self.mu0, prediction_BC[:,0], self.cmax, self.beta))[:2*self.N_BC2]  
        
        if self.weights[6]!=0:
            loss_BC += self.weights[6] * (self.criterion(prediction_uy[:self.N_BC2], self.u[:self.N_BC2])                    +
                                          self.criterion(prediction_uy[self.N_BC2:], self.u[self.N_BC2:2*self.N_BC2])   +
                                          self.criterion(prediction_ux[:self.N_BC2], self.u[2*self.N_BC2:3*self.N_BC2]) +
                                          self.criterion(prediction_ux[self.N_BC2:], self.u[3*self.N_BC2:])                  )    
        
        self.BC.append(loss_BC.item())

        # PDE
        if self.weights[0]!=0 or self.weights[1]!=0 or self.weights[2]!=0:
            conv, div, pxy = self.compute_PDE(self.x_PDE, self.y_PDE, self.t_PDE)
            loss_conv = self.criterion(conv, torch.zeros_like(conv), 1, self.t_PDE)
            loss_div  = self.criterion(div,  torch.zeros_like(div),  1, self.t_PDE)
            loss_PDE  = self.weights[0] * loss_conv + self.weights[1] * loss_div
            loss_corr = self.weights[2] * self.criterion(pxy, torch.zeros_like(pxy))

        self.PDE.append(loss_PDE.item())
        self.corr.append(loss_corr.item())

        if self.dataloss:
            pred_data = self.model([self.x_data, self.y_data, self.t_data], self.transform)[:,0]
            loss_data = self.weights[7] * self.criterion(pred_data, self.calculated_data)
            self.data.append(loss_data.item())
        
        if self.dataloss:
            losses = torch.stack([loss_PDE, loss_IC, loss_BC, loss_corr, loss_data]).to(self.device)
            if self.relobralo:
                weights = self.relobralo_func(losses)
            else:
                weights = torch.ones(5)
        else:
            losses = torch.stack([loss_PDE, loss_IC, loss_BC, loss_corr]).to(self.device)
            if self.relobralo:
                weights = self.relobralo_func(losses)
            else:
                weights = torch.ones(4)    
        
        loss = torch.sum(weights * losses)
        loss.backward()
        self.losses.append(loss.item())
        
        if self.epoch % self.k == 0:
            self.end = time.time()
            self.print_tab.add_rows([['|', f'{self.epoch}\t',             '|',
                                    f'{round(loss_PDE.item(),  6)}\t',    '|',
                                    f'{round(loss_corr.item(), 6)}\t',    '|',
                                    f'{round(loss_IC.item(),   6)}\t',    '|',
                                    f'{round(loss_BC.item(),   6)}\t',    '|',
                                    f'{round(self.losses[-1],  6)}\t',    '|',
                                    f'{round(self.end - self.start, 6)}', '|']])
            print(self.print_tab.draw())
            self.start = time.time()
        self.epoch += 1
        return loss
        
    
    def train(self):
        self.print_tab = Texttable()
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,10,1,15,1,15,1,15,1,15,1,15,1,10,1])
        self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','p corr loss','|','IC loss','|','BC loss','|','Summary loss','|','time','|']])
        print(self.print_tab.draw())
        self.model.train()

        for _ in range(self.max_iter):
            self.optimizer = self.model.set_optimizer('NAdam')
            while self.epoch <= self.Adam_epochs+1:
                self.optimizer.step(self.loss_function)
            self.optimizer = self.model.set_optimizer('LBFGS', self.max_epoch)
            self.optimizer.step(self.loss_function)
            self.Adam_epochs += self.epoch
            if self.save_after:
                self.save(self.path+f'/{self.epoch}')


    @staticmethod
    def load(path, loadloss=True):
        with open(path + '.json') as data_file:
            data = json.load(data_file)
        if loadloss:
            with open(path + '_loss.json') as data_file:
                data_loss = json.load(data_file)
            data = {**data, **data_loss}
    
        instance = Poisson_Convection(data)
        # state_dict = torch.load(path+'.pt', )
        instance.model = torch.load(path+'.pt', map_location=instance.device, weights_only=False)
        # instance.model.load_state_dict(torch.load(path+'.pt', weights_only=True))
        return instance
    

    def update_from_file(self, path, loadloss=True):
        with open(path+'.json') as data_file:
            data = json.load(data_file)
        if loadloss:
            with open(path+'_loss.json') as data_file:
                data_loss = json.load(data_file)
            data = {**data, **data_loss}
        self.update_data(data)
        self.model = torch.load(path+'.pt')
        self.make_distributed_points()

    
    def save(self, path='', saveloss=True):

        if path=='':
            current_date = datetime.now()
            folder_name = 'data/'+ current_date.strftime("%Y.%m.%d")
            os.makedirs(folder_name, exist_ok=True)
            path = folder_name + '/' + str(self.epoch)
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            path = path
            
        self.NN_params['act'] = [k for k, v in self.act_dict.items() if v == self.NN_params['act']][0]
        data = {'Adam_epochs' : self.Adam_epochs,
                'epoch'       : self.epoch,
                'k'           : self.k,
                'max_epoch'   : self.max_epoch,
                'max_iter'    : self.max_iter,
                'save_after'  : self.save_after,
                'path'        : self.path,

                'weights'     : self.weights,
                'time_weights': self.time_weights,
                'relobralo'   : self.relobralo,
                'rel_alpha'   : self.rel_alpha,
                'rel_T'       : self.rel_T,
                'rel_rho'     : self.rel_rho,

                'mu0'         : self.mu0,
                'beta'        : self.beta,
                'cmax'        : self.cmax,
                'u_in'        : self.u_in,

                'chi'         : self.chi,
                'size'        : self.size,
                'times'       : self.times,
                'a'           : self.a,
                'b'           : self.b,

                'N_IC'        : self.N_IC,
                'IC_type'     : self.IC_type,
                'IC_const'    : self.IC_const,

                'band_val'    : self.band_val,
                'bandshape'   : self.bandshape,
                'N_BC'        : self.N_BC,
                'c_cond'      : self.c_cond,

                'N_PDE'       : self.N_PDE,

                'w'           : self.w,
                'w1'          : self.w1,
                'w2'          : self.w2,
                'w3'          : self.w3,
                'w4'          : self.w4,
                'w_func'      : self.w_name,

                'NN_params'   : self.NN_params
               }
        
        with io.open(path+'.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                              indent=4, sort_keys=False,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(str(str_))

        if saveloss:
            dataloss = {'losses' : self.losses,
                        'PDE'    : self.PDE,
                        'BC'     : self.BC,
                        'IC'     : self.IC,
                        'corr'   : self.corr,
                        'data'   : self.data
                       }
            with io.open(path+'_loss.json', 'w', encoding='utf8') as outfile:
                str_ = json.dumps(dataloss,
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                outfile.write(str(str_))
        
        torch.save(self.model, path+'.pt')
        self.NN_params['act'] = self.act_dict.get(self.NN_params['act'])


    def criterion(self, pred, true, time_term=0, t=None):
        C = 2
        D = 5
        term = 1
        if self.time_weights:
            if time_term==1:
                term = C * (1 - t / self.size[5]) + 1 
            elif time_term==2:
                term = C * t / self.size[5] + 1 
            elif time_term==3:
                term = 1 - C * (D * t * (t - self.times[0])**2 - t) / self.size[5]
        raw_loss = self.crit_func(term * pred, term * true)
        return raw_loss
        

    def relobralo_func(self, losses):
        ratios = torch.exp(losses / (self.rel_T * self.prev_losses))
        weights = torch.softmax(ratios, dim=0)
        
        if torch.rand(1).item() < self.rel_rho:
            weights = self.prev_losses
        
        updated_weights = self.rel_alpha * self.prev_losses + (1 - self.rel_alpha) * weights
        self.prev_losses = updated_weights.detach()
        return updated_weights


    def Boundary_conditions(self, x, y, t):
        with torch.no_grad():
            psi = misc.psi(y, self.chi, self.size[3])
            c = torch.zeros(len(psi))
            p = torch.zeros(len(psi))
            u = torch.zeros(len(psi))
                                
            left_side_cond = torch.where(x==self.size[0], 1, 0)
            psi_cond = misc.psi(y, self.chi, self.size[3])
            times = [self.size[4]] + self.times + [self.size[5]]
            for i in range(len(times)-1):
                time_start_cond = torch.where(t>=times[i], 1., 0.)
                time_end_cond = torch.where(t<=times[i+1], 1., 0.)
                c = torch.where(time_start_cond + 
                                time_end_cond   +
                                left_side_cond  +
                                psi_cond     == 4,
                                self.c_cond[i], c)
                
                p = torch.where(time_start_cond + 
                                time_end_cond   +
                                left_side_cond  +
                                psi_cond     == 4,
                                -misc.viscosity(self.mu0, self.c_cond[i], self.cmax, self.beta), p) 
                
                u = torch.where(time_start_cond + 
                                time_end_cond   +
                                left_side_cond  +
                                psi_cond     == 4,
                                self.u_in, u) 
            
            right_side_cond = torch.where(x==self.size[1], 1, 0)
            w_right = (psi_cond*self.w_func(self.ones*self.size[0],y)).sum() / (psi_cond*self.w_func(self.ones*self.size[1],y)).sum()
            self.u_out = w_right * self.u_in
            p  = torch.where(right_side_cond+psi_cond==2, -self.mu0 * w_right, p)
            p *= 12 * self.u_in / self.w_func(x, y)**2
            u  = torch.where(right_side_cond+psi_cond==2, self.u_out, u)
            return c, p, u
    
    def Initial_conditions(self, x, y):
        if self.IC_type=='square':
            self.c_IC = self.IC_const*misc.psi(x, self.band_val, self.size[1])*misc.psi(y, self.band_val, self.size[3])
        else:
            self.c_IC = self.IC_const*torch.ones_like(x)     
        self.c_IC += self.c_cond[0] * misc.psi(y, self.chi, self.size[3]) * torch.where(x==0, 1, 0)
        
    def make_BC_dist(self, dist, xy, t):
        sampled_indices = torch.multinomial(dist/dist.sum(), self.N_BC2, replacement=True)
        xy = torch.index_select(xy, -1, sampled_indices)
        t  = torch.index_select(t,  -1, sampled_indices)
        return xy, t

    def generate_linear_points(self, num_points, ranges):
        grids = [torch.linspace(r[0], r[1], num_points).to(self.device) for r in ranges]
        return torch.stack(torch.meshgrid(*grids, indexing='ij')).reshape(len(ranges), -1)

    def generate_random_points(self, num_points, ranges):
        grids = [torch.Tensor(num_points).to(self.device).uniform_(r[0], r[1]) for r in ranges]
        return torch.stack(torch.meshgrid(*grids, indexing='ij')).reshape(len(ranges), -1)

    def make_distributed_points(self):

        x_range = [self.size[0], self.size[1]]
        y_range = [self.size[2], self.size[3]]
        t_range = [self.size[4], self.size[5]]

        # --- Initial Condition ---
        x_linear, y_linear = self.generate_linear_points(int(self.N_IC/2), [x_range, y_range])
        x_random, y_random = self.generate_random_points(self.N_IC,        [x_range, y_range])
        
        try: self.x_IC
        except AttributeError:
            self.x_IC = torch.Tensor([]).to(self.device)
            self.y_IC = torch.Tensor([]).to(self.device)

        x = Variable(torch.cat((x_random, self.x_IC)))
        y = Variable(torch.cat((y_random, self.y_IC)))
        t = Variable(torch.zeros_like(x))

        self.Initial_conditions(x,y)

        ic_dist = (self.model([x,y,t], self.transform)[:,0] - self.c_IC).abs()
        if (ic_dist.sum()==0).item(): ic_dist = torch.ones_like(ic_dist)
        sampled_indices_ic = torch.multinomial(ic_dist/ic_dist.sum(), self.N_IC**2, replacement=True)
            
        self.x_IC = Variable(torch.cat((x_linear, torch.index_select(x, -1, sampled_indices_ic))), requires_grad=True)
        self.y_IC = Variable(torch.cat((y_linear, torch.index_select(y, -1, sampled_indices_ic))), requires_grad=True)
        self.t_IC = Variable(torch.zeros_like(self.x_IC), requires_grad=True)
        
        self.Initial_conditions(self.x_IC, self.y_IC)
        
        # --- PDE Points ---
        x_linear, y_linear, t_linear = self.generate_linear_points(int(self.N_PDE/2), [x_range, y_range, t_range])
        x_random, y_random, t_random = self.generate_random_points(self.N_PDE,        [x_range, y_range, t_range])

        try: self.x_PDE
        except AttributeError:
            self.x_PDE = torch.Tensor([]).to(self.device)
            self.y_PDE = torch.Tensor([]).to(self.device)
            self.t_PDE = torch.Tensor([]).to(self.device)
        
        x = Variable(torch.cat((x_random, self.x_PDE)), requires_grad=True)
        y = Variable(torch.cat((y_random, self.y_PDE)), requires_grad=True)
        t = Variable(torch.cat((t_random, self.t_PDE)), requires_grad=True)

        conv, div, corr = self.compute_PDE(x,y,t)
        pde_dist = self.weights[0] * conv.abs().data + \
                   self.weights[1] * div.abs().data  + \
                   self.weights[2] * corr.abs().data
        sampled_indices_pde = torch.multinomial(pde_dist/pde_dist.sum(), self.N_PDE**3, replacement=True)

        self.x_PDE = Variable(torch.cat((x_linear, torch.index_select(x, -1, sampled_indices_pde))), requires_grad=True)
        self.y_PDE = Variable(torch.cat((y_linear, torch.index_select(y, -1, sampled_indices_pde))), requires_grad=True)
        self.t_PDE = Variable(torch.cat((t_linear, torch.index_select(t, -1, sampled_indices_pde))), requires_grad=True)

        #  --- Boundary Conditions ---
        x = torch.linspace(self.size[0], self.size[1], int(self.N_BC/2))
        y = torch.linspace(self.size[2], self.size[3], int(self.N_BC/2)).to(self.device)
        t = torch.linspace(self.size[4], self.size[5], int(self.N_BC/2)).to(self.device)
        c_condition_linear = cnd.form_boundaries([x, y, t], self.ones, self.zeros)
        self.lin = c_condition_linear
        x = torch.Tensor(self.N_BC).to(self.device).uniform_(self.size[0], self.size[1])
        y = torch.Tensor(self.N_BC).to(self.device).uniform_(self.size[2], self.size[3])
        t = torch.Tensor(self.N_BC).to(self.device).uniform_(self.size[4], self.size[5])
        c_condition_random = cnd.form_boundaries([x, y, t], self.ones, self.zeros)
        self.ran = c_condition_random
        try: self.x_BC
        except AttributeError:
            self.x_BC = torch.Tensor([]).to(self.device)
            self.y_BC = torch.Tensor([]).to(self.device)
            self.t_BC = torch.Tensor([]).to(self.device)

        x = Variable(torch.cat((self.x_BC[self.N_BC2:], c_condition_random[:,0])), requires_grad=True)
        y = Variable(torch.cat((self.y_BC[self.N_BC2:], c_condition_random[:,1])), requires_grad=True)
        t = Variable(torch.cat((self.t_BC[self.N_BC2:], c_condition_random[:,2])), requires_grad=True)

        N = self.N_BC2

        c, p, _ = self.Boundary_conditions(x,y,t)

        bc_dist = self.model([x,y,t], self.transform)
        
        bc_dist_top    = torch.cat(((bc_dist[:,2][:N] - p[:N]).abs(), (bc_dist[:,2][4*N:5*N] - p[4*N:5*N]).abs()))
        xtop, t1       = self.make_BC_dist(bc_dist_top, torch.cat((x[:N], x[4*N:5*N])), torch.cat((t[:N], t[4*N:5*N])))

        bc_dist_bottom = torch.cat(((bc_dist[:,2][N:2*N] - p[N:2*N]).abs(), (bc_dist[:,2][5*N:6*N] - p[5*N:6*N]).abs()))
        xbottom, t2    = self.make_BC_dist(bc_dist_bottom, torch.cat((x[N:2*N], y[5*N:6*N])), torch.cat((t[N:2*N], t[5*N:6*N])))

        bc_dist_left   = torch.cat((
                         (bc_dist[:,0] * misc.psi(y, self.chi, self.size[3]) - c)[2*N:3*N].abs() + \
                         (bc_dist[:,1] - p)[2*N:3*N].abs(),

                         (bc_dist[:,0] * misc.psi(y, self.chi, self.size[3]) - c)[6*N:7*N].abs() + \
                         (bc_dist[:,1] - p)[6*N:7*N].abs()))

        yleft, t3      = self.make_BC_dist(bc_dist_left, torch.cat((y[2*N:3*N], x[6*N:7*N])), torch.cat((t[2*N:3*N], t[6*N:7*N])))

        bc_dist_right  = torch.cat(((bc_dist[:,1][3*N:4*N] - p[3*N:4*N]).abs(), (bc_dist[:,1][7*N:] - p[7*N:]).abs()))
        yright, t4     = self.make_BC_dist(bc_dist_right, torch.cat((y[3*N:4*N], y[7*N:])), torch.cat((t[3*N:4*N], t[7*N:])))

        self.x_BC = Variable(torch.cat((c_condition_linear[:,0], xtop, xbottom, torch.zeros(self.N_BC2), torch.ones(self.N_BC2))), requires_grad=True)
        self.y_BC = Variable(torch.cat((c_condition_linear[:,1], torch.ones(self.N_BC2), torch.zeros(self.N_BC2), yleft, yright)), requires_grad=True)
        self.t_BC = Variable(torch.cat((c_condition_linear[:,2], t1, t2, t3, t4)), requires_grad=True)
        
        self.c, self.p, self.u = self.Boundary_conditions(self.x_BC, self.y_BC, self.t_BC)
    

    def eval(self):
        self.model.eval()

    def get_c(self, x, y, t):
        with torch.no_grad():
            c = self.model([x,y,t], self.transform)[:,0]
            return c.data.cpu().numpy()

    def get_px(self, x, y, t):
        with torch.no_grad():
            px = self.model([x,y,t], self.transform)[:,1]
            return px.data.cpu().numpy()

    def get_py(self, x, y, t):
        with torch.no_grad():
            py = self.model([x,y,t], self.transform)[:,2]
            return py.data.cpu().numpy()

    def get_ux(self, x, y, t):
        with torch.no_grad():
            pred = self.model([x,y,t], self.transform)
            ux = pred[:,1] * self.w_func(x,y)**2 / (-12 * misc.viscosity(self.mu0, pred[:,0], self.cmax, self.beta))
            return ux.data.cpu().numpy()

    def get_uy(self, x, y, t):
        with torch.no_grad():
            pred = self.model([x,y,t], self.transform)
            uy = pred[:,2] * self.w_func(x,y)**2 / (-12 * misc.viscosity(self.mu0, pred[:,0], self.cmax, self.beta))
            return uy.data.cpu().numpy()
    
    def get_mu(self, x, y, t):
        with torch.no_grad():
            mu = misc.viscosity(self.mu0, self.model([x,y,t], self.transform)[:,0], self.cmax, self.beta)
            return mu.data.cpu().numpy()

    def get_conv(self, x, y, t):
        pred = self.model([x,y,t], self.transform)
        ux = pred[:,1] * self.w_func(x,y)**2 / (-12 * misc.viscosity(self.mu0, pred[:,0], self.cmax, self.beta))
        uy = pred[:,2] * self.w_func(x,y)**2 / (-12 * misc.viscosity(self.mu0, pred[:,0], self.cmax, self.beta))
        с_t = misc.derivative(self.w_func(x,y)*pred[:,0], t).data.cpu().numpy()
        c_x = misc.derivative(self.w_func(x,y)*pred[:,0]*ux, x).data.cpu().numpy()
        c_y = misc.derivative(self.w_func(x,y)*pred[:,0]*uy, y).data.cpu().numpy()
        conv =  c_x + c_y + с_t
        return conv

    def get_div(self, x, y, t):
        pred = self.model([x,y,t], self.transform)
        ux = pred[:,1] * self.w_func(x,y)**2 / (-12 * misc.viscosity(self.mu0, pred[:,0], self.cmax, self.beta))
        uy = pred[:,2] * self.w_func(x,y)**2 / (-12 * misc.viscosity(self.mu0, pred[:,0], self.cmax, self.beta))
        uxx = misc.derivative(ux,x).data.cpu().numpy()
        uyy = misc.derivative(uy,y).data.cpu().numpy()
        div = uxx + uyy
        return div
        
    def get_corr(self, x, y, t):
        pred = self.model([x,y,t], self.transform)
        pxy = misc.derivative(pred[:,1],y).data.cpu().numpy()
        pyx = misc.derivative(pred[:,2],x).data.cpu().numpy()
        corr = pxy - pyx
        return corr

    @property
    def w(self):
        return self._w
    @w.setter
    def w(self, w):
        self._w = w
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)

    @property
    def w1(self):
        return self._w1
    @w1.setter
    def w1(self, w1):
        self._w1 = w1
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)

    @property
    def w2(self):
        return self._w2
    @w2.setter
    def w2(self, w2):
        self._w2 = w2
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)

    @property
    def w3(self):
        return self._w3
    @w3.setter
    def w3(self, w3):
        self._w3 = w3
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)

    @property
    def w4(self):
        return self._w4
    @w4.setter
    def w4(self, w4):
        self._w4 = w4
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)

    @property
    def w_name(self):
        return self._w_name
    @w_name.setter
    def w_name(self, w_name):
        self._w_name = w_name
        self.w_func = obj.Width(self._w_name, self._w, self._w1, self._w2, self._w3, self._w4)