import torch
import time
import json
import io

from torch.autograd import Variable
from texttable import Texttable

import programs.conditions as cnd
import programs.misc as misc
import programs.objects as obj
from programs.NN import *


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

        # Make a model
        self.model = Net(input_size  = self.NN_params.get('input_size'),
                         neurons_arr = self.NN_params.get('neurons_arr'),
                         output_size = self.NN_params.get('output_size'),
                         depth       = self.NN_params.get('depth'),
                         act = obj.Sin(1))

        self.optimizer = self.model.set_optimizer('NAdam')
        
        # Make first arrays of IC, BC
        self.make_distributed_points()
    
    def update_data(self, data:dict):
        # Iteration variables
        self.Adam_epochs = data.setdefault('Adam_epochs', 1000)
        self.epoch       = data.setdefault('epoch', 0)
        self.k           = data.setdefault('k', 10)
        self.max_epoch   = data.setdefault('max_epoch',50000)

        # Loss arrays
        self.losses = data.setdefault('losses', [])
        self.PDE    = data.setdefault('PDE',    [])
        self.BC     = data.setdefault('BC',     [])
        self.IC     = data.setdefault('IC',     [])
        self.corr   = data.setdefault('corr',   [])

        # Auxiliary tensors 0 and 1
        self.zeros = torch.FloatTensor([0]).to(self.device)
        self.ones  = torch.FloatTensor([1]).to(self.device)

        # Training parameters
        self.criterion = torch.nn.MSELoss()
        self.weights   = data.setdefault('weights', [1,1,1,1,1,1,1])
        self.adaptive_points = True
        self.adaptive_freq = data.setdefault('adaptive_freq', 10000)
        
        # Physical constants
        self.mu0  = data.get('mu0')
        self.beta = data.setdefault('beta', -2.5)
        self.cmax = data.get('cmax')
        self.u_in = data.get('u_in')

        # Geometry
        self.chi    = data.get('chi')
        self.size   = data.get('size')
        self.a      = data.setdefault('a', (self.size[3] - self.chi)/2)
        self.b      = data.setdefault('b', (self.size[3] + self.chi)/2)

        # Initial condition parameters
        self.IC_points = data.get('IC_points')
        self.IC_type   = data.setdefault('IC_type', 'zero')
        self.IC_const  = data.setdefault('IC_const', 0)
        self.band_val  = data.setdefault('band_val', 0.6)
        self.bandshape = data.setdefault('IC_bandshape', [0.2,0.4])

        # Boundary condition parameters
        self.BC_points   = data.get('BC_points')
        self.BC_points2  = self.BC_points**2
        self.c_cond      = data.get('c_cond')
        self.times       = data.get('times') 
        self.compare_vel = data.setdefault('compare_vel', False)

        # PDE parameters
        self.PDE_points = data.get('PDE_points')
        
        # Coordinates of the "break" points
        self.y_change = torch.FloatTensor([self.a, self.b]).to(self.device)
        self.t_change = torch.FloatTensor(self.times[1:-1]).to(self.device)

        # Crack width parameters
        self.w  = data.get('w')
        self.w1 = data.setdefault('w1', 0.2)
        self.w2 = data.setdefault('w2', 2)
        self.w3 = data.setdefault('w3', 2)
        self.w4 = data.setdefault('w4', 2)
        self.w_func = obj.Width(data.setdefault('w_func', 'const'), self.w, self.w1, self.w2, self.w3, self.w4)
        
        # Neural network parameters
        self.NN_params = {}
        if data.get('NN_params')!=None:
            self.NN_params = data.get('NN_params')
        else:
            self.NN_params['input_size']  = 3
            self.NN_params['neurons_arr'] = [48,48,48,48,48]
            self.NN_params['output_size'] = 3
            self.NN_params['depth']       = 4

    
    def load(self, path, loadloss=True):
        with open(path+'.json') as data_file:
            data = json.load(data_file)
        if loadloss:
            with open(path+'_loss.json') as data_file:
                data_loss = json.load(data_file)
            data = {**data, **data_loss}
        self.update_data(data)
        self.make_distributed_points()
        self.model.load_state_dict(torch.load(path, weights_only=True))

    
    def save(self, path, saveloss=True):
        data = {'Adam_epochs' : self.Adam_epochs,
                'epoch'       : self.epoch,
                'k'           : self.k,
                'max_epoch'   : self.max_epoch,
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
                'IC_points'   : self.IC_points,
                'IC_type'     : self.IC_type,
                'IC_const'    : self.IC_const,
                'band_val'    : self.band_val,
                'bandshape'   : self.bandshape,
                'BC_points'   : self.BC_points,
                'c_cond'      : self.c_cond,
                'compare_vel' : self.compare_vel,
                'PDE_points'  : self.PDE_points,
                'w'           : self.w,
                'w1'          : self.w1,
                'w2'          : self.w2,
                'w3'          : self.w3,
                'w4'          : self.w4,
                'w_func'      : self.w_func.name,
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
                        'corr'   : self.corr
                       }
            with io.open(path+'_loss.json', 'w', encoding='utf8') as outfile:
                str_ = json.dumps(dataloss,
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                outfile.write(str(str_))
        
        torch.save(self.model.state_dict(), path)


    def update_width(self, func_name, w=None, w1=None, w2=None, w3=None, w4=None):
        if w!=None:  self.w  = w
        if w1!=None: self.w1 = w1
        if w2!=None: self.w2 = w2
        if w3!=None: self.w3 = w3
        if w4!=None: self.w4 = w4
        self.w_func = obj.Width(func=func_name, w=self.w, w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4)
        
    
    def Boundary_conditions(self, x, y, t):
        psi = misc.psi(y, self.chi, self.size[3])
        self.c = torch.zeros(len(psi))
        self.p = torch.zeros(len(psi))
                            
        left_side_cond = torch.where(x==self.size[0], 1, 0)
        psi_cond = misc.psi(y, self.chi, self.size[3])
        times = [self.size[4]] + self.times + [self.size[5]]
        for i in range(len(times)-1):
            time_start_cond = torch.where(t>=times[i], 1., 0.)
            time_end_cond = torch.where(t<=times[i+1], 1., 0.)
            self.c = torch.where(time_start_cond + 
                                 time_end_cond   +
                                 left_side_cond  +
                                 psi_cond     == 4,
                                 self.c_cond[i], self.c)
            
            self.p = torch.where(time_start_cond + 
                                 time_end_cond   +
                                 left_side_cond  +
                                 psi_cond     == 4,
                                 -misc.viscosity(self.mu0, self.c_cond[i], self.cmax, self.beta), self.p) 
            
        right_side_cond = torch.where(x==self.size[1], 1, 0)
        self.p  = torch.where(right_side_cond+psi_cond==2, -self.mu0, self.p)
        self.p *= 12 * self.u_in / self.w_func(x, y)**2
        
    
    def make_BC_dist(self, dist):
        norm_dist = dist/dist.sum()
        sampled_indices = torch.multinomial(norm_dist, self.BC_points2, replacement=True)

        t = (sampled_indices % self.BC_points) / self.BC_points * (self.size[1] - self.size[0]) + self.size[0]
        xy = (sampled_indices // self.BC_points) / self.BC_points * (self.size[3] - self.size[2]) + self.size[2]
        return xy, t

    
    def make_distributed_points(self):
        # IC
        with torch.no_grad():
            x = torch.linspace(self.size[0], self.size[1],self.IC_points)
            y = torch.linspace(self.size[2], self.size[3],self.IC_points)
            xy = torch.stack(torch.meshgrid(x, y, indexing='ij')).reshape(2, -1)
            
            x = Variable(xy[1])
            y = Variable(xy[0])
            t = Variable(torch.zeros_like(xy[1]))

            if self.epoch!=0:
                ic_dist = (self.model([x,y,t], self.transform)[:,0] - self.c_IC).abs()
            else:
                ic_dist = torch.ones(x.shape)
            norm_ic = ic_dist/ic_dist.sum()

            sampled_indices_ic = torch.multinomial(norm_ic, self.IC_points**2, replacement=True)
            
            x = (sampled_indices_ic % self.IC_points) / self.IC_points * (self.size[1] - self.size[0]) + self.size[0]
            y = (sampled_indices_ic // self.IC_points) / self.IC_points * (self.size[3] - self.size[2]) + self.size[2]
            
        self.x_IC = Variable(x, requires_grad=True)
        self.y_IC = Variable(y, requires_grad=True)
        self.t_IC = Variable(torch.zeros_like(x), requires_grad=True)
        if self.IC_type=='square':
            self.c_IC = self.IC_const*misc.psi(self.x_IC, self.band_val, self.size[1])*misc.psi(self.y_IC, self.band_val, self.size[3])
        else:
            self.c_IC = self.IC_const*torch.ones_like(self.x_IC)
        self.c_IC += self.c_cond[0] * misc.psi(self.y_IC, self.chi, self.size[3]) * torch.where(self.x_IC==0, 1, 0)
        
        # PDE
        x = torch.linspace(self.size[0], self.size[1],self.PDE_points)
        y = torch.linspace(self.size[2], self.size[3],self.PDE_points)
        t = torch.linspace(self.size[4], self.size[5],self.PDE_points)
        xy = torch.stack(torch.meshgrid(x, y, t, indexing='ij')).reshape(3, -1)
        x = Variable(xy[0], requires_grad=True)
        y = Variable(xy[1], requires_grad=True)
        t = Variable(xy[2], requires_grad=True)
        conv, div, pxy = self.compute_PDE(x,y,t)
        
        with torch.no_grad():
            pde_dist = self.weights[0]*conv.abs().data + self.weights[1]*div.abs().data + self.weights[2]*pxy.abs().data
        
            norm_pde = pde_dist/pde_dist.sum()
            sampled_indices_pde = torch.multinomial(norm_pde, self.PDE_points**3, replacement=True)
            
            X = ((sampled_indices_pde % self.PDE_points) / self.PDE_points) * (self.size[1] - self.size[0])
            Y = ((sampled_indices_pde % (self.PDE_points * self.PDE_points) // self.PDE_points) / self.PDE_points) * (self.size[3] - self.size[2])
            T = ((sampled_indices_pde // (self.PDE_points * self.PDE_points)) / self.PDE_points) * (self.size[5] - self.size[4])

        self.X = Variable(T, requires_grad=True)
        self.Y = Variable(Y, requires_grad=True)
        self.T = Variable(X, requires_grad=True)
        
        # BC
        with torch.no_grad():
            x = torch.linspace(self.size[0], self.size[1], self.BC_points).to(self.device)
            y = torch.cat((torch.linspace(self.size[2], self.size[3], self.BC_points-2).to(self.device), self.y_change)).sort()[0]
            t = torch.cat((torch.linspace(self.size[4], self.size[5], self.BC_points-len(self.times[1:-1])).to(self.device), self.t_change)).sort()[0]
            
            psi = misc.psi(y, self.chi, self.size[3])
            psi = torch.stack(torch.meshgrid(psi, torch.zeros(self.BC_points), indexing='ij')).reshape(2, -1).T[:,0]
    
            zeros = torch.zeros((4, len(psi)))
            c_condition = cnd.form_boundaries([x, y, t], zeros, self.ones, self.zeros)

            x = Variable(c_condition[:,1], requires_grad=True)
            y = Variable(c_condition[:,2], requires_grad=True)
            t = Variable(c_condition[:,3], requires_grad=True)
            self.Boundary_conditions(x,y,t)

            bc_dist = self.model([x,y,t], self.transform)
                
            bc_dist_top    = (bc_dist[:,2][:self.BC_points2] - self.p[:self.BC_points2]).abs()
            xtop, t1 = self.make_BC_dist(bc_dist_top)

            bc_dist_bottom = (bc_dist[:,2][self.BC_points2:2*self.BC_points2] - self.p[self.BC_points2:2*self.BC_points2]).abs()
            xbottom, t2 = self.make_BC_dist(bc_dist_bottom)

            bc_dist_left   = ((bc_dist[:,0][2*self.BC_points2:3*self.BC_points2] *
                              misc.psi(y[2*self.BC_points2:3*self.BC_points2], self.chi, self.size[3]) -
                              self.c[2*self.BC_points2:3*self.BC_points2]).abs() +
                              (bc_dist[:,1][2*self.BC_points2:3*self.BC_points2] -
                               self.p[2*self.BC_points2:3*self.BC_points2]).abs())
            yleft, t3 = self.make_BC_dist(bc_dist_left)

            bc_dist_right  = (bc_dist[:,1][3*self.BC_points2:] - self.p[3*self.BC_points2:]).abs()
            yright, t4 = self.make_BC_dist(bc_dist_right)

            x = torch.cat((xtop,xbottom,torch.zeros(self.BC_points2),torch.ones(self.BC_points2)))
            y = torch.cat((torch.ones(self.BC_points2),torch.zeros(self.BC_points2),yleft,yright))
            t = torch.cat((t1,t2,t3,t4))
            self.Boundary_conditions(x,y,t)

        self.x_BC = Variable(x, requires_grad=True)
        self.y_BC = Variable(y, requires_grad=True)
        self.t_BC = Variable(t, requires_grad=True)

    
    def transform(self, net, coords): 
        # Ограничение концентрации от 0 до cmax
        eps = 0.001
        a = (self.cmax-eps) * torch.sigmoid(500*(net[:,0].clone() - self.cmax))
        net[:,0] *= torch.sigmoid(500*net[:,0]) + torch.sigmoid(500*(self.cmax - net[:,0])) - 1
        net[:,0] += a


    def compute_PDE(self,x,y,t):
        
        prediction_PDE = self.model([x,y,t], self.transform)
        
        c  = prediction_PDE[:,0]
        px = prediction_PDE[:,1]
        py = prediction_PDE[:,2]
        
        mu =  (1.0 - c / self.cmax).pow(self.beta)

        u_x = -self.w_func(x, y)**2 * px / (12. * self.mu0 * mu)
        u_y = -self.w_func(x, y)**2 * py / (12. * self.mu0 * mu)

        c_x = misc.derivative(u_x * c * self.w_func(x, y), x)
        c_y = misc.derivative(u_y * c * self.w_func(x, y), y)
        c_t = misc.derivative(      c * self.w_func(x, y), t)

        u_xx = misc.derivative(u_x, x)
        u_yy = misc.derivative(u_y, y)

        p_xy = misc.derivative(px, y)
        p_yx = misc.derivative(py, x)

        u_ = (u_xx + u_yy)
        c_ = (c_t + c_x + c_y)
        
        return c_, u_, p_xy-p_yx

        
    def loss_function(self):
        
        start = self.k * time.time()
        self.optimizer.zero_grad()
        
        # Initial condition
        predictions_IC = self.model([self.x_IC, self.y_IC, self.t_IC], self.transform)[:,0]
        loss_IC = self.weights[3] * self.criterion(predictions_IC, self.c_IC)
        self.IC.append(loss_IC.item())
        if torch.isnan(loss_IC)==True:
            raise ValueError("nan value reached")
        
        # Boundary conditions
        prediction_BC = self.model([self.x_BC, self.y_BC, self.t_BC], self.transform)

        prediction_c  = (prediction_BC[:,0][2*self.BC_points2:3*self.BC_points2] *
                         misc.psi(self.y_BC[2*self.BC_points2:3*self.BC_points2], self.chi, self.size[3]))
        prediction_px = prediction_BC[:,1][2*self.BC_points2:]
        prediction_py = prediction_BC[:,2][:2*self.BC_points2]
        
        loss_BC = self.weights[4] * (self.criterion(prediction_py[:self.BC_points2], self.p[:self.BC_points2])                    +
                                     self.criterion(prediction_py[self.BC_points2:], self.p[self.BC_points2:2*self.BC_points2])   +
                                     self.criterion(prediction_px[:self.BC_points2], self.p[2*self.BC_points2:3*self.BC_points2]) +
                                     self.criterion(prediction_px[self.BC_points2:], self.p[3*self.BC_points2:])                  )    
                      
        loss_BC += self.weights[5] * self.criterion(prediction_c[:self.BC_points2], self.c[2*self.BC_points2:3*self.BC_points2])

        if self.compare_vel:
            ux = (prediction_px * self.w_func(self.x_BC[2*self.BC_points2:], self.y_BC[2*self.BC_points2:])**2 / -12. /
                  misc.viscosity(self.mu0, prediction_BC[:,0][2*self.BC_points2:], self.cmax, self.beta))
            loss_BC += self.weights[6] * self.criterion(ux[:self.BC_points2],ux[self.BC_points2:])

            uy = (prediction_py * self.w_func(self.x_BC[:2*self.BC_points2], self.y_BC[:2*self.BC_points2])**2 / -12. /
                  misc.viscosity(self.mu0, prediction_BC[:,0][:2*self.BC_points2], self.cmax, self.beta))
            loss_BC += self.weights[6] * self.criterion(uy[:self.BC_points2],uy[self.BC_points2:])
        
        self.BC.append(loss_BC.item())
            
        if torch.isnan(loss_BC)==True:
            raise ValueError("nan value reached")

        # PDE   
        conv, div, pxy = self.compute_PDE(self.X, self.Y, self.T)
        loss_c   = self.criterion(conv, torch.zeros_like(conv))
        loss_p   = self.criterion(div, torch.zeros_like(div))
        loss_PDE = self.weights[0] * loss_c + self.weights[1] * loss_p
        loss_corr = self.weights[2] * self.criterion(pxy, torch.zeros_like(pxy))

        self.PDE.append(loss_PDE.item())
        self.corr.append(loss_corr.item())
        if torch.isnan(loss_PDE)==True:
            raise ValueError("nan value reached")
        if torch.isnan(loss_corr)==True:
            raise ValueError("nan value reached")

        loss = (loss_PDE   +
                loss_BC    +
                loss_IC    +
                loss_corr  )

        loss.backward()
        self.losses.append(loss.item())

        if self.adaptive_points==True and self.epoch%self.adaptive_freq==0 and self.epoch>self.Adam_epochs: 
            self.make_distributed_points()
            print('points adapted')
        
        end = self.k * time.time()

        if self.epoch % self.k == 0:
            self.print_tab.add_rows([['|', f'{self.epoch}\t',         '|',
                                    f'{round(loss_PDE.item(),  6)}\t','|',
                                    f'{round(loss_corr.item(), 6)}\t','|',
                                    f'{round(loss_IC.item(),   6)}\t','|',
                                    f'{round(loss_BC.item(),   6)}\t','|',
                                    f'{round(self.losses[-1],  6)}\t','|',
                                    f'{round(end - start, 6)}',       '|']])
            print(self.print_tab.draw())
        self.epoch += 1
        return loss
        
    
    def train(self):
        self.print_tab = Texttable()
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,10,1,15,1,15,1,15,1,15,1,15,1,10,1])
        self.print_tab.add_rows([['|','Epochs','|', 'PDE loss','|','p corr loss','|','IC loss','|','BC loss','|','Summary loss','|','time','|']])
        print(self.print_tab.draw())
        self.model.train()

        self.optimizer = self.model.set_optimizer('NAdam')

        while self.epoch<=self.max_epoch:
            print ('Adam')
            while self.epoch <= self.Adam_epochs+1:
                self.optimizer = self.model.set_optimizer('NAdam')
                self.optimizer.step(self.loss_function)
            print ('LBFGS')
            self.optimizer = self.model.set_optimizer('LBFGS',self.adaptive_freq)
            self.optimizer.step(self.loss_function)
            self.make_distributed_points()
            self.Adam_epochs = self.epoch+100
            # self.save(f'{self.epoch}')
            
            print ('update')

    
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
        with torch.no_grad():
            pred = self.model([x,y,t], self.transform)
            ux = self.get_ux(x,y,t)
            uy = self.get_uy(x,y,t)
            conv = misc.derivative(self.w_func(x,y)*pred[:,0], t) + misc.derivative(self.w_func(x,y)*pred[:,0]*ux, x) + misc.derivative(self.w_func(x,y)*pred[:,0]*uy, y)
            return conv.data.cpu().numpy()
    
    def get_div(self, x, y, t):
        with torch.no_grad():
            ux = self.get_ux(x,y,t)
            uy = self.get_uy(x,y,t)
            div = misc.derivative(ux,x) + misc.derivative(uy,y)
            return div.data.cpu().numpy()