import numpy as np 
import torch
import torch.nn as nn

import programs.NN as NN

torch.manual_seed(1234)


class Sin(nn.Module):
    """
    sin activation function for Neural Network
    """
    def __init__(self, f):
        super().__init__()
        self.name = 'Sin'
        self.f = f

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return torch.sin(self.f * input)
    

class Cos(nn.Module):
    """
    cos activation function for Neural Network
    """
    def __init__(self, f):
        super().__init__()
        self.name = 'Cos'
        self.f = f

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return torch.cos(self.f * input)
        

class Wave(nn.Module):
    """
    sin+cos activation function for Neural Network
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.cos(input) + torch.sin(input)


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


class Width():
    def __init__(self,
                 func : str,
                 w    : float,
                 w1   : float = 1,
                 w2   : float = 1,
                 w3   : float = 1,
                 w4   : float = 1
                ):
        self.name = func
        self.good_names = ('w_const',    'wconst',    'const',
                           'w_sincos',   'wsincos',   'sincos',
                           'w_elliptic', 'welliptic', 'elliptic',
                           'w_wave',     'wwave',     'wave'
                          )
        self.const_name    = ('w_const',    'wconst',    'const')
        self.sincos_name   = ('w_sincos',   'wsincos',   'sincos')
        self.elliptic_name = ('w_elliptic', 'welliptic', 'elliptic')
        self.wave_name     = ('w_wave',     'wwave',     'wave')
        self.sineband_name = ('w_sineband', 'wsineband', 'sineband')

        self.good_names = self.const_name + self.sincos_name + self.elliptic_name + self.wave_name + self.sineband_name
        
        self.w  = w
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        if not(func in self.good_names):
            raise NameError('Function not found')

    
    def __call__(self, x=None, y=None):
        if self.name in self.const_name:
            return self.const()
        elif self.name in self.sincos_name:
            return self.sincos(x, y)
        elif self.name in self.elliptic_name:
            return self.elliptic(x, y)
        elif self.name in self.wave_name:
            return self.wave(x, y)
        elif self.name in self.sineband_name:
            return self.sineband(x, y)


    def const(self):          return self.w

    def sincos(self, x, y):   return self.w + self.w1 * torch.cos(torch.pi*x) * torch.sin(torch.pi*y)
    
    def elliptic(self, x, y): return self.w + self.w1 * ((x / self.w2)**2 + ((y - self.w4) / self.w3)**2)

    def wave(self, x, y):     return self.w + self.w1 * torch.sin(torch.pi*(self.w2*x-self.w3*y))

    def sineband(self, x, y): return self.w + self.w1 * (1 + torch.cos(torch.pi*self.w2*y)) / 2 * (torch.sigmoid(200*(x-self.w3)) - torch.sigmoid(200*(x-self.w4)))