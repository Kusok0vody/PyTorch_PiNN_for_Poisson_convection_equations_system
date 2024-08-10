import numpy as np 
import torch
import torch.nn as nn

from torch.autograd import grad
from collections import OrderedDict


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
            return torch.optim.Adam(self.parameters(), weight_decay=1e-5)
        if optimizer_type=='NAdam':
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