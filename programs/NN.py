import numpy as np 
import torch
import torch.nn as nn

from torch.autograd import grad
from collections import OrderedDict

torch.manual_seed(1234)

class Sin(nn.Module):
    """
    sin activation function for Neural Network
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)
    

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


class Net(nn.Module):
    """
    The main Class for all Neural Networks
    """
    def __init__(
        self,
        input_size:int,
        neurons_arr:list,
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
        self.sin = Sin()

    def forward(self, inputs: list, transform_func=None):
        """
        Forward pass through the network.
        Concatenates the input tensors and feeds them through the layers.
        """
        inputs_united = torch.cat([input_tensor.reshape(-1, 1) for input_tensor in inputs], axis=1)
        outputs = self.layers(inputs_united)
        if transform_func!=None:
            transform_func(outputs)
        return outputs
        
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


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
                                     # lr=0.1, 
                                     max_iter=100000, 
                                     max_eval=100000, 
                                     history_size=200,
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