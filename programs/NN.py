import torch
import torch.nn as nn
import programs.objects as obj

from collections import OrderedDict

torch.manual_seed(1234)


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
        act
        ):

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
        self.sin = act

    def forward(self, inputs: list, transform_func=None):
        """
        Forward pass through the network.
        Concatenates the input tensors and feeds them through the layers.
        """
        inputs_united = torch.cat([input_tensor.reshape(-1, 1) for input_tensor in inputs], axis=1)
        outputs = self.layers(inputs_united)
        if transform_func!=None:
            transform_func(outputs, inputs)
        return outputs
    
    def set_optimizer(self, name):
        self.optimizer = obj.Optimizer(name, self.parameters)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)