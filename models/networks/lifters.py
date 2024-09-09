import torch
import torch.nn as nn
from typing import List


class LiftNet(nn.Module):
    """
    A lifting network from 2d to 3d
    """

    def __init__(self,
                 layers_list: List[int] = [52, 60, 65, 70, 78],
                 activation = nn.ReLU(),
                 dropout = 0.2):
        super().__init__()
        layers = nn.ModuleList()
        for i in range(len(layers_list)):
            if i != (len(layers_list)-1):
                layers.append(nn.Linear(layers_list[i], layers_list[i+1]))
                layers.append(nn.BatchNorm1d(layers_list[i+1]))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
            else:
                layers.extend(
                    [
                        nn.Linear(layers_list[i], layers_list[i+1])
                    ]
                )
        self.layers = layers

        def forward(self, x: torch.tensor):
            for layer in self.layers:
                x = layer(x)
            return x



class NormalEstimator(nn.Module):
    """
    Estimates the normal. 2D Pose -> Vector of size 3
    """
    def __init__(self):
        super().__init__()
        ...

    # TODO: Implement this



