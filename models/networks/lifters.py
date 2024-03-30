import torch
import torch.nn as nn
from typing import List


class LiftNet(nn.Module):
    """
    A lifting network from 2d to 3d
    """

    def __init__(self, layers: List[int]):
        super().__init__()
        layers = nn.ModuleList()
        for i in layers:
            ...


class NormalEstimator(nn.Module):
    """
    Estimates the normal. 2D Pose -> Vector of size 3
    """
    def __init__(self):
        super().__init__()
        ...

    # TODO: Implement this



