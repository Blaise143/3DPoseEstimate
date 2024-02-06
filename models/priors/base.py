import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Prior(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass
