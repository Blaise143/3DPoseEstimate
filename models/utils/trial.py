# from helpers import VariationalAutoEncoder
from abc import ABC, abstractclassmethod
import torch.nn as nn
from dataclasses import dataclass


class NeuralNetBase(ABC):

    @abstractclassmethod
    def run():
        ...

    @abstractclassmethod
    def run_again():
        ...

    def my_info(self):
        print("hELLO DUDE")


class Child(NeuralNetBase):
    def __init__(self):
        ...

    def run(self):
        pass

    def run_again(self):
        pass


class Trial:
    CONST = 11

    def __init__(self):
        self.CONST = 12


@dataclass
class DeezNuts:
    DEEZ: int
    NUTS: int
    DUDE: str = "he"


if __name__ == "__main__":
    me = DeezNuts(12, 21,)
    print(me.DEEZ+me.NUTS)
