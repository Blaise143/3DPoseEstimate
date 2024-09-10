import torch
import torch.nn as nn
from typing import List


class LiftNet(nn.Module):
    """
    A lifting network from 2d to 3d
    """

    def __init__(self,
                 layers_list: tuple[int] = (38, 52, 256, 256, 128, 57),
                 activation=nn.ReLU(),
                 dropout=0.2):
        super().__init__()
        layers = nn.ModuleList()
        for i in range(len(layers_list)): # [0,1,2,3,4]
            if i < (len(layers_list) - 2):
                layers.append(
                    FC(
                        in_features=layers_list[i],
                        out_features=layers_list[i + 1],
                        activation=activation,
                        dropout=dropout
                    )
                )
            else:

                layers.append(
                    nn.Linear(in_features=layers_list[i],out_features=layers_list[i+1])
                )
                break

        self.layers = layers

    def forward(self, x: torch.tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class FC(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float,
                 activation: nn.Module = nn.ReLU(),
                 batch_norm: nn.Module = nn.BatchNorm1d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out_features
            ),
            batch_norm(out_features),
            activation,
            nn.Dropout(p=dropout)

        )

    def forward(self, x: torch.tensor):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    a = torch.rand(60, 38)
    net = LiftNet(dropout=0.2)
    print(net(a))
