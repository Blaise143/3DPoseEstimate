import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    """
    The decoder for the VQ-VAE
    """

    def __init__(self, layers_order: List[int], dropout: float = 0.3):
        super(Decoder, self).__init__()
        decode_layers = []
        for i in range(len(layers_order) - 2):
            decode_layers.extend(
                [
                    nn.Linear(layers_order[i], layers_order[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
            )
        decode_layers.append(
            nn.Linear(layers_order[-2], layers_order[-1]))  # Output layer
        self.layers = nn.Sequential(*decode_layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    import torch

    layers = [16, 24, 56, 232]
    net = Decoder(layers)
    random_tensor = torch.rand(layers[0])
    assert net(random_tensor).shape[0] == layers[-1]
