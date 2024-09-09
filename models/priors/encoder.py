import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """
    The encoder for the VQ-VAE
    """

    def __init__(self, layers_order: List[int], dropout: float = 0.3):
        super(Encoder, self).__init__()
        encode_layers = []
        for i in range(len(layers_order) - 2):
            encode_layers.extend(
                [
                    nn.Linear(layers_order[i], layers_order[i + 1]),
                    # nn.BatchNorm1d(num_features=layers_order[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
            )
        encode_layers.append(nn.Linear(layers_order[-2], layers_order[-1]))
        self.layers = nn.Sequential(*encode_layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    m = Encoder(
        [38, 25, 23, 15]
    )
    print(m)
