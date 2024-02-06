import torch.nn as nn


class Decoder(nn.Module):
    """
    The decoder for the VQ-VAE
    """

    def __init__(self, layers_order, dropout=0.3):
        super(Decoder, self).__init__()
        decode_layers = []
        for i in range(len(layers_order) - 2):
            decode_layers.extend([
                nn.Linear(layers_order[i], layers_order[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        decode_layers.append(
            nn.Linear(layers_order[-2], layers_order[-1]))  # Output layer
        self.layers = nn.Sequential(*decode_layers)

    def forward(self, x):
        return self.layers(x)
