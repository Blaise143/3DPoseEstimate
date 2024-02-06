import torch.nn as nn


class Encoder(nn.Module):
    """
    The encoder for the VQ-VAE
    """

    def __init__(self, layers_order, dropout=0.3):
        super(Encoder, self).__init__()
        encode_layers = []
        for i in range(len(layers_order) - 1):
            encode_layers.extend([
                nn.Linear(layers_order[i], layers_order[i + 1]),
                nn.BatchNorm1d(num_features=layers_order[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.layers = nn.Sequential(*encode_layers)

    def forward(self, x):
        return self.layers(x)
