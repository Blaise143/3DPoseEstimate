import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.priors import VectorQuantizer, Encoder, Decoder
from typing import List
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 500


class VQVAE(pl.LightningModule):

    def __init__(self,
                 encoder_layers: List[int],
                 decoder_layers: List[int],
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float,
                 learning_rate: float = 1e-4,
                 encoder_dropout: float = 0.3,
                 decoder_dropout: float = 0.3,
                 denoise: bool = False,
                 dataset=None):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(encoder_layers, dropout=encoder_dropout)
        self.decoder = Decoder(decoder_layers, dropout=decoder_dropout)
        self.quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, ortho_loss_weight=0)
        self.learning_rate = learning_rate
        self.reconstruction_loss = nn.MSELoss()
        self.dataset = dataset
        self.denoise = denoise
        if self.denoise:
            print("this is denoising")

    def forward(self, x):
        if self.denoise:
            noisy_x = self.add_noise(x)
            z = self.encoder(noisy_x)
        else:
            z = self.encoder(x)
        quantized, quantization_loss, _, _ = self.quantizer(z)
        recon_x = self.decoder(quantized)
        return recon_x, quantization_loss

    @staticmethod
    def add_noise(x, noise_factor=0.1):
        noise = torch.randn_like(x) * noise_factor
        x = x + noise
        return x

    def _common_step(self, batch, batch_idx):
        (x, mask), (x_v, mask_v) = batch
        recon_x, quantization_loss = self(x)
        recon_x_v, quantization_loss_v = self(x_v)

        mask = mask.repeat(1, 1, 2).reshape(x.shape)
        mask_v = mask_v.repeat(1, 1, 2).reshape(x.shape)
        # print(mask)
        # print(mask_v)

        recon_loss = self.reconstruction_loss(recon_x * mask, x * mask)
        recon_loss_v = self.reconstruction_loss(recon_x_v * mask_v, x_v * mask_v)

        total_recon_loss = recon_loss + recon_loss_v
        total_quant_loss = quantization_loss + quantization_loss_v

        loss = (0.1 * total_quant_loss) + total_recon_loss

        return total_recon_loss, total_quant_loss, loss

    def training_step(self, batch, batch_idx):
        recon_loss, quantization_loss, loss = self._common_step(
            batch, batch_idx)
        self.log("train_loss", loss),
        self.log("train_recon_loss", recon_loss),
        self.log("train_quant_loss", quantization_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        recon_loss, quantization_loss, loss = self._common_step(
            batch, batch_idx)
        self.log_dict({"val_loss": loss,
                       "val_recon_loss": recon_loss,
                       "val_quant_loss": quantization_loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        return optimizer

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size])

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return val_loader
