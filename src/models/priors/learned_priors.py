import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import wandb

BATCH_SIZE = 500


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, layers_order: list, latent_dim: int = 10, dataset=None, dropout=0.3, learning_rate=3e-4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.dataset = dataset

        # Encoder Layers
        encode_layers = []
        for i in range(len(layers_order) - 1):
            encode_layers.extend(
                [
                    nn.Linear(layers_order[i], layers_order[i + 1]),
                    nn.BatchNorm1d(num_features=layers_order[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
            )
        self.encoder = nn.Sequential(*encode_layers)

        # Latent Space Layers for mean and variance
        self.mu = nn.Linear(layers_order[-1], latent_dim)
        # Use log variance for stability
        self.log_var = nn.Linear(layers_order[-1], latent_dim)

        # Decoder Layers
        layers_order.reverse()
        # Start with mapping from latent to first decode layer
        decode_layers = [nn.Linear(latent_dim, layers_order[0])]
        for i in range(len(layers_order) - 1):
            decode_layers.append(
                nn.Linear(layers_order[i], layers_order[i + 1]))
            if i < len(layers_order) - 2:
                decode_layers.append(nn.BatchNorm1d(layers_order[i+1]))
                decode_layers.append(nn.ReLU())
                decode_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decode_layers)

        self.reconstruction_loss = nn.MSELoss(reduction="sum")

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)  # log variance for numerical stability
        return mu, log_var

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        # Random noise from the normal distribution
        eps = torch.randn_like(std)
        z = mu + std * eps  # The reparametrization trick
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, x_v = batch
        reconstructed, mu, log_var = self.forward(x)
        # Calculating reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        # calculating kl divergence loss
        kl_divergence_loss = self.kl_divergence(mu, log_var)

        reconstructed_v, mu_v, log_var_v = self.forward(x_v)
        recon_loss_v = self.reconstruction_loss(reconstructed_v, x_v)
        kl_divergence_loss_v = self.kl_divergence(mu_v, log_var_v)

        loss = recon_loss+kl_divergence_loss + recon_loss_v + kl_divergence_loss_v
        self.log("train_loss", loss)
        self.log("train_reconstruction_loss", recon_loss+recon_loss_v)
        self.log("train_kl_divergence_loss",
                 kl_divergence_loss+kl_divergence_loss_v)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, x_v = batch
        reconstructed, mu, logvar = self.forward(x)
        reconstructed_v, mu_v, logvar_v = self.forward(x_v)
        # loss = self.loss_function(
        # x=x, x_reconstructed=reconstructed, mu=mu, log_var=logvar)
        recon_loss = self.reconstruction_loss(reconstructed, x)
        kl_divergence_loss = self.kl_divergence(mu, logvar)

        recon_loss_v = self.reconstruction_loss(reconstructed_v, x_v)
        kl_divergence_loss_v = self.kl_divergence(mu_v, logvar_v)

        loss = recon_loss+kl_divergence_loss + recon_loss_v+kl_divergence_loss_v

        self.log("val_loss", loss)
        self.log("val_reconstruction_loss", recon_loss + recon_loss_v)
        self.log("val_kl_divergence_loss",
                 kl_divergence_loss + kl_divergence_loss_v)

        return loss

    def kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor):
        var = log_var.exp()
        mu_squared = mu.pow(2)
        kl_div_elementwise = 1 + log_var - mu_squared - var
        total_kl_div = -0.5 * torch.sum(kl_div_elementwise)
        return total_kl_div

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.8*dataset_size)
        val_size = dataset_size-train_size

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


if __name__ == "__main__":
    a = VariationalAutoEncoder(
        layers_order=[38, 30, 25, 20, 15], latent_dim=10)
    print(a)
    rand_tensor = torch.rand(38)
    b = a(rand_tensor)

    print(b[0].shape, b[1].shape, b[2].shape)
