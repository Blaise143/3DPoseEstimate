import torch
import torch.nn as nn
import pytorch_lightning as pl


class VQVAE(pl.LightningModule):
    def __init__(self,
                 encoder,
                 decoder,
                 num_embeddings,
                 embedding_dim,
                 commitment_cost,
                 learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost)
        self.learning_rate = learning_rate
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        quantization_loss, quantized = self.quantizer(z)
        recon_x = self.decoder(quantized)
        return recon_x, quantization_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, quantization_loss = self(x)
        recon_loss = self.reconstruction_loss(recon_x, x)
        loss = recon_loss + quantization_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, quantization_loss = self(x)
        recon_loss = self.reconstruction_loss(recon_x, x)
        loss = recon_loss + quantization_loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    me = Encoder([38, 20, 15])
    print("___")
    you = Decoder([15, 20, 38])
    print(me)
    print(you)
