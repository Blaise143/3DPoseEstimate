import torch
import torch.nn as nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float) -> None:
        """
        Args:
            num_embeddings (int): Number of embeddings in the quantized space.
            embedding_dim (int): Dimensionality of the embeddings.
            commitment_cost (float): Coefficient for the commitment loss.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # The embedding table (aka codebook) got weights from a uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        """
        Forward pass of the vector quantizer.

        Args:
            inputs (Tensor): A batch of input vectors with shape (batch_size, embedding_dim).

        Returns:
            quantized: The quantized version of the input.
            loss: The VQ-VAE loss.
            quantized_idx: Indices of the quantized vectors in the embedding table.
            inputs: The original input vectors (for convenience).
        """
        # Calculate distances between input vectors and embedding vectors
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.weight.t()))

        # Find the closest embeddings
        closest_embeddings_idx = torch.argmin(distances, dim=1)
        quantized = self.embedding(closest_embeddings_idx).view(inputs.shape)

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return quantized, loss, closest_embeddings_idx, inputs


if __name__ == "__main__":
    quantizer = VectorQuantizer(
        num_embeddings=100,
        embedding_dim=10,
        commitment_cost=0.25
    )
    x = torch.randn(32, 10)
    out = quantizer(x).shape
    print(out)
