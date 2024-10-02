import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union


class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float,
                 ortho_loss_weight: float = 0.09) -> None:
        """
        Args:
            num_embeddings (int): Number of embeddings in the quantized space.
            embedding_dim (int): Dimensionality of the embeddings.
            commitment_cost (float): Coefficient for the commitment loss.
            ortho_loss_weight (float): the weight of the orthogonal regularization loss. 
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # The embedding table (aka codebook) got weights from a uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            num_embeddings, 1.0 / num_embeddings)
        # self.embedding.weight.data.uniform_(-1, 1)
        self.ortho_loss_weight = ortho_loss_weight

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
        # print(f"inputs shape: {inputs.shape}")
        input_shape = inputs.shape
        # inputs = inputs.unsqueeze(2) # converted to shape (batch_size, len, embedding_dim)
        # print(f"new shape: {inputs.shape}")
        # exit()
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        # Finding the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        # print(f"encodings: {encodings}, shape: {encodings.shape}")
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(
            encodings, self.embedding.weight).view(input_shape)

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        ortho_loss = self.compute_orthogonality_loss()

        loss = q_latent_loss + self.commitment_cost * \
               e_latent_loss + self.ortho_loss_weight * ortho_loss

        quantized = inputs + (quantized - inputs).detach()
        # print(f"closest indices: {encoding_indices}")

        return quantized, loss, encoding_indices, inputs

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Computes the orthogonatity loss. 
        Calculates the dot product between pairs of codebook vectors
        """
        # Compute orthogonality loss
        # Calculate dot products between pairs of codebook vectors
        dot_products = torch.matmul(
            self.embedding.weight, self.embedding.weight.t())
        # Exclude diagonal elements to avoid penalizing similarity to itself
        ortho_loss = torch.norm(
            dot_products - torch.diag(dot_products.diag()), p='fro')

        return ortho_loss


class EMAQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float,
                 decay: float,
                 eps: float = 1e-5,
                 reset_codebook: bool = False,
                 reset_threshold: float = None,
                 ortho_loss_weight: float = 0.09) -> None:
        """
        Exponential Moving Average Quantizer, also performs some code resetting
        Args:
            num_embeddings (int): Number of embeddings in the quantized space.
            embedding_dim (int): Dimensionality of the embeddings.
            commitment_cost (float): Coefficient for the commitment loss.
            decay (float): EMA decay rate
            eps (float): constant for numerical stability
            ortho_loss_weight (float): the weight of the orthogonal regularization loss.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.reset_codebook = reset_codebook
        if reset_threshold:
            self.reset_threshold = reset_threshold

        # The embedding table (aka codebook) got weights from a uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # print(f"embedding shape: {self.embedding.weight}")
        self.embedding.weight.data.uniform_(-1.0 /
                                            num_embeddings, 1.0 / num_embeddings)
        # self.embedding.weight.data.uniform_(-1, 1)
        self.ortho_loss_weight = ortho_loss_weight
        self.decay = decay
        self.eps = eps

        # EMA Variables
        self.ema_cluster_size = nn.Parameter(torch.zeros(num_embeddings), requires_grad=False)
        self.ema_embedding_avg = nn.Parameter(torch.zeros(num_embeddings, embedding_dim), requires_grad=False)

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
        # print(f"inputs shape: {inputs.shape}")
        input_shape = inputs.shape
        # inputs = inputs.unsqueeze(2) # converted to shape (batch_size, len, embedding_dim)
        # print(f"new shape: {inputs.shape}")
        # exit()
        # print(f"Input_shape: {inputs.shape}")
        flat_input = inputs.view(-1, self.embedding_dim)
        # print(f"flat_input_shape: {flat_input.shape}")
        distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        # print(distances.shape, " distances shape")
        # Finding the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        print("closest indices: ", encoding_indices)
        # print(encoding_indices)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        # print(f"encodings: {encodings}, shape: {encodings.shape}")
        encodings.scatter_(1, encoding_indices, 1)
        # print(f'scattered encodings: {encodings}')

        quantized = torch.matmul(
            encodings, self.embedding.weight).view(input_shape)
        # print(f"quantized shape: {quantized.shape}")
        # print("encodings",encodings.sum(0).shape)
        # print(self.ema_cluster_size)

        # EMA Update
        if self.training:
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)

            embedding_sum = torch.matmul(encodings.t(), flat_input)
            self.ema_embedding_avg.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)

            # laplace smoothing
            n = torch.sum(self.ema_cluster_size) + self.eps
            cluster_size = ((self.ema_cluster_size + self.eps) / n)
            normalized_embedding_avg = self.ema_embedding_avg / cluster_size.unsqueeze(1)
            # print(f"Normalising weights")

            if self.reset_codebook and (self.ema_cluster_size < self.reset_threshold).any():
                # print("resetting codebook")
                self.ema_cluster_size.data.fill_(0)
                self.ema_embedding_avg.data.zero_()
                self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
            else:
                self.embedding.weight.data.copy_(normalized_embedding_avg)

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        ortho_loss = self.compute_orthogonality_loss()

        loss = q_latent_loss + self.commitment_cost * \
               e_latent_loss + self.ortho_loss_weight * ortho_loss

        quantized = inputs + (quantized - inputs).detach()
        # print(f"closest indices: {encoding_indices}")

        return quantized, loss, encoding_indices, inputs

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Computes the orthogonatity loss.
        Calculates the dot product between pairs of codebook vectors
        """
        # Compute orthogonality loss
        # Calculate dot products between pairs of codebook vectors
        dot_products = torch.matmul(
            self.embedding.weight, self.embedding.weight.t())
        # Exclude diagonal elements to avoid penalizing similarity to itself
        ortho_loss = torch.norm(
            dot_products - torch.diag(dot_products.diag()), p='fro')

        return ortho_loss


if __name__ == "__main__":

    quantizer = VectorQuantizer(
        num_embeddings=100,
        embedding_dim=10,
        commitment_cost=0.25
    )

    ema_quantizer = EMAQuantizer(
        num_embeddings=100,
        embedding_dim=10,
        commitment_cost=0.25,
        decay=0.9,
        reset_codebook=True,
        reset_threshold=0.1
    )

    # print(ema_quantizer.training)
    x = torch.randn(32, 10)
    # out = quantizer(x)
    out_2 = ema_quantizer(x)
    # print(out_2[0].shape)
