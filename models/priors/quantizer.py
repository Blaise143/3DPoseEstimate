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
        # self.embedding.weight.data.uniform_(-1, 1)

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
        # print(f"input shape: {input_shape}")
        # print(f"flat input: {flat_input.shape}")
        # exit()
        # distances between input vectors and embedding vectors
        # distances = (torch.sum(inputs**2, dim=1, keepdim=True)
        #  + torch.sum(self.embedding.weight**2, dim=1)
        #  - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        # print(f"distance shape: {distances.shape}")

        # Finding the closest embeddings
        # closest_embeddings_idx = torch.argmin(distances, dim=1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        # print(f"encodings: {encodings}, shape: {encodings.shape}")
        encodings.scatter_(1, encoding_indices, 1)

        # print(f"encodings after: {encodings}")
        # print(f"encodings shape: {encodings.shape}")
        # self.log("closest_embeding_idx", closest_embeddings_idx)
        # exit()
        # print(f"closest embedding: \n{closest_embeddings_idx}")
        # quantized = self.embedding(closest_embeddings_idx).view(inputs.shape)
        quantized = torch.matmul(
            encodings, self.embedding.weight).view(input_shape)
        # print(f"quantized: {quantized}")
        # exit()

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized-inputs).detach()
        # print(f"closest indices: {encoding_indices}")

        return quantized, loss, encoding_indices, inputs


if __name__ == "__main__":
    quantizer = VectorQuantizer(
        num_embeddings=100,
        embedding_dim=10,
        commitment_cost=0.25
    )
    x = torch.randn(32, 10)
    out = quantizer(x)
    print(out)
