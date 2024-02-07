import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 /
                                            self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # distances between input embeddings and codeblock embeddings
        distances = (torch.sum(inputs**2, dim=2, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.weight.t()))

        # Finding the closest codebook embeddings
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(2)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantizing input
        quantized = torch.matmul(
            min_encodings, self.embedding.weight).transpose(1, 2)

        # Loss for maintaining codebook commitment
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized
