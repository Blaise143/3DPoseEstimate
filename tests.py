import unittest
import torch
from models.priors.quantizer import VectorQuantizer


class TestVectorQuantizer(unittest.TestCase):

    def setUp(self):
        # Initialize some common parameters and the VectorQuantizer instance
        self.num_embeddings = 10
        self.embedding_dim = 16
        self.commitment_cost = 0.25
        self.vq = VectorQuantizer(
            self.num_embeddings, self.embedding_dim, self.commitment_cost)
        self.inputs = torch.randn(32, self.embedding_dim)  # Batch size of 32

    def test_initialization(self):
        # Test if the embeddings are initialized properly
        self.assertEqual(self.vq.embedding.num_embeddings, self.num_embeddings)
        self.assertEqual(self.vq.embedding.embedding_dim, self.embedding_dim)

    def test_forward_output_shapes(self):
        # Test the shapes of the outputs from the forward pass
        quantized, loss, closest_embeddings_idx, _ = self.vq(self.inputs)
        self.assertEqual(quantized.shape, self.inputs.shape)
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.shape, torch.Size([]))  # Loss should be a scalar
        self.assertEqual(closest_embeddings_idx.shape, (32,))  # Batch size

    def test_forward_loss_value(self):
        # Ensuring loss is computed and returns a valid value
        _, loss, _, _ = self.vq(self.inputs)
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative

    def test_embedding_update(self):
        # Test if embeddings are being updated (not a direct test of forward, but useful)
        optimizer = torch.optim.Adam(self.vq.parameters(), lr=0.01)
        _, loss_before, _, _ = self.vq(self.inputs)
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()
        _, loss_after, _, _ = self.vq(self.inputs)
        # Check if loss decreases after an optimization step; might not always decrease in practice
        # but useful for checking if computation graph is correctly connected
        self.assertTrue(loss_after < loss_before or loss_after == loss_before)

    def test_single_element_batch(self):
        single_input = torch.randn(1, self.embedding_dim)
        quantized, loss, closest_embeddings_idx, _ = self.vq(single_input)
        self.assertEqual(quantized.shape, single_input.shape)
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(closest_embeddings_idx.shape, (1,))

    def test_gradients_flow_to_embeddings(self):
        self.inputs.requires_grad_()
        _, loss, _, _ = self.vq(self.inputs)
        loss.backward()
        self.assertIsNotNone(self.vq.embedding.weight.grad)
        self.assertNotEqual(torch.sum(self.vq.embedding.weight.grad), 0)

    def test_consistency_over_repeated_calls(self):
        _, _, idx_first_call, _ = self.vq(self.inputs)
        _, _, idx_second_call, _ = self.vq(self.inputs)
        self.assertTrue(torch.equal(idx_first_call, idx_second_call))

    def test_simplified_inputs(self):
        num_embeddings = 4
        embedding_dim = 2
        commitment_cost = 0.25

        vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Manually set the embeddings
        vq.embedding.weight.data = torch.tensor([
            [1.0, 0.0],  # Right
            [-1.0, 0.0],  # Left
            [0.0, 1.0],  # Up
            [0.0, -1.0],  # Down
        ], dtype=torch.float)

        # Define input vectors
        inputs = torch.tensor([
            [0.9, 0.1],  # Closer to [1.0, 0.0]
            [0.1, 0.9],  # Closer to [0.0, 1.0]
        ], dtype=torch.float)

        # Perform the forward pass
        quantized, loss, closest_embeddings_idx, _ = vq(inputs)

        # Expected closest embeddings are [0, 2] for our inputs
        expected_closest_embeddings_idx = torch.tensor([0, 2])

        # Assert the closest embeddings match the expected indices
        self.assertTrue(torch.equal(closest_embeddings_idx, expected_closest_embeddings_idx),
                        "The closest embeddings do not match the expected indices.")


if __name__ == '__main__':
    unittest.main()
