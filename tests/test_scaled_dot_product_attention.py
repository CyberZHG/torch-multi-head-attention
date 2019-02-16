from unittest import TestCase
import torch
from torch_multi_head_attention import ScaledDotProductAttention, MultiHeadAttention


class TestScaledDotProductAttention(TestCase):

    def test_sample(self):
        x = torch.Tensor([[
            [0.2, 0.3, 0.4, 0.6, 0.5],
            [0.4, 0.7, 0.2, 0.6, 0.9],
            [0.3, 0.5, 0.8, 0.9, 0.1],
            [0.2, 0.3, 0.4, 0.6, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ]])
        mask = torch.Tensor([[1, 1, 1, 1, 0]])
        y = ScaledDotProductAttention()(x, x, x, mask)[0]
        self.assertTrue(y[0].allclose(y[3]), y)
        self.assertTrue(y[2].allclose(torch.Tensor([0.27883747, 0.45767492, 0.47448885, 0.69199574, 0.47368336])), y[2])

    def test_history_only(self):
        x = torch.Tensor([[
            [0.2, 0.3, 0.4, 0.6, 0.5],
            [0.4, 0.7, 0.2, 0.6, 0.9],
            [0.3, 0.5, 0.8, 0.9, 0.1],
            [0.2, 0.3, 0.4, 0.6, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ]])
        mask = MultiHeadAttention.gen_history_mask(x)
        y = ScaledDotProductAttention()(x, x, x, mask)[0]
        self.assertFalse(y[0].allclose(y[3]), y)
        self.assertTrue(y[0].allclose(x[0, 0]), y[0])
