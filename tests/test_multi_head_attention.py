import os
import tempfile
import random
from unittest import TestCase
import torch
import keras
import numpy as np
from keras_multi_head import MultiHeadAttention as KerasMultiHeadAttention
from torch_multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention(TestCase):

    def test_weight_size(self):
        net = MultiHeadAttention(in_features=120, head_num=6)
        model_path = os.path.join(tempfile.gettempdir(), 'test_multi_head_attention_%f.pth' % random.random())
        torch.save(net, model_path)
        net = torch.load(model_path)
        print(net)
        self.assertEqual(torch.Size([120, 480]), net.weight.size())
        self.assertEqual(torch.Size([480]), net.bias.size())

    def test_no_bias(self):
        net = MultiHeadAttention(in_features=120, head_num=6, bias=False)
        self.assertEqual(torch.Size([120, 480]), net.weight.size())
        self.assertEqual(None, net.bias)

    def test_divisible(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(in_features=73, head_num=5)

    @staticmethod
    def get_torch_layer_with_weights(feature_dim, head_num, weights, bias):
        layer = MultiHeadAttention(feature_dim, head_num)
        layer.weight = torch.nn.Parameter(torch.from_numpy(weights))
        layer.bias = torch.nn.Parameter(torch.from_numpy(bias))
        return layer

    @staticmethod
    def get_keras_layer_weight_weights(seq_len, feature_dim, head_num, weights, bias, history_only=False):
        input_layer = keras.layers.Input(shape=(seq_len, feature_dim), name='Input')
        attention_layer = KerasMultiHeadAttention(
            head_num,
            history_only=history_only,
            weights=[
                weights[:, :feature_dim],
                bias[:feature_dim],
                weights[:, feature_dim:feature_dim * 2],
                bias[feature_dim:feature_dim * 2],
                weights[:, feature_dim * 2:feature_dim * 3],
                bias[feature_dim * 2:feature_dim * 3],
                weights[:, -feature_dim:],
                bias[-feature_dim:],
            ],
        )(input_layer)
        model = keras.models.Model(input_layer, attention_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_same_output_without_mask(self):
        batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
        weights = np.random.standard_normal((feature_dim, feature_dim * 4))
        bias = np.random.standard_normal((feature_dim * 4,))
        torch_net = self.get_torch_layer_with_weights(feature_dim, head_num, weights, bias)
        keras_net = self.get_keras_layer_weight_weights(seq_len, feature_dim, head_num, weights, bias)
        for _ in range(100):
            x = np.random.standard_normal((batch_size, seq_len, feature_dim))
            y = keras_net.predict(x)
            x = torch.from_numpy(x)
            y_hat = torch_net(x, x, x)
            self.assertTrue(np.allclose(y, y_hat.detach().numpy(), rtol=0.0, atol=1e-4), (y[0, 0], y_hat[0, 0]))

    def test_same_output_history_only(self):
        batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
        weights = np.random.standard_normal((feature_dim, feature_dim * 4))
        bias = np.random.standard_normal((feature_dim * 4,))
        torch_net = self.get_torch_layer_with_weights(feature_dim, head_num, weights, bias)
        keras_net = self.get_keras_layer_weight_weights(seq_len, feature_dim, head_num, weights, bias, True)
        for _ in range(100):
            x = np.random.standard_normal((batch_size, seq_len, feature_dim))
            y = keras_net.predict(x)
            x = torch.from_numpy(x)
            y_hat = torch_net(x, x, x, MultiHeadAttention.gen_history_mask(x))
            self.assertTrue(np.allclose(y, y_hat.detach().numpy(), rtol=0.0, atol=1e-4), (y[0, 0], y_hat[0, 0]))
