import unittest

import torch
from utils import get_torch_device

from loss import NeuralNDCGLoss


class TestNeuralNDCG(unittest.TestCase):

    def setUp(self):
        # テスト用のデバイスを取得
        self.device = get_torch_device()

        # テスト用のデータセットを指定されたデバイスに移動
        self.y_pred = torch.tensor(
            [[0.2, 0.4, 0.3, 0.8]],
            dtype=torch.float32,
            requires_grad=True,
        ).to(self.device)
        self.y_true = torch.tensor([[0, 1, 2, 3]], dtype=torch.float32).to(self.device)
        self.padded_value_indicator = -1
        self.loss = NeuralNDCGLoss()

    def test_neuralNDCG_deterministic(self):
        # Deterministicバージョンのテスト
        self.loss.stocastic = False
        loss = self.loss(self.y_pred, self.y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        print(f"Deterministic neuralNDCG loss: {loss.item()}")
        loss.backward()

    def test_neuralNDCG_stochastic(self):
        # Stochasticバージョンのテスト
        self.loss.stocastic = True
        self.loss.n_samples = 10
        loss = self.loss(self.y_pred, self.y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        print(f"Stochastic neuralNDCG loss: {loss.item()}")
        loss.backward()

    def test_neuralNDCG_transposed_deterministic(self):
        # Transposed Deterministicバージョンのテスト
        self.loss.stocastic = False
        self.loss.transposed = True

        loss = self.loss(self.y_pred, self.y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        print(f"Transposed deterministic neuralNDCG loss: {loss.item()}")
        loss.backward()

    def test_neuralNDCG_transposed_stochastic(self):
        # Transposed Stochasticバージョンのテスト
        self.loss.stocastic = True
        self.loss.transposed = True
        self.loss.n_samples = 10
        loss = self.loss(self.y_pred, self.y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        print(f"Transposed stochastic neuralNDCG loss: {loss.item()}")
        loss.backward()


if __name__ == "__main__":
    unittest.main()
