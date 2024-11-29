import unittest

import torch
import numpy as np
from neuralndcgloss import NeuralNDCGLoss
from neuralndcgloss.utils import get_torch_device


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
        self.assertAlmostEqual(loss.item(), -0.84705, places=5)
        loss.backward()

    def test_neuralNDCG_stochastic(self):
        # Stochasticバージョンのテスト
        self.loss.stocastic = True
        self.loss.n_samples = 10
        loss = self.loss(self.y_pred, self.y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        print(f"Stochastic neuralNDCG loss: {loss.item()}")
        self.assertAlmostEqual(loss.item(), -0.84705, places=5)
        loss.backward()

    def test_neuralNDCG_transposed_deterministic(self):
        # Transposed Deterministicバージョンのテスト
        self.loss.stocastic = False
        self.loss.transposed = True

        loss = self.loss(self.y_pred, self.y_true)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        print(f"Transposed deterministic neuralNDCG loss: {loss.item()}")
        self.assertAlmostEqual(loss.item(), -0.84705, places=5)
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
        self.assertAlmostEqual(loss.item(), -0.84705, places=5)
        loss.backward()


class TestNeuralNDCGTemperature(unittest.TestCase):

    def setUp(self):
        # デバイスを取得
        self.device = get_torch_device()

        # y_true と x_tests を設定
        self.y_true = [1, 2, 3, 4, 5]
        x = np.arange(0, 5.1, 0.1)
        self.x_tests = [[1, 2, 3, 4, float(i)] for i in x]

        # 期待される結果
        self.expected_losses = [
            [
                0.7304436564445496, 0.7304459810256958, 0.7304520010948181, 0.7304686307907104, 0.7305136322975159,
                0.7306350469589233, 0.7309604287147522, 0.7318086624145508, 0.7338765263557434, 0.738190770149231,
                0.7448548078536987, 0.7515106797218323, 0.7558330297470093, 0.7579248547554016, 0.7588378190994263,
                0.7593415975570679, 0.7599439024925232, 0.7612420320510864, 0.7643117904663086, 0.7706870436668396,
                0.7805296778678894, 0.7903529405593872, 0.7967343926429749, 0.7998256683349609, 0.8011835813522339,
                0.8019534349441528, 0.8029108047485352, 0.8050052523612976, 0.809971809387207, 0.8202906847000122,
                0.8362164497375488, 0.8521291613578796, 0.8624646067619324, 0.8674797415733337, 0.8697069883346558,
                0.8710300326347351, 0.8727825284004211, 0.8767004013061523, 0.886026918888092, 0.9054151773452759,
                0.9353219866752625, 0.9652078151702881, 0.9845781922340393, 0.9938634037971497, 0.9976717233657837,
                0.9991326332092285, 0.9996786713600159, 0.9998807311058044, 0.9999551773071289, 0.9999825954437256,
                0.9999927282333374,
            ],
            [
                0.7248393297195435, 0.7254757285118103, 0.7262188196182251, 0.7270830869674683, 0.7280853390693665,
                0.7292471528053284, 0.7305970191955566, 0.7321717143058777, 0.7340125441551208, 0.7361516356468201,
                0.7385908365249634, 0.7412964105606079, 0.7442038655281067, 0.7472615242004395, 0.7504481673240662,
                0.7537791728973389, 0.7573021650314331, 0.7610902190208435, 0.7652236819267273, 0.7697499990463257,
                0.7746357321739197, 0.7797633409500122, 0.7849951386451721, 0.790265679359436, 0.7956052422523499,
                0.8011196851730347, 0.806965708732605, 0.8133363723754883, 0.820426344871521, 0.8283524513244629,
                0.8370505571365356, 0.8462504744529724, 0.8556345105171204, 0.8649888038635254, 0.8742533326148987,
                0.8834628462791443, 0.8926714658737183, 0.9019033312797546, 0.91112220287323, 0.9202179312705994,
                0.9290149211883545, 0.9372393488883972, 0.9447989463806152, 0.9516355991363525, 0.9577600955963135,
                0.9632158875465393, 0.9680500626564026, 0.972307026386261, 0.9760289192199707, 0.9792581796646118,
                0.9820382595062256,
            ],
        ]
        self.temperatures = [0.1, 0.5]

    def test_neuralNDCG_temperature(self):
        for temp_idx, temp in enumerate(self.temperatures):
            lossfn = NeuralNDCGLoss(log_scores=False, stochastic=False, temperature=temp)
            expected_losses = self.expected_losses[temp_idx]

            for i, test in enumerate(self.x_tests):
                y_pred = torch.tensor([test], dtype=torch.float32, requires_grad=True).to(self.device)
                y_true = torch.tensor([self.y_true], dtype=torch.float32).to(self.device)

                loss = -1 * lossfn(y_pred, y_true).item()
                self.assertAlmostEqual(loss, expected_losses[i], places=6)


if __name__ == "__main__":
    unittest.main()
