import torch
import torch.nn as nn

from .utils import (
    DEFAULT_EPS,
    PADDED_Y_VALUE,
    dcg,
    deterministic_neural_sort,
    get_torch_device,
    sinkhorn_scaling,
    stochastic_neural_sort,
)


class NeuralNDCGLoss(nn.Module):
    def __init__(
        self,
        padded_value_indicator: int = PADDED_Y_VALUE,
        temperature: float = 1.0,
        powered_relevancies: bool = True,
        k: str = None,
        stochastic: bool = False,
        n_samples: int = 32,
        beta: float = 0.1,
        log_scores: bool = True,
        max_iter: int = 50,
        tol: float = 1e-6,
        transposed: bool = False,
    ):
        """Neural NDCG loss function.

        Args:
            padded_value_indicator (_type_, optional): y_trueのインデックスのうち、パディングされた項目を示すインジケータ. Defaults to -1.
            temperature (float, optional): NeuralSortアルゴリズムで使用される温度パラメータ. Defaults to 1.0.
            powered_relevancies (bool, optional): 2^x - 1のゲイン関数を適用するかどうか. Falseの場合はxが適用される. Defaults to True.
            k (int, optional): 損失が切り捨てられるランク（順位）. Defaults to None.
            stochastic (bool, optional): 確率的なバリアントを計算するかどうか. Defaults to False.
            n_samples (int, optional): 確率的な場合に取得するサンプル数. stochastic=Trueの時に使用される. Defaults to 32.
            beta (float, optional): NeuralSortアルゴリズムのベータパラメータ. stochastic=Trueの時に使用される. Defaults to 0.1.
            log_scores (bool, optional): NeuralSortアルゴリズムのlog_scoresパラメータ. stochastic=Trueの時に使用される. Defaults to True.
            max_iter (int, optional): Sinkhornスケーリングの最大反復回数. Defaults to 50.
            tol (float, optional): Sinkhornスケーリングの許容誤差. Defaults to 1e-6.
            transposed (bool, optional): Falseの場合、予測順位をそのまま評価する. モデルが出力するスコアがそのまま評価に直結しやすくなる. Trueの場合, ディスカウントを先に適用し、その後に予測順位が考慮される. 損失関数がより柔軟に設定され、異なるタスクに適した最適化を行うことができる. Defaults to False.
        """
        super(NeuralNDCGLoss, self).__init__()
        self.padded_value_indicator = padded_value_indicator
        self.temperature = temperature
        self.powered_relevancies = powered_relevancies
        self.k = k
        self.stochastic = stochastic
        self.n_samples = n_samples
        self.beta = beta
        self.log_scores = log_scores
        self.max_iter = max_iter
        self.tol = tol
        self.transposed = transposed

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Neural NDCG loss.

        Args:
            y_pred (torch.Tensor): モデルからの予測値。形状は[batch_size, slate_length]。
            y_true (torch.Tensor): 正解ラベル。形状は[batch_size, slate_length]。

        Returns:
            torch.Tensor: Loss値.
        """
        dev = get_torch_device()

        if self.k is None:
            k = y_true.shape[1]
        else:
            k = self.k

        mask = y_true == self.padded_value_indicator

        if self.stochastic:
            P_hat = stochastic_neural_sort(
                y_pred.unsqueeze(-1),
                n_samples=self.n_samples,
                tau=self.temperature,
                mask=mask,
                beta=self.beta,
                log_scores=self.log_scores,
            )
        else:
            P_hat = deterministic_neural_sort(
                y_pred.unsqueeze(-1), tau=self.temperature, mask=mask
            ).unsqueeze(0)

        if not self.transposed:
            P_hat = sinkhorn_scaling(
                P_hat.view(
                    P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]
                ),
                mask.repeat_interleave(P_hat.shape[0], dim=0),
                tol=self.tol,
                max_iter=self.max_iter,
            )
            P_hat = P_hat.view(
                int(P_hat.shape[0] / y_pred.shape[0]),
                y_pred.shape[0],
                P_hat.shape[1],
                P_hat.shape[2],
            )

            P_hat = P_hat.masked_fill(
                mask[None, :, :, None] | mask[None, :, None, :], 0.0
            )
            y_true_masked = y_true.masked_fill(mask, 0.0).unsqueeze(-1).unsqueeze(0)
            if self.powered_relevancies:
                y_true_masked = torch.pow(2.0, y_true_masked) - 1.0

            ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
            discounts = (
                torch.tensor(1.0)
                / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.0)
            ).to(dev)
            discounted_gains = ground_truth * discounts

        else:
            P_hat_masked = sinkhorn_scaling(
                P_hat.view(
                    P_hat.shape[0] * y_pred.shape[0], y_pred.shape[1], y_pred.shape[1]
                ),
                mask.repeat_interleave(P_hat.shape[0], dim=0),
                tol=self.tol,
                max_iter=self.max_iter,
            )
            P_hat_masked = P_hat_masked.view(
                P_hat.shape[0], y_pred.shape[0], y_pred.shape[1], y_pred.shape[1]
            )
            discounts = (
                torch.tensor(1.0)
                / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.0)
            ).to(dev)

            discounts[k:] = 0.0
            discounts = discounts[None, None, :, None]
            discounts = torch.matmul(
                P_hat_masked.permute(0, 1, 3, 2), discounts
            ).squeeze(-1)

            if self.powered_relevancies:
                gains = torch.pow(2.0, y_true) - 1
            else:
                gains = y_true

            discounted_gains = gains.unsqueeze(0) * discounts

        if self.powered_relevancies:
            idcg = (
                dcg(y_true, y_true, ats=[k]).permute(1, 0)
                if not self.transposed
                else dcg(y_true, y_true, ats=[k]).squeeze()
            )
        else:
            idcg = (
                dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)
                if not self.transposed
                else dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).squeeze()
            )

        if not self.transposed:
            discounted_gains = discounted_gains[:, :, :k]
            ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
            idcg_mask = idcg == 0.0
            ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.0)
        else:
            ndcg = discounted_gains.sum(dim=2) / (idcg + DEFAULT_EPS)
            idcg_mask = idcg == 0.0
            ndcg = ndcg.masked_fill(idcg_mask, 0.0)

        assert (ndcg < 0.0).sum() == 0, "every ndcg should be non-negative"
        if idcg_mask.all():
            return torch.tensor(0.0)

        mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
        return -1.0 * mean_ndcg  # -1 cause we want to maximize NDCG
