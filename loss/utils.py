import numpy as np
import torch

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1


def get_torch_device():
    """
    PyTorchで利用可能なデバイスを取得します。
    :return: CUDA対応のGPUが利用可能であればそれを、なければCPUを返します。
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhornスケーリングの手続き。
    :param mat: 行列テンソル（形状はN x M x M、Nはバッチサイズ）
    :param mask: マスクテンソル（形状はN x M）
    :param tol: Sinkhornスケーリングの許容誤差
    :param max_iter: Sinkhornスケーリングの最大反復回数
    :return: 二重確率行列のテンソル
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if (
            torch.max(torch.abs(mat.sum(dim=2) - 1.0)) < tol
            and torch.max(torch.abs(mat.sum(dim=1) - 1.0)) < tol
        ):
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(s, tau, mask):
    """
    確定的なNeuralSortアルゴリズム。
    :param s: ソート対象の値（形状は[batch_size, slate_length]）
    :param tau: ソフトマックス関数の温度パラメータ
    :param mask: パディング要素を示すマスク
    :return: 近似的な置換行列（形状は[batch_size, slate_length, slate_length]）
    """
    device = get_torch_device()
    n = s.size(1)
    one = torch.ones((n, 1), dtype=torch.float32, device=device)

    s = s.masked_fill(mask[:, :, None], -1e8)
    a_s = torch.abs(s - s.permute(0, 2, 1)).masked_fill(
        mask[:, :, None] | mask[:, None, :], 0.0
    )
    B = torch.matmul(a_s, torch.matmul(one, one.T))

    temp = [
        n - m + 1 - 2 * (torch.arange(n - m, device=device) + 1)
        for m in mask.squeeze(-1).sum(dim=1)
    ]
    scaling = torch.stack(
        [
            torch.cat((t.type(torch.float32), torch.zeros(n - len(t), device=device)))
            for t in temp
        ]
    )

    C = torch.matmul(s.masked_fill(mask[:, :, None], 0.0), scaling.unsqueeze(-2))
    p_max = (
        (C - B)
        .permute(0, 2, 1)
        .masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    )

    return torch.nn.functional.softmax(p_max / tau, dim=-1)


def sample_gumbel(samples_shape, device, eps=1e-10):
    """
    Gumbel分布からのサンプリング。
    :param samples_shape: 出力サンプルテンソルの形状
    :param device: 出力サンプルテンソルのデバイス
    :param eps: ログ関数のイプシロン
    :return: Gumbel分布に従ったサンプルテンソル
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(
    s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10
):
    """
    ストキャスティックなNeuralSortアルゴリズム。
    :param s: ソート対象の値（形状は[batch_size, slate_length]）
    :param n_samples: 各置換行列のサンプル数
    :param tau: ソフトマックス関数の温度パラメータ
    :param mask: パディング要素を示すマスク
    :param beta: Gumbel分布のスケールパラメータ
    :param log_scores: Gumbel摂動前にスコアに対して対数を取るかどうか
    :param eps: ログ関数のイプシロン
    :return: 近似的な置換行列（形状は[n_samples, batch_size, slate_length, slate_length]）
    """
    device = get_torch_device()
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, s.size(0), s.size(1), 1], device=device)

    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * s.size(0), s.size(1), 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)
    p_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)

    return p_hat.view(n_samples, s.size(0), s.size(1), s.size(1))


def dcg(
    y_pred,
    y_true,
    ats=None,
    gain_function=lambda x: torch.pow(2, x) - 1,
    padding_indicator=PADDED_Y_VALUE,
):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(
        y_pred, y_true, padding_indicator
    )

    discounts = (
        torch.tensor(1)
        / torch.log2(
            torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0
        )
    ).to(device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, : np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def __apply_mask_and_get_true_sorted_by_preds(
    y_pred, y_true, padding_indicator=PADDED_Y_VALUE
):
    mask = y_true == padding_indicator

    y_pred[mask] = float("-inf")
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)
