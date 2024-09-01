import torch
import torch.nn as nn
import torch.nn.functional as F


class NDCGLoss(nn.Module):
    def __init__(self, k: int = None):
        """
        Args:
            k (int, optional): NDCG@k, if None, calculate NDCG for the entire list. Defaults to None.
        """
        super(NDCGLoss, self).__init__()
        self.k = k

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            y_pred (torch.Tensor): Predicted scores (not probabilities), shape [batch_size, list_size]
            y_true (torch.Tensor): True relevance scores, shape [batch_size, list_size]

        Returns:
            torch.Tensor: NDCG Loss, shape [batch_size, 1]
        """
        device = y_pred.device
        # Sort true scores by predicted order
        _, indices = torch.sort(y_pred, dim=1, descending=True)
        y_true_sorted = torch.gather(y_true, dim=1, index=indices)

        if self.k is not None:
            y_true_sorted = y_true_sorted[:, : self.k]

        # Calculate DCG
        gains = 2**y_true_sorted - 1
        discounts = torch.log2(
            torch.arange(y_true_sorted.size(1), device=device).float() + 2
        )
        dcg = torch.sum(gains / discounts, dim=1)

        # Ideal DCG
        _, ideal_indices = torch.sort(y_true, dim=1, descending=True)
        y_true_sorted_ideal = torch.gather(y_true, dim=1, index=ideal_indices)

        if self.k is not None:
            y_true_sorted_ideal = y_true_sorted_ideal[:, : self.k]

        ideal_gains = 2**y_true_sorted_ideal - 1
        ideal_dcg = torch.sum(ideal_gains / discounts, dim=1)

        # Normalize DCG by Ideal DCG
        ndcg = dcg / (ideal_dcg + 1e-10)  # add epsilon to prevent division by zero
        loss = 1.0 - ndcg.mean()

        return loss


# 使用例
if __name__ == "__main__":
    # ダミーデータの作成
    y_pred = torch.tensor(
        [[0.2, 0.3, 0.8, 0.4], [0.5, 0.1, 0.4, 0.7]], requires_grad=True
    )
    y_true = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 3]], dtype=torch.float32)

    # NDCG@kの損失計算 (k=Noneで全体)
    ndcg_loss = NDCGLoss(k=3)
    loss = ndcg_loss(y_pred, y_true)
    print("NDCG Loss:", loss.item())

    # 逆伝播
    loss.backward()
