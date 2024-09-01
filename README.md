# ListwiseRankingLoss-NeuralNDCG

リストワイズランキングロスは、リスト全体のランキングを一度に評価します。これにより、アイテムの相対的な順序だけでなく、リスト全体のランキングの質を最適化します。  
代表的な Ranking Loss は、Pairwise Ranking Loss, Listwise Ranking Loss があります。  
このレポジトリは、Listwise Ranking Loss の代表的な Neural Normalized Discounted Cumulative Gain (NeuralNDCG)の実装を公開する。  
NeuralNDCG は、学習ランキング（LTR）における最適化と評価のギャップを埋めるために提案された新しい手法です。通常の正規化割引累積利得（NDCG）メトリックを滑らかに近似し、NeuralSort と呼ばれるソートの微分可能な近似を使用することで、評価メトリックに基づく直接的な最適化を実現しています。実験結果から、提案手法が従来の方法と比較して優れた性能を示すことが確認されています。

## 特徴

- 直接的な NDCG 最適化: NeuralSort を使用して、NDCG メトリックの滑らかな近似を行い、直接的な最適化が可能です。
- 確率的および決定論的アプローチ: 確率的アプローチと決定論的アプローチの両方に対応し、タスクに応じた適切な選択が可能です。
- 柔軟なハイパーパラメータ: temperature, beta, max_iter, tol などのハイパーパラメータを通じて、トレーニングプロセスを細かく調整可能です。

**利点**: リスト全体のランキングの質を直接最適化できるため、最終的なランキングの精度が高くなる傾向があります。  
**欠点**: 計算コストが高く、特にリストが大きい場合には効率が悪くなる可能性があります。

## インストール

    ```bash
    git clone https://github.com/yourusername/NeuralNDCG.git
    cd NeuralNDCG
    pip install -r requirements.txt
    pip install .
    ```

Git リポジトリから直接インストールできます.

    ```bash
    pip install git+https://github.com/JohnYKiyo/ListwiseRankingLoss-NeuralNDCG.git
    ```

## 評価方法

リスト全体に対して損失を定義し、そのリストが正しいランキング順序にどれだけ近いかを評価します。

### Normalized Discounted Cumulative Gain (NDCG)

$$
NDCG = \frac{DCG}{IDCG}, \quad DCG = \sum_{i=1}^{n} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

ここで、$rel_i$ はアイテム i の関連度スコア、IDCG は理想的なランキングの DCG です。

## Sample Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from neuralndcgloss.loss import NeuralNDCGLoss
# 3層のシンプルなニューラルネットワークの定義
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# モデル、データ、ロス、オプティマイザの準備
input_size = 10  # 入力特徴量の数
hidden_size = 50  # 隠れ層のユニット数
output_size = 5  # 出力スコアの数（ランキングのスレートサイズ）

model = SimpleNN(input_size, hidden_size, output_size)
model.to('mps')
criterion = NeuralNDCGLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ダミーデータの作成
batch_size = 16
X = torch.randn(batch_size, input_size)  # ランダムな入力データ
y_true = torch.randint(0, 5, (batch_size, output_size)).float()  # ランダムなターゲットランキング


# トレーニングループ
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # 順伝播
    optimizer.zero_grad()
    y_pred = model(X.to('mps'))

    # Neural NDCG Lossの計算
    loss = criterion(y_pred.to('mps'), y_true.to('mps'))

    # 逆伝播と最適化
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
```
