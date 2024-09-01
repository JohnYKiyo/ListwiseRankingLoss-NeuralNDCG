# ListwiseRankingLoss-NDCG

ランキング損失（Ranking Loss）: 順位そのものを最適化したい場合のロス関数です。
代表的な Ranking Loss は、Pairwise Ranking Loss, Listwise Ranking Loss があります。
このレポジトリは、Listwise Ranking Loss の代表的な Normalized Discounted Cumulative Gain (NDCG)の実装を公開する。

## 概要

リストワイズランキングロスは、リスト全体のランキングを一度に評価します。これにより、アイテムの相対的な順序だけでなく、リスト全体のランキングの質を最適化します。

## 評価方法

リスト全体に対して損失を定義し、そのリストが正しいランキング順序にどれだけ近いかを評価します。

### Normalized Discounted Cumulative Gain (NDCG)

$$
NDCG = \frac{DCG}{IDCG}, \quad DCG = \sum_{i=1}^{n} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

ここで、$rel_i$ はアイテム i の関連度スコア、IDCG は理想的なランキングの DCG です。

## 特徴

利点: リスト全体のランキングの質を直接最適化できるため、最終的なランキングの精度が高くなる傾向があります。  
欠点: 計算コストが高く、特にリストが大きい場合には効率が悪くなる可能性があります。
