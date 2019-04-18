# -*- coding: utf-8 -*-

from torch.nn import Module, EmbeddingBag, Linear, Parameter

class NBoW(Module):
    """Neural Bag-of-Wordsモデルを定義"""
    def __init__(self, class_size, vectors):
        super(NBoW, self).__init__()
        # 単語ベクトルの平均を用いて特徴ベクトルを作成するレイヤー
        self.nbow_layer = EmbeddingBag(vectors.size(0), vectors.size(1))
        # 単語ベクトルの初期値として学習した単語の意味ベクトルを用いる
        self.nbow_layer.weight = Parameter(vectors)
        # 各クラスに対応するスコアの出力を行うレイヤー
        self.output_layer = Linear(vectors.size(1), class_size)

    def forward(self, words):
        """単語IDのリストを入力として、各クラスに対応するスコアを返す"""
        # 単語ベクトルの平均を用いて特徴ベクトルを作成する
        feature_vector = self.nbow_layer(words)
        # 各クラスに対応するスコアの出力を行う
        return self.output_layer(feature_vector)
