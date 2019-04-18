# -*- coding: utf-8 -*-

import sys
import numpy as np
from gensim.models import Word2Vec

model_file = sys.argv[1]
(pos1, pos2, neg) = sys.argv[2:]

# 学習したモデルのロード
model = Word2Vec.load(model_file)

# 意味ベクトルのノルムを調整
model.init_sims(replace=True)

# クエリベクトルを計算
vec = model[pos1] + model[pos2] - model[neg]

# 全単語の意味ベクトルを含んだ行列を取得
emb = model.wv.vectors_norm

# 全単語に対するクエリベクトルの類似度を計算
sims = np.dot(emb, vec)

# 類似度が最大の単語を選択し、予測結果として出力
for index in np.argsort(-sims):
    word = model.wv.index2word[index]
    if word not in (pos1, pos2, neg):
        print('予測結果:', word)
        break
