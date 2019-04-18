# -*- coding: utf-8 -*-

import sys
import numpy as np
from gensim.models import Word2Vec

COUNT = 3
model_file = sys.argv[1]
target_word = sys.argv[2]

# 学習したモデルのロード
model = Word2Vec.load(model_file)

# 意味ベクトルのノルム (長さ) を調整
model.init_sims(replace=True)

# 指定された単語の意味ベクトルを取得
vec = model[target_word]

# 全単語の意味ベクトルを含んだ行列を取得
emb = model.wv.vectors_norm

# 指定された単語の意味ベクトルとすべての単語の意味ベクトルの類似度を計算
sims = np.dot(emb, vec)

count = 0
# 類似度の高い順にソートして順に処理
for index in np.argsort(-sims):
    word = model.wv.index2word[index]
    if word != target_word:
        print('%s (類似度: %.2f)' % (word, sims[index]))
        count += 1
        if count == COUNT:
            break
