# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# 学習したモデルのロード
model = Word2Vec.load(sys.argv[1])

# 「砂糖」に近い単語を50個取得して視覚化を行う
words = [word for (word, score) in model.wv.most_similar('砂糖', topn=50)]
words.append('砂糖')
vectors = np.vstack([model[word] for word in words])

# t-SNEで意味ベクトルを2次元の空間にマップする
tsne = TSNE(n_components=2)
Y = tsne.fit_transform(vectors)

# 散布図上にベクトルをあらわす点を描画する
x_coords = Y[:, 0]
y_coords = Y[:, 1]
plt.scatter(x_coords, y_coords)

# 散布図上に単語を描画する
for (word, x, y) in zip(words, x_coords, y_coords):
    plt.annotate(word, xy=(x, y), xytext=(5, -10), textcoords='offset points')

# 散布図を書き出す
plt.savefig(sys.argv[2])
