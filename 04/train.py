# -*- coding: utf-8 -*-

import logging
import multiprocessing
import sys
from gensim.models import word2vec

logging.basicConfig(level=logging.INFO)  # 学習状況を標準出力に表示する

cpu_count = multiprocessing.cpu_count()  # CPUのコア数を取得する
model = word2vec.Word2Vec(
    word2vec.LineSentence(sys.argv[1]),
    sg=1,              # skip-gramを用いる
    size=100,          # ベクトルの次元数を100に設定
    window=5,          # ウィンドウ幅を5に設定
    min_count=5,       # 最小単語出現数を5に設定
    iter=5,            # イテレーション数を5に設定
    workers=cpu_count  # スレッド数をCPUコア数と同じ値に設定
)
model.save(sys.argv[2])
