# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn.functional as F
from torchtext.data import Iterator
import dataset
import nbow_model

def train(dataset_dir, emb_file, epoch, batch_size):
    """テキスト分類問題を解くNeural Bag-of-Wordsモデルを訓練する"""
    (train_data, test_data, text_field, label_field) = dataset.load_data(dataset_dir, emb_file)

    # NBoWモデルのインスタンスを作成する
    class_size = len(label_field.vocab)
    nbow = nbow_model.NBoW(class_size, text_field.vocab.vectors)

    # モデルを訓練モードに設定する
    nbow.train()

    # パラメータの更新を行うオプティマイザーを作成する
    optimizer = torch.optim.Adam(nbow.parameters())

    # データセットのイテレータを作成する
    train_iter = Iterator(train_data, batch_size)
    for n in range(epoch):
        for batch in train_iter:
            # オプティマイザーを初期化する
            optimizer.zero_grad()
            # モデルの出力を計算する
            logit = nbow(batch.text.t())
            # 誤差逆伝搬を実行する
            loss = F.cross_entropy(logit, batch.label)
            loss.backward()
            # パラメータを更新する
            optimizer.step()

        # モデルを評価モードに設定する
        nbow.eval()
        # テストデータを用いてモデルの正解率を計算
        (accuracy, num_correct) = compute_accuracy(nbow, test_data)
        print('Epoch: {} Accuracy: {:.2f}% ({}/{})'.format(n + 1, accuracy * 100, num_correct, len(test_data)))
        # モデルを訓練モードに設定する
        nbow.train()

def compute_accuracy(model, test_data):
    """モデルの分類精度を計測し表示する"""
    test_size = len(test_data)
    # テスト用のデータを取り出す
    test_data = next(iter(Iterator(test_data, test_size)))
    # モデルにテストデータを入力
    logit = model(test_data.text.t())
    # 正解したアイテム数と分類精度を計算
    num_correct = (torch.max(logit, 1)[1].view(test_size) == test_data.label).sum().item()
    accuracy = float(num_correct) / test_size
    return (accuracy, num_correct)

if __name__ == '__main__':
    train(dataset_dir=sys.argv[1], emb_file=sys.argv[2], epoch=int(sys.argv[3]),
          batch_size=int(sys.argv[4]))
