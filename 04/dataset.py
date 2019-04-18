# -*- coding: utf-8 -*-

import os
import MeCab
from torchtext.data import Dataset, Example, Field
from torchtext.vocab import Vectors

tagger = MeCab.Tagger()
tagger.parse('')  # mecab-python3の不具合に対応 https://github.com/SamuraiT/mecab-python3/issues/3

def tokenize(text):
    """MeCabを用いて単語を分割して返す"""
    node = tagger.parseToNode(text)
    ret = []
    while node:
        if node.stat not in (2, 3):  # 文頭と文末を表すトークンは無視する
            ret.append(node.surface)
        node = node.next
    return ret

def load_data(data_dir, emb_file):
    """livedoorニュースコーパスと意味ベクトルをロードする"""
    # torchtextを用いてデータセットの各アイテムの持つフィールドを定義
    # アイテムはtextとlabelの2つのフィールドを持つ
    text_field = Field(sequential=True, tokenize=tokenize)
    label_field = Field(sequential=False, unk_token=None)
    fields = [('text', text_field), ('label', label_field)]
    examples = []

    # データセット内のディレクトリを順に処理する
    for entry in os.scandir(data_dir):
        if entry.is_file():
            continue

        # ディレクトリ名をラベル名として用いる
        label = entry.name

        # ディレクトリ内の記事を順に読み込む
        for doc_file in os.scandir(entry.path):
            if doc_file.name.startswith(label):
                with open(doc_file.path) as f:
                    # 1-2行目はURLと日付のため3行目以降を用いる
                    text = '\n'.join(f.read().splitlines()[2:])
                    # アイテム（torchtextのExampleインスタンス）を作成
                    example = Example.fromlist([text, label], fields)
                    examples.append(example)

    # アイテムのリストとフィールドの定義を用いてDatasetインスタンスを作成
    data = Dataset(examples, fields)

    # 訓練、テスト用のデータセットを70%、30%の分割比率で作成する
    (train_data, test_data) = data.split(0.7)

    # フィールドごとに辞書を作成する
    text_field.build_vocab(train_data)
    label_field.build_vocab(data)

    # 学習した意味ベクトルを読み込む
    vectors = Vectors(emb_file)
    text_field.vocab.load_vectors(vectors)

    return (train_data, test_data, text_field, label_field)
