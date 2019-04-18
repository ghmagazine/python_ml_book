# -*- coding: utf-8 -*-

import bz2
import sys
import MeCab
from rdflib import Graph

tagger = MeCab.Tagger('')
tagger.parse('')  # mecab-python3の不具合に対応 https://github.com/SamuraiT/mecab-python3/issues/3

def read_ttl(f):
    """Turtle形式のファイルからデータを読み出す"""
    while True:
        # 高速化のため100KBずつまとめて処理する
        lines = [line.decode("utf-8").rstrip() for line in f.readlines(102400)]
        if not lines:
            break

        for triple in parse_lines(lines):
            yield triple

def parse_lines(lines):
    """Turtle形式のデータを解析して返す"""
    g = Graph()
    g.parse(data='\n'.join(lines), format='n3')
    return g

def tokenize(text):
    """MeCabを用いて単語を分割して返す"""
    node = tagger.parseToNode(text)
    while node:
        if node.stat not in (2, 3):  # 文頭と文末を表すトークンは無視する
            yield node.surface
        node = node.next

with bz2.BZ2File(sys.argv[1]) as in_file:
    for (_, p, o) in read_ttl(in_file):
        if p.toPython() == 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString':
            for line in o.toPython().split('\n'):
                words = list(tokenize(line))
                if len(words) > 20:  # 20単語以下の行は無視する
                    print(' '.join(words))
