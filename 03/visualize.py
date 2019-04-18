import pydot
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# ワインのデータをインポート
wine = load_wine()

# 特徴量とラベルデータを取り出す
data = wine.data
target = wine.target

# データを分割
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

# 決定木をインスタンス化
clf = tree.DecisionTreeClassifier()

# 学習データから、決定木に学習をさせる
clf = clf.fit(X_train, Y_train)

# 正解率を表示
print (clf.score(X_test, Y_test))

# DOT言語でグラフを表現した、tree.dotを生成
tree.export_graphviz(
  clf,                               # 決定木インスタンス
  feature_names=wine.feature_names,  # 特徴量の名前
  class_names=wine.target_names,     # 分類先の名前
  filled=True,                       # 最も多数を占める分類先ごとに色分け
  rounded=True,                      # 各ノードのボックスの角を丸くし、Helveticaフォントで見やすく
  out_file='tree.dot',               # 生成されるファイル名を指定
)

# tree.dotを視覚化して、画像で出力
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
