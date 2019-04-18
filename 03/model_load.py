import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

wine = load_wine()
data = wine.data
target = wine.target
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

# 学習済みの決定木アルゴリズムをファイルから読み込む
with open('model.pkl', mode='rb') as fp:
    clf = pickle.load(fp)

# テストデータのラベルを予測
Y_pred = clf.predict(X_test)

# `evaluation.py`とまったく同じ値が出力されることを確認
print (classification_report(Y_test, Y_pred, target_names=wine.target_names))
