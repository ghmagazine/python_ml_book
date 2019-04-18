from sklearn import ensemble
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# ワインのデータをインポート
wine = load_wine()

# 特徴量とラベルデータを取り出す
data = wine.data
target = wine.target

# データを分割
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

# ランダムフォレストをインスタンス化
clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=1)

# 学習データからランダムフォレストが学習
clf = clf.fit(X_train, Y_train)

# 正解率を表示
print (clf.score(X_test, Y_test))
