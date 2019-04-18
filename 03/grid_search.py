from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# データの準備
wine = load_wine()
data = wine.data
target = wine.target

# 各パラメータの試したい値を配列で指定
parameters = {
         'n_estimators'      : [3, 5, 10, 30, 50, 100],
         'max_features'      : [1, 3, 5, 10],
         'random_state'      : [0],
         'min_samples_split' : [3, 5, 10, 30, 50],
         'max_depth'         : [3, 5, 10, 30, 50]
}

# 分類器やパラメータを引数として渡す
clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5, iid=False)

# 今までと同じ書き方でグリッドサーチが可能
clf.fit(data, target)

# 最も精度の高かったパラメータの値を出力
print(clf.best_estimator_)
