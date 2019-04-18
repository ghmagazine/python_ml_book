import pickle
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
data = wine.data
target = wine.target
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

# 学習済みの決定木アルゴリズムをファイルに保存
with open('model.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
