from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

wine = load_wine()
data = wine.data
target = wine.target
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

# テストデータのラベルを予測
Y_pred = clf.predict(X_test)

# 各クラスの適合率と再現率を表示
print (classification_report(Y_test, Y_pred, target_names=wine.target_names))
