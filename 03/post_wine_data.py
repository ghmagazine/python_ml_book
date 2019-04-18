import requests
from sklearn.datasets import load_wine

wine = load_wine()
data = wine.data
target = wine.target

# APIのURL
url = 'http://127.0.0.1:5000/'

# 1つ目の特徴量を取り出す
unknown_wine = data[0]

# 1つ目のラベルを取り出す
label = target[0]

# NumPy配列をlistへ変換
post_data = {'wine': unknown_wine.tolist()}

# APIへPOST
response = requests.post(url, json=post_data).json()

# 予測結果の確認
print ('正解のラベル: %d' % label)
print ('予測結果のラベル: %d' % response['pred_label'])
