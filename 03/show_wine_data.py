import pandas as pd
from sklearn.datasets import load_wine

# ワインのデータをインポート
data = load_wine() 

# pandasを用いて特徴量とカラム名を取り出す
data_x = pd.DataFrame(data=data.data,columns=data.feature_names)

# データが持つ特徴量を上から5行表示
print (data_x.head())

# pandasを用いてラベルを取り出す
data_y = pd.DataFrame(data=data.target)

# カラム名が「0」となっており分かりづらいので、「class」に変更
data_y = data_y.rename(columns={0: 'class'})

# データに割り振られたラベルを上から5行表示
print (data_y.head())
