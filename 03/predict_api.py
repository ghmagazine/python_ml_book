import pickle
import numpy as np
from flask import Flask, jsonify, request
app = Flask(__name__)

# pickleを用いて保存された学習結果の読み込み
with open('model.pkl', mode='rb') as fp:
    clf = pickle.load(fp)

# 呼び出されるURIとコードを対応付け
@app.route('/', methods=['POST'])
def predict():
    # 送られてきたデータの取り出し
    unknown_wine = request.json['wine']

    # listで受け取ったデータをNumPy配列に変換
    unknown_wine = np.array([unknown_wine])

    # 学習結果を用いて予測
    pred_label = clf.predict(unknown_wine)

    # レスポンスを返すためにlistに変換
    pred_label = pred_label.tolist()

    # JSON形式にしてレスポンスを返す
    return jsonify(dict(pred_label=pred_label[0]))

if __name__ == '__main__':
    app.run()
