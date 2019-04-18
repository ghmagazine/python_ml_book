from flask import Flask
app = Flask(__name__)

# TOPページで実行するコードを指定
@app.route('/')
def hello():
    # 表示する文字を指定
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
