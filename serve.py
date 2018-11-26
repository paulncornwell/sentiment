from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/sentiment")
def sentiment():
    return jsonify({"gold": 123})


if __name__ == "__main__":
    app.run()