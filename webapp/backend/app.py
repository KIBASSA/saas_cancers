from flask import Flask,render_template,Response, jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)  # type: Api

@app.route("/")
def hello():
    return render_template("index.html")


@app.route('/exams')
def get_exams():
    print("KIBS")
    data = [{"test":1}, {"test":2}]
    return jsonify(data)

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response
    
if __name__ == "__main__":
     app.run()