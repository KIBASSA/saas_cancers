
import os
import sys
sys.path.append("../../db_api")

from flask import Flask,render_template,Response, jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from cancer_db_api import CancerDBAPI
from models import Patient, PatientEncoder
app = Flask(__name__)
api = Api(app)  # type: Api

db_api = CancerDBAPI()

@app.route("/")
def hello():
    return render_template("index.html")


@app.route('/exams')
def get_exams():
    print("KIBS")
    data = [{"test":1}, {"test":2}]
    return jsonify(data)

@app.route('/undiagnosed_patients')
def get_get_undiagnosed_patients():
    result = db_api.get_undiagnosed_patients()
    response = []
    for patient in result:
        response.append(PatientEncoder().encode(patient))
    #patient = Patient("qsdq","qsddqsd", "dsqq", False, False)
    #{
    #              name = "Martin Smith",
    #              image = "https://www.bootstrapdash.com/demo/breeze/angular/preview/demo_1/assets/images/faces/face1.jpg",
    #              is_diagnosed = True,
    #              has_cancer = True                  
    #            }
    #PatientEncoder().encode(employee)
    return jsonify(response)

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response
    
if __name__ == "__main__":
     app.run()