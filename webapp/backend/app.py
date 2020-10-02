
import os
import sys
sys.path.append("../../db_api")
sys.path.append("../../cloud_api")

from flask import Flask,render_template,Response, jsonify, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from cancer_db_api import CancerDBAPI
from apis_utilities import BlobStorageManager
from models import Patient, PatientEncoder
import json
import datetime
app = Flask(__name__)
api = Api(app)  # type: Api

from utilities import PatientImageCloudManager

#initialization
blob_manager = BlobStorageManager("DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net")
patient_img_manager = PatientImageCloudManager(blob_manager)
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
def get_undiagnosed_patients():
    #db_api = CancerDBAPI()
    result = db_api.get_diagnosed_patients(False)
    response = []
    for patient in result:
        response.append(PatientEncoder().encode(patient))
    print(jsonify(response))
    return jsonify(response)

@app.route('/diagnosed_patients')
def get_diagnosed_patients():
    #db_api = CancerDBAPI()
    result = db_api.get_diagnosed_patients(True)
    response = []
    for patient in result:
        response.append(PatientEncoder().encode(patient))
    print(jsonify(response))
    return jsonify(response)

@app.route('/all_patients')
def all_patients():
    #db_api = CancerDBAPI()
    result = db_api.get_all_patients()
    response = []
    for patient in result:
        response.append(PatientEncoder().encode(patient))
    return jsonify(response)

@app.route('/add_patient', methods=['POST'])
def add_patient():
    patient_form = request.form['patient']
    patient_form = json.loads(patient_form)
    patient_model = Patient("", patient_form["name"])
    patient_model.registration_date = datetime.datetime.now()
    patient_model.diagnosis_date = datetime.datetime.min
    patient_model.id =  db_api.insert_patient(patient_model)
    
    if "image" in patient_form:
        patient_model.image = patient_img_manager.upload_profile(patient_model, patient_form["image"])
        db_api.update_patient(patient_model)

    return jsonify(PatientEncoder().encode(patient_model))

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response
    
if __name__ == "__main__":
     app.run()