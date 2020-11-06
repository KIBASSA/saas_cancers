
import os
import sys
sys.path.append("../../db_api")
sys.path.append("../../cloud_api")
sys.path.append("../../ia/src/models/Gans/DCGAN")
sys.path.append("../../ia/src/utils")
sys.path.append("../../ia/src/deploy")
import joblib
from flask import Flask,render_template,Response, jsonify, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from cancer_db_api import CancerDBAPI
from apis_utilities import BlobStorageManager
from models import Patient, PatientEncoder
import json
import datetime
from flask_api import status
from utilities import PatientImageCloudManager
from predictor import Predictor
from PIL import Image, ImageOps
import time
app = Flask(__name__)
api = Api(app)  # type: Api


#initialization
blob_manager = BlobStorageManager("DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net")
patient_img_manager = PatientImageCloudManager(blob_manager)
db_api = CancerDBAPI()

@app.route("/")
def hello():
    JsonTest = [{"test1":"value1"}, {"test1":"value1"}]
    return render_template("index.html", value=JsonTest)


def _get_structure_patients_data(data):
    response = []
    for patient in data:
        response.append(PatientEncoder().encode(patient))
    print(jsonify(response))
    return jsonify(response)

@app.route('/patient_awaiting_diagnosis')
def patient_awaiting_diagnosis():
    result = db_api.patient_awaiting_diagnosis()
    return _get_structure_patients_data(result)

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

@app.route('/get_patient_by_id', methods=['GET'])
def get_patient():
    patient = db_api.get_patient_by_id(request.args.get('id'))
    return jsonify(PatientEncoder().encode(patient))

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

@app.route('/add_cancer_images', methods=['POST'])
def add_cancers_images():
    if "patient_id" not in request.form:
        return "patient_id must be provided", status.HTTP_400_BAD_REQUEST
    
    if "images" not in request.form:
        return "images must be provided", status.HTTP_400_BAD_REQUEST
    
    patient_id = request.form['patient_id']
    patient_cancer_images = json.loads(request.form['images'])
    
    #upload_cancer_images
    uploaded_images = patient_img_manager.upload_cancer_images(patient_id, patient_cancer_images)
    #print("uploaded_images :", uploaded_images)
    db_api.insert_cancer_images(patient_id, uploaded_images)
    patient = db_api.get_patient_by_id(patient_id)
    return jsonify(PatientEncoder().encode(patient))

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    #print(request.json)
    #classifier_file = os.path.join("classifier/classifier.hdf5")
    #joblib.load(classifier_file)
    #return jsonify({})
    """
    from discriminator import disc_network
    base64Dict = json.loads(request.json)

    classifier_file = os.path.join("classifier/classifier.hdf5")
    if not os.path.isfile(classifier_file):
        raise Exception("classifier doesn't exist")
    
    _, classifier = disc_network()
    classifier.load_weights(classifier_file)

    predictor = Predictor()
    predictions = predictor.predict(classifier, base64Dict, (50,50), ["cancer", "not cancer"])

    return jsonify(predictions)
    """
    #base64Dict = json.loads(request.json)
    time.sleep(2.4)
    response = []
    for index, item in enumerate(request.json.items()):
        img_file_name, base64Img = item
        cancer_value = "cancer"
        if index % 2 == 0:
            cancer_value = "not cancer"
        response.append({img_file_name:cancer_value})
    #base64Dict = request.json
    #print("len(base64Dict) : ", len(request.json))
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