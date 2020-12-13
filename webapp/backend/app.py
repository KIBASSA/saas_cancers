
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
from models import Patient, PatientEncoder, Doctor, DoctorEncoder
from global_helpers import Helper, ConfigHandler, WorkspaceProvider
import json
import datetime
from flask_api import status
from utilities import PatientImageCloudManager, Consts,AnnotatedDataManager, SampedDataDataManager
from predictor import Predictor
from PIL import Image, ImageOps
import time
import base64
import shutil
import tempfile
from azureml.core.webservice import Webservice
from shutil import copyfile
import requests
# copy configfile for flask app
new_config_path = "config.yaml"
copyfile("../../ia/config.yaml", new_config_path)
configHandler = ConfigHandler()
config = configHandler.get_file("config.yaml")
ws_provider = WorkspaceProvider(config)
work_space,_ = ws_provider.get_ws()

#initialization for flask app
app = Flask(__name__)
api = Api(app)  # type: Api


#initialization for blob storage
blob_manager = BlobStorageManager("DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net")
patient_img_manager = PatientImageCloudManager(blob_manager, config.SERVICE_BLOB)
annotated_data_manager = AnnotatedDataManager(blob_manager, config.SERVICE_BLOB)
sampled_data_manager = SampedDataDataManager(blob_manager, config.SERVICE_BLOB)
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
    patient_model = Patient("", patient_form["name"], patient_form["email"])
    patient_model.registration_date = datetime.datetime.now()
    patient_model.diagnosis_date = datetime.datetime.min
    patient_model.id =  db_api.insert_patient(patient_model)
    
    if "image" in patient_form:
        patient_model.image = patient_img_manager.upload_profile(patient_model, patient_form["image"])
        db_api.update_patient(patient_model)

    return jsonify(PatientEncoder().encode(patient_model))

@app.route('/add_cancer_images', methods=['POST'])
def add_cancer_images():
    if "patient_id" not in request.form:
        return "patient_id must be provided", status.HTTP_400_BAD_REQUEST
    
    if "images" not in request.form:
        return "images must be provided", status.HTTP_400_BAD_REQUEST
    
    patient_id = request.form['patient_id']
    patient_cancer_images = json.loads(request.form['images'])
    
    #get old patient : before update with new images 
    old_patient = db_api.get_patient_by_id(patient_id)
    #upload_cancer_images
    uploaded_images = patient_img_manager.upload_cancer_images(patient_id, len(old_patient.cancer_images), patient_cancer_images)
    db_api.insert_cancer_images(patient_id, uploaded_images)
    patient = db_api.get_patient_by_id(patient_id)
    return jsonify(PatientEncoder().encode(patient))

#The transformation to base64 is done by flask because with angular it is complicated.
@app.route('/get_base64_image', methods=['GET'])
def get_base64_image():
    image_urls = request.args.get('image_urls')
    image_urls = list(image_urls.split(",")) 
    return Helper.get_base64_image_by_urls(image_urls)

@app.route('/predict_cancer', methods=['GET'])
def predict_cancer():
    service = Webservice(workspace=work_space, name='diagnozinferenceservice')
    image_urls = request.args.get('image_urls')
    image_urls = list(image_urls.split(","))
    images_dict = Helper.get_base64_image_by_urls(image_urls)
    data_raw = json.dumps({"data": images_dict})
    prediction = service.run(input_data=data_raw)
    return jsonify(prediction)

@app.route('/update_patients_as_diagnosed', methods=['POST'])
def update_patients_as_diagnosed():
    if "patients" not in request.form:
        return "patients must be provided", status.HTTP_400_BAD_REQUEST

    patients = json.loads(request.form['patients'])
    
    for patient in patients:
        patient_model = Patient(patient["id"], patient["name"])
        patient_model.diagnosis_date = datetime.datetime.now()
        patient_model.is_diagnosed = patient["isDiagnosed"]
        patient_model.has_cancer = patient["hasCancer"]
        db_api.update_patient(patient_model)

    return "Every thing is OK", status.HTTP_200_OK

@app.route('/get_sampled_images', methods=['GET'])
def get_sampled_images():
    images = sampled_data_manager.get_sampled_data()
    return jsonify(images)

@app.route('/upload_annotated_data', methods=['POST'])
def upload_annotated_data():
    if "images" not in request.form:
        return "images must be provided", status.HTTP_400_BAD_REQUEST

    images = json.loads(request.form['images'])
    annotated_data_manager.upload_data(images)
    """
    archive the data because it has just been annotated
    """
    sampled_data_manager.archive()
    return "Every thing is OK", status.HTTP_200_OK

#/users/authenticate
@app.route('/users/authenticate', methods=['POST'])
def users_authenticate():
    if "images" not in request.form:
        return "images must be provided", status.HTTP_400_BAD_REQUEST

    images = json.loads(request.form['images'])
    annotated_data_manager.upload_data(images)
    """
    archive the data because it has just been annotated
    """
    sampled_data_manager.archive()
    return "Every thing is OK", status.HTTP_200_OK

@app.route('/login', methods=['POST'])
def login():
    json_data = request.json
    print('username : ',json_data['username'])
    print('password : ',json_data['password'])
    user = db_api.login(json_data['username'], json_data['password'])
    if user == None:
        return "User Not found", status.HTTP_404_NOT_FOUND 
    print("type(user) :", type(user))
    if type(user) is Patient:
        return jsonify(PatientEncoder().encode(user))
    elif type(user) is Doctor:
        return jsonify(DoctorEncoder().encode(user))
    
    return "User found but no entity", status.HTTP_404_NOT_FOUND

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response


if __name__ == "__main__":
    try:
        app.run()
    finally:
        if os.path.isfile(new_config_path):
            os.remove(new_config_path)
    