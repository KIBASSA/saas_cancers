
import json
import numpy as np
import os
import pickle
import joblib
from discriminator import disc_network # pylint: disable=import-error
from predictor import Predictor # pylint: disable=import-error
import traceback

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'classifier.hdf5')
    #model = joblib.load(model_path)
    _, model = disc_network()
    model.load_weights(model_path)


def run(raw_data):
    try:
        print("raw_data :", raw_data)
        print("type(raw_data) :", type(raw_data))
        data = json.loads(raw_data)['data']
        predictor = Predictor()
        predictions = predictor.predict(model, data, (50,50), ["cancer", "not cancer"])
        return predictions
    except Exception as e:
        return [str(e), traceback.format_exc()]