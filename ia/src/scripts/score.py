
import json
import numpy as np
import os
import pickle
import joblib
from discriminator import disc_network
from predictor import Predictor

def init():
    global model
    for root, dir_, files in os.walk(os.getcwd()):
        print("dir_", dir_)
        for filename in files:
            print("filename :", filename)

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'classifier.hdf5')
    #model = joblib.load(model_path)
    _, model = disc_network()
    model.load_weights(model_path)


def run(raw_data):
    try:
        predictor = Predictor()
        predictions = predictor.predict(model, raw_data, (50,50), ["cancer", "not cancer"])
        return jsonify(predictions)
        #data = json.loads(raw_data)['data']
        #data = np.array(data)
        #result = model.predict(data)
        #return result.tolist()

    except Exception as e:
        result = str(e)
        return result