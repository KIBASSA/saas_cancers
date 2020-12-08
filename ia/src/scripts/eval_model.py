
import argparse
from azureml.core import Run
from global_helpers import AzureMLLogsProvider, WebServiceDeployer
from keras.preprocessing.image import ImageDataGenerator
from abstract_trainer import AbstractModelTrainer
from discriminator import disc_network
import os
import shutil
from global_helpers import ConfigHandler
import keras
from keras import backend as K
import keras_metrics
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    print("--- recall :", recall)
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision * recall) / (precision + recall + K.epsilon()))

class ModelValidateProcessor(AbstractModelTrainer):

    

    def evaluate(self,input_data,
                          model_candidate_folder):
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
                                os.path.join(input_data, "eval/"),
                                target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                batch_size=self.BATCH_SIZE_TRAINING_LABELED_SUBSET,
                                class_mode='categorical',
                                shuffle=True) # set as training data

        steps = len(test_generator.filenames)/self.BATCH_SIZE_TRAINING_LABELED_SUBSET

        _, classifier = disc_network()
        classifier_name = "classifier.hdf5"
        model_candidate_file = os.path.join(model_candidate_folder, classifier_name)
        classifier.load_weights(model_candidate_file)

        #classifier.compile(loss="categorical_crossentropy", metrics=["accuracy", f1_m, keras.metrics.Precision(), keras.metrics.Recall()], optimizer="adam")
        #loss, acc, f1_score, precision, recall = classifier.evaluate_generator(test_generator, steps=steps, verbose=0)

        classifier.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
        loss, acc = classifier.evaluate_generator(test_generator, steps=steps, verbose=0)

        result = classifier.predict_generator(test_generator,verbose=1,steps=steps)
        y_true = test_generator.classes
        print("y_true :", y_true)
        y_pred = np.argmax(result, axis=1)
        print("y_pred :", y_pred)
        #print("result :", result)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1_s = f1_score(y_true, y_pred, average='macro')
        #classifier.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
        #loss, acc = classifier.evaluate_generator(test_generator, steps=steps, verbose=0)
        #accuracy, f1_score, precision, recall
        print("f1_score :", f1_s)
        print("---precision :", precision)
        print("---recall :", recall)
        return [loss, acc, precision, recall, f1_s], model_candidate_file

class ModelValidator(AbstractModelTrainer):
    def __init__(self, run, azure_ml_logs_provider):
        super().__init__()
        self.run = run
        self.azure_ml_logs_provider = azure_ml_logs_provider

    def evaluate(self,
                      model_validate_processor,
                          input_data,
                                model_candidate_folder, 
                                    validated_model_folder):
        
        IGNORE_TRAIN_STEP = self.azure_ml_logs_provider.get_tag_from_brother_run("prep_data.py","IGNORE_TRAIN_STEP")   
        if IGNORE_TRAIN_STEP == True:
            print("Ignore evaluate step")
            return
        
        #acc, model_candidate_file = model_validate_processor.evaluate(input_data, model_candidate_folder)
        [_, acc, _, _, _], model_candidate_file = model_validate_processor.evaluate(input_data, model_candidate_folder)
        self.run.log("acc", round(acc,5))
        print("acc : ", round(acc,5))

        classifier_name = "classifier.hdf5"
        validated_model_file = os.path.join(validated_model_folder, classifier_name)
        
        os.makedirs(validated_model_folder)
        
        _ = shutil.copy(model_candidate_file, validated_model_file)

        return [acc]

if __name__ == "__main__":

    # get hold of the current run
    run = Run.get_context()
    azure_ml_logs_provider = AzureMLLogsProvider(run)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, dest='input_data', help='data folder mounting point')
    parser.add_argument('--model_candidate_folder', type=str, dest='model_candidate_folder', help='data destination folder mounting point')
    parser.add_argument('--validated_model_folder', type=str, dest='validated_model_folder', help='data destination folder mounting point')
    parser.add_argument('--mode', type=str, dest="mode")

    args = parser.parse_args()
        
    input_data = args.input_data
    model_candidate_folder = args.model_candidate_folder
    validated_model_folder = args.validated_model_folder
    mode = args.mode
    
    if mode == "execute":
        
        """We create an ConfigProvider object from the existing configuration file.
        This object contains all information of config file and facilitates its access.
        """
        configHandler = ConfigHandler()
        config = configHandler.get_file("config.yaml")

        """We create the instance of the ModelValidator class by passing the Run to it 
            and then we launch the evaluation of model.
        """
        model_validate_processor = ModelValidateProcessor()
        validator = ModelValidator(run, azure_ml_logs_provider)
        validator.evaluate(model_validate_processor, 
                                input_data,
                                    model_candidate_folder, 
                                        validated_model_folder)
    else:
        print("the mode has value '{0}' so no need to execute evaluation step".format(mode))
