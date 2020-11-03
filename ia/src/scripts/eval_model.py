
import argparse
from azureml.core import Run
from global_helpers import AzureMLLogsProvider, WebServiceDeployer
from keras.preprocessing.image import ImageDataGenerator
from abstract_model import AbstractProcessorModel
from discriminator import disc_network
class ModelValidator(AbstractProcessorModel):
    def __init__(self, ws, config, azure_ml_logs_provider):
        super().__init__()
        self.ws = ws
        self.config = config
        self.azure_ml_logs_provider = azure_ml_logs_provider

    def evaluate(self,input_data,
                                model_candidate_folder, 
                                    validated_model_folder):
        
        IGNORE_TRAIN_STEP = self.azure_ml_logs_provider.get_tag_from_brother_run("prep_data.py","IGNORE_TRAIN_STEP")   
        if IGNORE_TRAIN_STEP == True:
            print("Ignore evaluate step")
            return

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
                                os.path.join(input_data, "diagnoz/mldata/train_data/"),
                                target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                batch_size=self.BATCH_SIZE_TRAINING_LABELED_SUBSET,
                                class_mode='categorical') # set as training data

        steps = len(test_generator.filenames)/self.BATCH_SIZE_TRAINING_LABELED_SUBSET

        _, classifier = disc_network()
        classifier_name = "classifier.hdf5"
        classifier.load_weights(os.path.join(model_candidate_file, classifier_name))

        classifier.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
        loss, acc = classifier.evaluate_generator(test_testing_generator, steps=steps, verbose=0)
        self.run.log("acc", round(acc,5))

        validated_model_file = os.path.join(validated_model_folder, classifier_name)
        
        os.makedirs(validated_model_folder)
        
        _ = shutil.copy(model_candidate_file, validated_model_file)

       
if __name__ == "__main__":

    # get hold of the current run
    run = Run.get_context()
    azure_ml_logs_provider = AzureMLLogsProvider(run)
    mode = azure_ml_logs_provider.get_tag_from_brother_run("prep_data.py","MODE")
    
    if mode == "execute":
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_data', type=str, dest='input_data', help='data folder mounting point')
        parser.add_argument('--model_candidate_folder', type=str, dest='model_candidate_folder', help='data destination folder mounting point')
        
        parser.add_argument('--validated_model_folder', type=str, dest='validated_model_folder', help='data destination folder mounting point')

        args = parser.parse_args()
        
        input_data = args.input_data
        model_candidate_folder = args.model_candidate_folder
        validated_model_folder = args.validated_model_folder
        
        """We create an ConfigProvider object from the existing configuration file.
        This object contains all information of config file and facilitates its access.
        """
        configHandler = ConfigHandler()
        config = configHandler.get_file("config.yaml")

        """We create the instance of the ModelValidator class by passing the Run to it 
            and then we launch the evaluation of model.
        """
        validator = ModelValidator(run, azure_ml_logs_provider)
        validator.evaluate(input_data,
                                model_candidate_folder, 
                                    validated_model_folder)
    else:
        print("the mode has value '{0}' so no need to execute evaluation step".format(mode))
