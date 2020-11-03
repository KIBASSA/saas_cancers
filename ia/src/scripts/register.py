
from discriminator import disc_network
from azureml.core import Run
from azureml.core import Model

class ModelRegister(object):
    def __init__(self, run, config):
        self.run = run
        self.config = config
    
    def register(self,input_data_folder,
                         validated_model_folder,
                           registered_model_folder,
                            azure_ml_logs_provider, 
                                        web_service_deployer):
        
        IGNORE_TRAIN_STEP = azure_ml_logs_provider.get_tag_from_brother_run("prep_data.py","IGNORE_TRAIN_STEP")   
        if IGNORE_TRAIN_STEP == True:
            print("Ignore register step")
            self._execute_sampling_pipeline()
            print("launch sampling state")
            return

        _, classifier = disc_network()
        classifier_name = "classifier.hdf5"
        classifier_file = os.path.join(validated_model_folder, classifier_name)
        classifier.load_weights(classifier_file)

        run.upload_file(name = self.config.MODEL_NAME, path_or_stream = classifier_file)

        Model.register(workspace=self.run.experiment.workspace,
                    model_path = validated_model_file,
                    model_name = self.config.MODEL_NAME,
                    tags={'Training context':'Pipeline'})
        acc = azure_ml_logs_provider.get_log_from_brother_run("eval_model.py", "acc")
        
        #deploy model
        if web_service_deployer.to_deploy(acc):
            print("deploying...")
            web_service_deployer.deploy()
            print("model deployed")
        
if __name__ == "__main__":

    run = Run.get_context()
    azure_ml_logs_provider = AzureMLLogsProvider(run)
    mode = azure_ml_logs_provider.get_tag_from_brother_run("prep_data.py","MODE")
    if mode == "execute":
        parser = argparse.ArgumentParser()
        """The parameter <validated_model_folder> which is output, 
            passed through the pipeline will contain the valided model file.
        """
        parser.add_argument('--input_data', type=str, dest='input_data', help='data folder mounting point')
        parser.add_argument('--validated_model_folder', type=str, dest='validated_model_folder', help='model location')
        parser.add_argument('--registered_model_folder', type=str, dest='registered_model_folder', help='model location')
        """We retrieve the arguments and put them into variables.
        """
        args = parser.parse_args()
        input_data = args.input_data
        validated_model_folder = args.validated_model_folder
        registered_model_folder = args.registered_model_folder
        
        """Since our registration will need information from the configuration file,
        we load the Config object from this file <config.yaml>.
        """
        configHandler = ConfigHandler()
        config = configHandler.get_file("config.yaml")

        """We create our ModelRegister object by passing it azure ml Run and the config object 
        then we launch the registration of the model coming from the validated_model_folder
        """
        
        deployer = WebServiceDeployer(run.experiment.workspace, config)
        register = ModelRegister(run, config)
        register.register(input_data, validated_model_folder,registered_model_folder, azure_ml_logs_provider, deployer)
    else:
        print("the mode has value '{0}' so no need to execute registration step".format(mode))
