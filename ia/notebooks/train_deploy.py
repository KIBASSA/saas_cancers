import os
import sys

script_trainers = os.path.join(os.getcwd(), "../src/trainers")
script_DCGAN = os.path.join(os.getcwd(), "../src/models/Gans/DCGAN")
script_moqs = os.path.join(os.getcwd(), "../src/moqs")
script_utils = os.path.join(os.getcwd(), "../src/utils")
script_scripts = os.path.join(os.getcwd(), "../src/scripts") 
sys.path.append(script_trainers)
sys.path.append(script_DCGAN)
sys.path.append(script_moqs)
sys.path.append(script_utils)
sys.path.append(script_scripts)
from global_helpers import ConfigHandler
from azure_moqs import AzureMLRunMoq
from simple_trainer import ModelTrainer, SimpleClassifierModelTrainer, SimpleClassifierModelTrainerKeras, BalancedClassifierModelTrainerKeras, ResnetClassifierModelTrainer, PaulMooneyCreditModelTrainer
from eval_model import ModelValidateProcessor

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Model

configHandler = ConfigHandler()
config = configHandler.get_file("../config.yaml")

svc_pr = ServicePrincipalAuthentication(
                            tenant_id=config.SPA_TENANTID,
                            service_principal_id=config.SPA_APPLICATIONID,
                            service_principal_password=config.SPA_PASSWORD)
        
ws = Workspace(subscription_id=config.SUBSCRIPTION_VALUE,
                    resource_group=config.RESOURCEGROUP,
                    workspace_name=config.WORKSPACENAME,
                    auth=svc_pr)

run = AzureMLRunMoq(None)
trainer = PaulMooneyCreditModelTrainer()
input_data = "C:\\Users\\kibas\\Desktop\\data_to_azure\\mldata.back"
model_candidate_folder = os.path.join(os.getcwd(),"../src/benchmarks/models")

""" Training
"""
trainer.set_params(run, 2)
trainer.train(input_data, model_candidate_folder)

""" Evaluation

model_validate_processor = ModelValidateProcessor()
[loss, acc, precision, recall, f1_s], model_candidate_file = model_validate_processor.evaluate(input_data, model_candidate_folder)
print("f1_s : ", f1_s)
print("acc : ", acc)
Model.register(workspace=ws,
                    model_path = os.path.join(model_candidate_folder,"classifier.hdf5"),
                    model_name = config.MODEL_NAME,
                    tags={'Training context':'Pipeline'})
"""

