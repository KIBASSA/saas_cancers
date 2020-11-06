from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.datastore import Datastore
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from msrest.exceptions import HttpOperationError
from azureml.core.runconfig import RunConfiguration
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig, Model
from azureml.pipeline.core import PipelineEndpoint
from azureml.exceptions import WebserviceException
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import yaml
import ntpath
import os
from os.path import isfile, join
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np

class ImagePathListUploader(object):
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
        self.host = "https://diagnozstorage.blob.core.windows.net/"
    
    def upload(self,files_source, blob_container_dest):
        for file_source in files_source:
            file_source_name = ntpath.basename(file_source)
            self.blob_manager.upload(blob_container_dest, file_source, overwrite = True)
            uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container_dest, file_source_name)
            print("uploaded_image :", uploaded_image)

class AzureMLLogsProvider:
    def __init__(self, run):
        self.run = run
    
    def get_log_from_brother_run(self, script_name, log_name):
        if not self.run.parent :
            raise Exception("this run has not parent")
        
        log_value = None
        
        for brother_run in self.run.parent.get_children():
            if brother_run.get_details()["runDefinition"]["script"] != script_name:
                continue
            run_metrics = brother_run.get_metrics()
            
            if log_name in run_metrics:
                log_value = run_metrics[log_name]
                print("log_value :", log_value)

        return  log_value
    
    def get_tag_from_brother_run(self, script_name, tag_name):
        if not self.run.parent :
            raise Exception("this run has not parent")

        tag_value = None
        for brother_run in self.run.parent.get_children():
            if brother_run.get_details()["runDefinition"]["script"] != script_name:
                continue
            run_tags = brother_run.get_tags()
            
            if tag_name in run_tags:
                tag_value = run_tags[tag_name]
                print("tag_value :", tag_value)

        #check if bool
        if (tag_value == "True"):
            tag_value = True
        elif (tag_value == "False"):
            tag_value = False

        return  tag_value

class WebServiceDeployer:
    def __init__(self, ws, config):
        self.ws = ws
        self.config = config
        
    def to_deploy(self, principal_metric_value):
       return principal_metric_value >= self.config.DEPLOY_THRESHOLD

    def deploy(self):
        myenv = CondaDependencies()                                                        
        myenv.add_pip_package("azureml-sdk")
        myenv.add_pip_package("joblib")
        myenv.add_pip_package("tensorflow")
        myenv.add_pip_package("Pillow")
        myenv.add_pip_package("azureml-dataprep[pandas,fuse]>=1.1.14")

        with open("diagnoz_env.yml","w") as f:
            f.write(myenv.serialize_to_string())

        huml_env = Environment.from_conda_specification(name="diagnoz_env", file_path="diagnoz_env.yml")

        inference_config = InferenceConfig(entry_script="score.py",source_directory='.', environment=huml_env)
        print("file deployement : ")
        for root, dir_, files in os.walk(os.getcwd()):
            print("dir_", dir_)
            for filename in files:
                print("filename :", filename)

        aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                    memory_gb=1, 
                                                    tags={"data": "cancer-data",  "method" : "tensorflow"}, 
                                                    description='Predicting cancer with tensorflow')

        try:
            AciWebservice(self.ws, self.config.DEPLOY_SERVICE_NAME).delete()
            print("webservice deleted")
        except WebserviceException:
            pass

        model = self.ws.models[self.config.MODEL_NAME]
        

        service = Model.deploy(workspace=self.ws, 
                            name=self.config.DEPLOY_SERVICE_NAME, 
                            models=[model], 
                            inference_config=inference_config, 
                            deployment_config=aciconfig)
        
        service.wait_for_deployment(show_output=True)
        print("success deployement")

class FilesProviders:
    @staticmethod
    def get_path_files(root, exclude_files=[]):
        """[summary]

        Arguments:
            root {[type]} -- [description]

        Keyword Arguments:
            exclude_files {list} -- [description] (default: {[]})

        Returns:
            [type] -- [description]
        """
        result = []
        for root, _, files in os.walk(root):
            for filename in files:
                filepath = join(root, filename)
                dirname = os.path.basename(filepath)
                if dirname in exclude_files:
                    continue
                if filename in exclude_files:
                    continue
                result.append(filepath)

        return result

class WorkspaceProvider:
    def __init__(self, config):
        """Initializing WorkspaceProvider's class from config object

        Arguments:
            config {object} -- Object containing all the invalidations of the yaml config file
        """
        self.config = config
    
    def get_ws(self):
        """Creates the Workspace (ws) using information from config object.

        Returns:
            Workspace -- Defines an Azure Machine Learning resource for managing training and deployment artifacts.
        """
        print("tenant_id:", self.config.SPA_TENANTID)
        print("service_principal_id:", self.config.SPA_APPLICATIONID)
        print("service_principal_password:", self.config.SPA_PASSWORD)
        print("subscription_id:", self.config.SPA_TENANTID)
        print("service_principal_id:", self.config.SPA_APPLICATIONID)
        print("service_principal_password:", self.config.SPA_PASSWORD)
        svc_pr = ServicePrincipalAuthentication(
                            tenant_id=self.config.SPA_TENANTID,
                            service_principal_id=self.config.SPA_APPLICATIONID,
                            service_principal_password=self.config.SPA_PASSWORD)

        
        ws = Workspace(subscription_id=self.config.SUBSCRIPTION_VALUE,
                            resource_group=self.config.RESOURCEGROUP,
                            workspace_name=self.config.WORKSPACENAME,
                            auth=svc_pr
                )
        return ws, svc_pr
        
class ComputeTargetConfig:
    @staticmethod
    def config_create(ws,cluster_name, vm_type, min_nodes, max_nodes,idle_seconds ):
        #Create or Attach existing compute resource
        # choose a name for your cluster
        compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", cluster_name)
        compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", min_nodes)
        compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", max_nodes)

        # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
        vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", vm_type)

        print("#### vm_type : ", vm_type)
        if compute_name in ws.compute_targets:
            compute_target = ws.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print("found compute target: " + compute_name)
        else:
            print("creating new compute target...")
            provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                        min_nodes = compute_min_nodes, 
                                                                        max_nodes = compute_max_nodes,
                                                                        idle_seconds_before_scaledown = idle_seconds)
            # create the cluster
            compute_target = ComputeTarget.create(ws, compute_name.strip(), provisioning_config)
            
            # can poll for a minimum number of nodes and for a specific timeout. 
            # if no min node count is provided it will use the scale settings for the cluster
            compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
            
            # For a more detailed view of current AmlCompute status, use get_status()
            print(compute_target.get_status().serialize())
        
        return compute_target

    @staticmethod
    def config_attach(ws,compute_target_name, resource_id,username, password):
        try:
            attached_dsvm_compute = RemoteCompute(workspace=ws, name=compute_target_name)
            print('found existing:', attached_dsvm_compute.name)
        except ComputeTargetException:
            # Attaching a virtual machine using the public IP address of the VM is no longer supported.
            # Instead, use resourceId of the VM.
            # The resourceId of the VM can be constructed using the following string format:
            # /subscriptions/<subscription_id>/resourceGroups/<resource_group>/providers/Microsoft.Compute/virtualMachines/<vm_name>.
            # You can also use subscription_id, resource_group and vm_name without constructing resourceId.
            
            attach_config = RemoteCompute.attach_configuration(resource_id=resource_id,
                                                                ssh_port=22,
                                                                username=username,
                                                                password=password)
            attached_dsvm_compute = ComputeTarget.attach(ws, compute_target_name.strip(), attach_config)
            attached_dsvm_compute.wait_for_completion(show_output=True)

        return attached_dsvm_compute
        

class DataStoreConfig:
    @staticmethod
    def config(ws, blob_datastore_name,account_name,container_name,account_key):
        
        try:
            blob_datastore = Datastore.get(ws, blob_datastore_name)
            print("Found Blob Datastore with name: %s" % blob_datastore_name)
        except HttpOperationError:
            blob_datastore = Datastore.register_azure_blob_container(
                workspace=ws,
                datastore_name=blob_datastore_name,
                account_name=account_name, # Storage account name
                container_name=container_name, # Name of Azure blob container
                account_key=account_key) # Storage account key
            print("Registered blob datastore with name: %s" % blob_datastore_name)
        
        return blob_datastore

class ConfigProvider:
    def __init__(self, config_path):
        self.config_path = config_path

    def _load_data(self):
        with open(self.config_path) as stream:
            data = yaml.safe_load(stream)
        return data
    
    def load(self):
        data = self._load_data()

        #AmlComputes
        self.AML_COMPUTE_PREP_CLUSTER_NAME = data["Azure"]["AmlComputes"]["DataPreparation"]["ClusterName"]
        self.AML_COMPUTE_PREP_CLUSTER_VM_TYPE = data["Azure"]["AmlComputes"]["DataPreparation"]["ClusterType"]
        self.AML_COMPUTE_DS_CLUSTER_NAME = data["Azure"]["AmlComputes"]["DataScience"]["ClusterName"]
        self.AML_COMPUTE_DS_CLUSTER_VM_TYPE = data["Azure"]["AmlComputes"]["DataScience"]["ClusterType"]
        #self.AML_COMPUTE_SAMPLING_CLUSTER_NAME = data["Azure"]["AmlComputes"]["Sampling"]["ClusterName"]
        #self.AML_COMPUTE_SAMPLING_DS_CLUSTER_VM_TYPE = data["Azure"]["AmlComputes"]["Sampling"]["ClusterType"]

        self.AML_COMPUTE_CLUSTER_MIN_NODES = data["Azure"]["AmlComputes"]["ClusterMinNode"]
        self.AML_COMPUTE_CLUSTER_MAX_NODES = data["Azure"]["AmlComputes"]["ClusterMaxNode"]
        self.IDLE_SECONDS_BEFORE_SCALEDOWN = data["Azure"]["AmlComputes"]["IdleSecondes_Before_Scaledown"]

        #StorageAccount
        self.BLOB_DATASTORE_NAME = data["Azure"]["StorageAccount"]["BlobDatastoreName"]
        self.ACCOUNT_NAME = data["Azure"]["StorageAccount"]["AccountName"]
        self.CONTAINER_NAME = data["Azure"]["StorageAccount"]["ContainerName"]
        self.ACCOUNT_KEY = data["Azure"]["StorageAccount"]["AccountKey"]
        self.BLOB_STORAGE_CONNECTION_STRING = data["Azure"]["StorageAccount"]["BlobStorageConnectionString"]
        #Azureml
        self.LOCATION = data["Azure"]["Azureml"]["Location"]
        self.RESOURCEGROUP = data["Azure"]["Azureml"]["ResourceGroup"]
        self.WORKSPACENAME = data["Azure"]["Azureml"]["WorkspaceName"]


        #ExperimentName
        #self.EXPERIMENT_NAME = data["Azure"]["Azureml"]["Experiment"]["Name"]
        self.EXPERIMENT_PREP_NAME = data["Azure"]["Azureml"]["Experiments"]["DataPreparation"]["Name"]
        self.EXPERIMENT_DS_NAME = data["Azure"]["Azureml"]["Experiments"]["DataScience"]["Name"]
        #self.EXPERIMENT_SAMPLING_NAME = data["Azure"]["Azureml"]["Experiments"]["Sampling"]["Name"]
        
        #self.PIPELINE_NAME = data["Azure"]["Azureml"]["Pipeline"]["Name"]

        self.PIPELINE_PREP_NAME = data["Azure"]["Azureml"]["Pipelines"]["DataPreparation"]["Name"]
        self.PIPELINE_PREP_ENDPOINT = data["Azure"]["Azureml"]["Pipelines"]["DataPreparation"]["EndPoint"]
        self.PIPELINE_DS_NAME = data["Azure"]["Azureml"]["Pipelines"]["DataScience"]["Name"]
        self.PIPELINE_DS_ENDPOINT = data["Azure"]["Azureml"]["Pipelines"]["DataScience"]["EndPoint"]
        #self.PIPELINE_SAMPLING_NAME = data["Azure"]["Azureml"]["Pipelines"]["Sampling"]["Name"]
        #self.PIPELINE_SAMPLING_ENDPOINT = data["Azure"]["Azureml"]["Pipelines"]["Sampling"]["EndPoint"]
        #Model
        self.MODEL_NAME = data["Azure"]["Azureml"]["Model"]["Name"]
        #Deploy
        self.DEPLOY_SERVICE_NAME = data["Azure"]["Azureml"]["Deploy"]["ServiceName"]
        self.DEPLOY_THRESHOLD =  data["Azure"]["Azureml"]["Deploy"]["ModelThreshold"]

        #ServicePrincipalAuthentication
        self.SPA_TENANTID = data["Azure"]["ServicePrincipalAuthentication"]["TenantId"]
        self.SPA_APPLICATIONID = data["Azure"]["ServicePrincipalAuthentication"]["ApplicationId"]
        self.SPA_PASSWORD = data["Azure"]["ServicePrincipalAuthentication"]["Password"]

        #Subscriptions
        self.SUBSCRIPTION_VALUE = data["Azure"]["Subscriptions"]["Value"]
        #self.SUBSCRIPTION_ENTERPRISE = data["Azure"]["Subscriptions"]["Enterprise"]
        #self.SUBSCRIPTION_PROFESSIONAL = data["Azure"]["Subscriptions"]["Professional"]

class ConfigGenerator:
    def __init__(self, config_template_file):
        """initializing class with path to the template of config file

        Arguments:
            config_template_file {str} -- path that points to template of the config file
        """
        self.config_template_file = config_template_file

    def by_file(self, config_value_file, confile_fle):
        """[summary]

        Arguments:
            config_value_file {str} -- path that contains file where there are values to replace in the template
            confile_fle {str} -- new config file to be created
        """
        config_template = open(self.config_template_file, "rt").read()
        with open(config_value_file) as fp:
            for line in fp:
                arr = line.split(":")
                config_template = config_template.replace(arr[0].strip(),arr[1].strip())

        with open(confile_fle,'w') as f:
            f.write(config_template)        

    def _keys_from_template(self):
        """returns keys found in the template file ex: azure.amlcompute.clustername

        Returns:
            array -- list of retrieved keys
        """
        keys = []
        with open(self.config_template_file) as fp:
            for line in fp:
                if ":" not in line:
                    continue
                if "{{" in line:
                    arr = line.split(":")
                    key = arr[1].strip()
                    key = key.replace("{{","").replace("}}", "").replace(".", "_")
                    keys.append(key)
        
        return keys

    def _create_config_values(self,args):
        """creation of dictionary containing keys and values for conf file

        Arguments:
            args {array} -- which contains arguments passed in parameters

        Returns:
            {dict} -- contains information that will be in yaml config file.
        """
        config_values = {}
        last_value = 0
        for index, value in enumerate(args):
            if index == 0:
                continue
            if index % 2 != 0:
                last_value = value
            else:
                if value.isnumeric():
                    config_values[last_value] = value
                else:
                    config_values[last_value] = '"' + value + '"'
        
        return config_values

    def _valide_config_values(self,args):
        """ validate information format for new config file

        Arguments:
            args {dict} -- contains information that will be in yaml config file

        Raises:
            TypeError: [description]
            Exception: [description]
            Exception: [description]
        """

        for key in args.keys():

            if "-" not in key:
                raise TypeError("{0} argname must be preceded by '-'".format(key))
        
        key_templates = self._keys_from_template()

        if len(key_templates) != len(args.keys()):
            raise Exception("number of arguments are not sufficient ({0}). {1} are needed. Refer to the config.template.yaml file".format(len(args.keys()), len(key_templates)))
    
        for key in args.keys():
            key = key.replace("-","")
            if key not in key_templates:
                raise Exception("{0} is not a supported key. Refer to the config.template.yaml file".format(key))

    def by_args(self, args, confile_fle):
        """ Generates config file by the arguments passed in parameters 

        Arguments:
            args {array} -- arguments passed in parameters 
            confile_fle {[type]} -- new config file to be created
        """

        config_values = self._create_config_values(args)

        self._valide_config_values(config_values)

        config_template = open(self.config_template_file, "rt").read()

        for key,value in config_values.items():
            key = key.replace("_", ".").replace("-","")
            config_template = config_template.replace("{{" + key + "}}",  value)

        with open(confile_fle,'w') as f:
            f.write(config_template)

class ConfigHandler:
    def get_file(self,config_path):
        """Loading config object to be used in the entire program

        Arguments:
            config_path {str} -- generated config file path

        Returns:
            ConfigProvider -- Object containing all the information in the config file
        """
        config = ConfigProvider(config_path)
        config.load()
        return config

    def generate(self, config_template, config_path):
        configGen = ConfigGenerator(config_template)

        # If it run locally, then we're going to use the config.values.txt 
        # file containing all the necessary information
        if len(sys.argv) == 1:
            configGen.by_file("../../config.values.txt",config_path)
        # If it's launched via DevOps, then the values will be pass as params
        else:
            configGen.by_args(sys.argv, config_path)