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
import random
import string
import urllib
import urllib.request
import base64
import shutil
import tempfile
import json
import traceback
import sys
from datetime import datetime
import requests
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse

class FolderUtilities:
    @staticmethod
    def make_missing_dir_from_file(file_path):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    @staticmethod 
    def make_dir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

class BlobStorageHandler(object):
    def __init__(self, connection_string="DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net"):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    def upload(self, blob_container, file_path, overwrite = False):
        file_name = ntpath.basename(file_path)
        print("file_name :",file_name)
        blob_client=self.blob_service_client.get_blob_client(container=blob_container,blob=file_name)
        with open(file_path,"rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)
    
    def delete(self,blob_container,  file_path):
        """deletes the file from the blob storage

        Arguments:
            blob_container {str} -- container (or folder) in blob storage 
            file_path {str} -- [description]
        """

        if os.path.isabs(file_path):
            file_name = ntpath.basename(file_path)
        else:
            file_name = file_path
        # Create a blob client using the local file name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=blob_container, blob=file_name)

        print("\Deleting from Azure Storage:\n\t" + file_name)

        # Upload the created file
        blob_client.delete_blob()
    
    def download_json_by_url(self, file_path):
        data = {}
        try:
            response = urllib.request.urlopen(file_path)
            data = json.loads(response.read())
        except HTTPError as e:
            # do something
            print('Error code: ', e.code)
        except URLError as e:
            # do something
            print('Reason: ', e.reason)

        return data
    
    def download_text_by_url(self, file_path):
        """download the file from the http link

        Arguments:
            file_path {str} -- http link

        Returns:
            str -- downloaded file content
        """
        data = None
        try:
            response = urllib.request.urlopen(file_path)
            data = response.read().decode('utf-8')
        except HTTPError as e:
            # do something
            print('Error code: ', e.code)
        except URLError as e:
            # do something
            print('Reason: ', e.reason)
        except Exception as e:
            # do something
            print('Reason: ', e)

        return data

class Helper:
    @staticmethod
    def get_base64_image_by_urls(image_urls):
        images_result = {}
        for image_url in image_urls:
            images_result[image_url] = Helper.get_base64_image_by_url(image_url)
        return images_result
    @staticmethod
    def get_base64_image_by_url(image_url):
        #NamedTemporaryFile  has not been used because of a problem with access rights.
        temp_file = os.path.join(os.getcwd(), Helper.generate_name() + "/" + Helper.generate_name())
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        urllib.request.urlretrieve(image_url, temp_file)
        with open(temp_file, 'rb') as read_file:
                image_string = base64.b64encode(read_file.read())
        shutil.rmtree(os.path.dirname(temp_file))
        return image_string.decode("utf-8")

    @staticmethod
    def generate(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
    
    @staticmethod
    def generate_name(size=6, chars=string.ascii_uppercase):
        return ''.join(random.choice(chars) for _ in range(size))

class ImagePathListUploader(object):
    def __init__(self, blob_manager, host):
        self.blob_manager = blob_manager
        self.host = host
    
    def upload(self,files_source, blob_container_dest):
        for file_source in files_source:
            file_source_name = ntpath.basename(file_source)
            self.blob_manager.upload(blob_container_dest, file_source, overwrite = True)
            uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container_dest, file_source_name)
            print("uploaded_image :", uploaded_image)

class SampedDataDataManager:
    def __init__(self, blob_manager, host):
        self.blob_manager = blob_manager
        self.host = host
    
    def _get_http_sampled_data(self, sampled_data):
        items = []
        for item in sampled_data:
            print("item[0]", item[0])
            print("item[1]", item[1])
            image_name = os.path.basename(item[0])
            url_file =  "{0}/diagnoz/mldata/unlabeled/data/{1}".format(self.host, image_name)
            items.append(url_file)
        return items

    def upload_data(self, sampled_data):
        sampled_data = self._get_http_sampled_data(sampled_data)
        with tempfile.TemporaryDirectory() as dir:
            sampled_data_file = os.path.join(dir,'sampled_data.json')
            with open(sampled_data_file, 'w') as outfile:
                json.dump(sampled_data, outfile)
            blob_container = "diagnoz/mldata/sampled_data/current"
            self.blob_manager.upload(blob_container, sampled_data_file, overwrite = True)
            print("file {0} uploaded".format(sampled_data_file))

class AnnotatedDataManager:
    def __init__(self, blob_manager, host):
        self.blob_manager = blob_manager
        self.host = host
        self.annotated_data_file = "annotated_data.json"
        self.annotated_data_url = "{0}/diagnoz/mldata/annotated_data/current/{1}".format(self.host,self.annotated_data_file)
        self.ARCHIVE_ANNOTATED_PATH = "diagnoz/mldata/annotated_data/archive"
        self.CURRENT_ANNOTATED_PATH = "diagnoz/mldata/annotated_data/current"

    def archive(self, working_dir):
        try:
            print("step 1")
            annotated_data_path = os.path.join(working_dir, self.annotated_data_file)
            r = requests.get(self.annotated_data_url, allow_redirects=True)
            open(annotated_data_path, "wb").write(r.content)
            print("step 2")
            archive_name = "annotated_data_{0}{1}".format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"),".json")
            archived_file = os.path.join(working_dir, archive_name)
            #copy sampled file as archived file
            shutil.copyfile(annotated_data_path, archived_file)
            print("step 3")
            #upload sampled file to blob
            self.blob_manager.upload(self.ARCHIVE_ANNOTATED_PATH, archived_file)
            #upload sampled file to blob
            self.blob_manager.delete(self.CURRENT_ANNOTATED_PATH, annotated_data_path)
            #delete sampled file
            os.remove(archived_file)
            print("step 4")
        except Exception as e:
            print(traceback.format_exc())
            print(sys.exc_info()[2])
            raise Exception(e)

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
        try:
            AciWebservice(self.ws, self.config.DEPLOY_SERVICE_NAME).delete()
            print("webservice deleted")
        except WebserviceException:
            pass

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

        aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                    memory_gb=1, 
                                                    tags={"data": "cancer-data",  "method" : "tensorflow"}, 
                                                    description='Predicting cancer with tensorflow')

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
        self.AML_COMPUTE_BM_CLUSTER_NAME = data["Azure"]["AmlComputes"]["Benchmark"]["ClusterName"]
        self.AML_COMPUTE_BM_CLUSTER_VM_TYPE = data["Azure"]["AmlComputes"]["Benchmark"]["ClusterType"]
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
        self.CONTAINER_HUML_NAME = data["Azure"]["StorageAccount"]["ContainerNameHuml"]
        self.ACCOUNT_KEY = data["Azure"]["StorageAccount"]["AccountKey"]
        self.BLOB_STORAGE_CONNECTION_STRING = data["Azure"]["StorageAccount"]["BlobStorageConnectionString"]
        self.SERVICE_BLOB = data["Azure"]["StorageAccount"]["ServiceBlob"]
        #Azureml
        self.LOCATION = data["Azure"]["Azureml"]["Location"]
        self.RESOURCEGROUP = data["Azure"]["Azureml"]["ResourceGroup"]
        self.WORKSPACENAME = data["Azure"]["Azureml"]["WorkspaceName"]


        #ExperimentName
        self.EXPERIMENT_BM_NAME = data["Azure"]["Azureml"]["Experiments"]["Benchmark"]["Name"]
        self.EXPERIMENT_PREP_NAME = data["Azure"]["Azureml"]["Experiments"]["DataPreparation"]["Name"]
        self.EXPERIMENT_DS_NAME = data["Azure"]["Azureml"]["Experiments"]["DataScience"]["Name"]
        #self.EXPERIMENT_SAMPLING_NAME = data["Azure"]["Azureml"]["Experiments"]["Sampling"]["Name"]
        
        #self.PIPELINE_NAME = data["Azure"]["Azureml"]["Pipeline"]["Name"]
        self.PIPELINE_BM_NAME = data["Azure"]["Azureml"]["Pipelines"]["Benchmark"]["Name"]
        self.PIPELINE_BM_ENDPOINT = data["Azure"]["Azureml"]["Pipelines"]["Benchmark"]["EndPoint"]
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

class PipelineEndpointLauncher:
    def start(self, ws, svc_pr, endpoint_pipeline, json_data):
        """The PIPELINE_ENDPOINT pipeline is launched through its published REST address.
        """
        #workspaceProvider = WorkspaceProvider(self.config)
        #ws,svc_pr = workspaceProvider.get_ws()
        auth_header = svc_pr.get_authentication_header()
        pipeline_endpoint_by_name = PipelineEndpoint.get(workspace=ws, name=endpoint_pipeline)
        rest_endpoint = pipeline_endpoint_by_name.endpoint
        print(datetime.now(), " - pipeline_endpoint_by_name.endpoint : ", rest_endpoint)
        _ = requests.post(rest_endpoint, headers=auth_header, json=json_data)

class PipelinePublisher:
    def __init__(self, ws):
        self.ws = ws
    
    def publish(self, experiment_name,pipeline_name):
        """Publish an experiment from name and the last Run that has just performed

        Arguments:
            experiment_name {str} -- The name of the experiment in the workspace...
            pipeline_name {str} --The name of the published pipeline

        Returns:
            str --pipeline id published, str --endpoint : http address through which the pipeline can be called
        """
        
        pipeline_experiment = self.ws.experiments.get(experiment_name)
        run = list(pipeline_experiment.get_runs())[0]
       
        published_pipeline = run.publish_pipeline(name = pipeline_name, description='pipelines', version="2.1")

        #return published_pipeline.id, published_pipeline.endpoint
        return published_pipeline


class EndpointPipelinePublisher:
    def __init__(self, ws):
        self.ws = ws
    
    def publish(self, experiment_name,pipeline, pipeline_name,pipeline_endpoint_name):
        pipeline_endpoint = None
        try:
            pipeline_endpoint = PipelineEndpoint.get(workspace=self.ws, name=pipeline_endpoint_name)
        except Exception as e:
            s = str(e)
            if not "BadRequest" in s:
                raise Exception(e)

        if not pipeline_endpoint:
            pipeline_endpoint = PipelineEndpoint.publish(workspace=self.ws,
                                                        name=pipeline_endpoint_name,
                                                        pipeline=pipeline,
                                                        description="New Pipeline Endpoint for {0}".format(pipeline_endpoint_name))
            published_endpoint = pipeline_endpoint.endpoint
            
        else:
            publisher = PipelinePublisher(self.ws)
            published_pipeline = publisher.publish(experiment_name, pipeline_name)
            
            pipeline_endpoint.add_default(published_pipeline)
            published_endpoint = published_pipeline.endpoint
            
        return published_endpoint

class LogicAppPipelineConfigManager:
    def __init__(self, config):
        """Initializing class with the config object.
            
        Arguments:
            config {object} -- Object containing all the information in the yaml configuration file
        """
        # BlobStorageHandler object is created from the connectionstring coming from config object.
        self.blobManager = BlobStorageHandler(config.BLOB_STORAGE_CONNECTION_STRING)
        # declaration of container (or folder) which will contain the json file
        self.BLOB_CONTAINER = "{0}/configs".format(config.CONTAINER_NAME)
        self.host = config.SERVICE_BLOB
    
    def update(self, pipelineid, pipeline_endpoint, pipeline_file_name):
        """ Create or update json file of the pipeline in the blob storage

        Arguments:
            pipelineid {str} -- pipeline id published
            pipeline_endpoint {str} --endpoint : http address through which the pipeline can be called
        """
        
        file_url = "{0}/{1}/{2}".format(self.host, self.BLOB_CONTAINER, pipeline_file_name)

        data = self.blobManager.download_json_by_url(file_url)

        data["pipelineid"] = pipelineid
        data["published_pipeline_endpoint"] = pipeline_endpoint

        temp = "./temp"
        FolderUtilities.make_dir(temp)

        temp_file = os.path.join(temp, pipeline_file_name)

        with open(temp_file, 'w') as outfile:
            json.dump(data, outfile)

        print("update remote file")
        self.blobManager.upload(self.BLOB_CONTAINER, temp_file, overwrite = True)

        shutil.rmtree(temp_file, ignore_errors=True)
