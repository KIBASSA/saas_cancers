from azureml.core import Run
import glob
import shutil
import ntpath
import os
from global_helpers import ConfigHandler, BlobStorageHandler, PipelineEndpointLauncher, WorkspaceProvider
import argparse
import tempfile
from urllib.parse import urlparse
import requests

class DataUploader(object):
    def __init__(self, blob_manager, host):
        self.blob_manager = blob_manager
        self.host = host
    
    def upload_image(self, image_url_source, working_dir, blob_container_dest):
        parsed = urlparse(image_url_source)
        image_name = os.path.basename(parsed.path)
        image_path = os.path.join(working_dir, image_name)
        r = requests.get(image_url_source, allow_redirects=True)
        open(image_path, "wb").write(r.content)

        self.blob_manager.upload(blob_container_dest, image_path, overwrite = True)
        uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container_dest, image_name)
        print("uploaded_image :", uploaded_image)



class DataPreparator(object):
    def __init__(self, run, config):
        self.run = run
        self.config = config

    def prepare(self, input_data, prepped_data, data_uploader, pipeline_endpoint_launcher):
        annotated_data = []
        annotated_file = os.path.join(input_data, "annotated_data/current/annotated_data.json")
        if not os.path.isfile(annotated_file):
            self.run.tag("IGNORE_TRAIN_STEP", True)
            print("No annotation data provided")
            return
        
        with open(annotated_file, 'r') as myfile:
            annotated_data = json.loads(myfile.read())
        
        train_blob_container = "diagnoz/mldata/train"
        working_dir = os.path.join(prepped_data, "working_dir")
        os.makedirs(working_dir, exist_ok=True)
        for item in annotated_data:
            print(item["url"])
            if  item["hasCancer"] == False:
                data_uploader.upload_image(item["url"], working_dir, "{0}/0".format(train_blob_container))
            else:
                data_uploader.upload_image(item["url"], working_dir, "{0}/1".format(train_blob_container))

        #launch of the ds pipeline
        workspaceProvider = WorkspaceProvider(self.config)
        ws,svc_pr = workspaceProvider.get_ws()
        json_data = {"ExperimentName": self.config.EXPERIMENT_DS_NAME,
                                          "ParameterAssignments": {"mode": "execute"}}
        pipeline_endpoint_launcher.start(ws,svc_pr, self.config.PIPELINE_DS_ENDPOINT, json_data)

if __name__ == "__main__":
    run = Run.get_context()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, dest='input_data', help='data folder mounting point')
    parser.add_argument('--prepped_data', type=str, dest='prepped_data', help='data destination folder mounting point')
    parser.add_argument('--mode', type=str, dest="mode")
    args = parser.parse_args()

    input_data = args.input_data
    prepped_data_path = args.prepped_data
    mode = args.mode
    
    run.tag("MODE", mode)
    if mode == "execute":
        configHandler = ConfigHandler()
        config = configHandler.get_file("config.yaml")

        blob_manager = BlobStorageHandler()
        data_uploader = DataUploader(blob_manager, config.SERVICE_BLOB)
        data_peparator = DataPreparator(run, config)
        pipeline_endpoint_launcher = PipelineEndpointLauncher()
        data_peparator.prepare(input_data,prepped_data_path, data_uploader, pipeline_endpoint_launcher)
    else:
        print("the mode has value '{0}' so no need to execute data preparation step".format(mode))