from azureml.core import Run
import glob
import shutil
import ntpath
import os
class DataMerger(object):
    def __init__(self, run):
        self.run = Run
    
    def merge(self,annotated_set_folder,  train_set_folder):
        files = glob.glob(annotated_set_folder + '/*.png')
        for file_source in files:
            source_file_name = ntpath.basename(file_source)
            file_dest = os.path.join(train_set_folder, "0/{0}".format(source_file_name))
            shutil.copyfile(file_source, file_dest)
            print(file_dest, " copied")

class DataUploader(object):
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
        self.host = "https://diagnozstorage.blob.core.windows.net/"
    
    def upload(self,folder_source, blob_container_dest):
        with tempfile.TemporaryDirectory() as dir:
            files_source = glob.glob(folder_source + '/*.png')
            for file_source in files_source:
                file_source_name = ntpath.basename(file_source)
                self.blob_manager.upload(blob_container_dest, file_source, overwrite_v = True)
                uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container_dest, file_source_name)
                print("uploaded_image :", uploaded_image)


class DataPreparator(object):
    def __init__(self, run, config):
        self.run = run 
        self.config = config
    
    def _merge(self, data_merger, annotated_folder,train_folder, label):
        annotated_folder = os.path.join(annotated_folder, label)
        if not os.path.isdir(annotated_folder):
            raise Exception("label {0} for annotated data not provided".format(label))
        train_folder = os.path.join(train_folder, label)
        data_merger.merge(annotated_folder, train_folder)
    
    def _upload(self, data_uploader, annotated_folder,train_container, label):
        annotated_folder = os.path.join(annotated_folder, label)
        if not os.path.isdir(annotated_folder):
            raise Exception("label {0} for annotated data not provided".format(label))
        train_container = "{0}/{1}".format(train_container, label)
        data_uploader.upload(annotated_folder, train_container)

    def prepare(input_data, prepped_data, data_merger, data_uploader):

        self.run.tag("IGNORE_TRAIN_STEP", False)

        annotated_file_folder = os.path.join(input_data, "diagnoz/mldata/annotated_data/current")
        if not os.path.isdir(annotated_file_folder):
            self.run.tag("IGNORE_TRAIN_STEP", True)
            print("No annotation data provided")
            return

        eval_read_folder = os.path.join(input_data, "diagnoz/mldata/eval_data")
        if not os.path.isdir(eval_read_folder):
            eval_write_folder = os.path.join(prepped_data, "diagnoz/mldata/eval_data")
            self._merge(annotated_file_folder,eval_write_folder, "0")
            self._merge(annotated_file_folder,eval_write_folder, "1")

            eval_blob_container = "diagnoz/mldata/eval_data"
            self._upload(data_uploader, annotated_file_folder, eval_blob_container,"0")
            self._upload(data_uploader, annotated_file_folder, eval_blob_container,"1")
            self.run.tag("IGNORE_TRAIN_STEP", True)
            print("There is no training data available.")
            return

        train_folder = os.path.join(prepped_data, "diagnoz/mldata/train_data")
        self._merge(annotated_file_folder,train_folder, "0")
        self._merge(annotated_file_folder,train_folder, "1")

        #upload to the cloud
        train_blob_container = "diagnoz/mldata/train_data"
        self._upload(data_uploader, annotated_file_folder, train_blob_container,"0")
        self._upload(data_uploader, annotated_file_folder, train_blob_container,"1")


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

        preparator = DataPreparator(run, config)
        preparator.prepare(input_data,prepped_data_path)
    else:
        print("the mode has value '{0}' so no need to execute data preparation step".format(mode))