
import os
import sys
import glob
from distutils.dir_util import copy_tree
import shutil
script_path = os.path.join(os.path.dirname(__file__), "../../src/scripts")
sys.path.append(script_path)
from prep_data import DataMerger, DataPreparator

"""|||||||||| DataMerger ||||||||||
Testing the useful functions of the DataMerger class.
"""

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class BlobManagerMoq(object):
    def upload(self, blob_container, file_path, overwrite = False):
        print("upload")
    
class DataUploaderMoq(object):
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager

    def upload(self,folder_source, blob_container_dest):
        print("upload")

class AzureExperimentMoq:
    def __init__(self):
        self.runs = []

    def get_runs(self):
        return self.runs

class AzureMLRunMoq:
    def __init__(self):

        self.children = []
        self.json_data = {}
        self.tags = {}
        self.metrics = {}

    def get_children(self):
        return self.children
    
    def get_details(self):
        return self.json_data
    
    def get_tags(self):
        return self.tags
    
    def get_metrics(self):
        return self.metrics
    
    def tag(self, title, value):
        print("title :", title)
        print("value :", value)

    def log(self, title, value):
        print("title :", title)
        print("value :", value)

    def log_image(self, title, plot):
        print("tile:", title)
        print("plot :", plot)
        id = Helper.generate()
        if self.temp_dir:
            plot.savefig(os.path.join(self.temp_dir,"{0}_{1}.png".format(title, id)))
    
    def upload_file(self, name, path_or_stream):
        print("name :", name)
        print("path_or_stream :", path_or_stream)
    
    def set_temp_dir(self, temp_dir):
        self.temp_dir = temp_dir

def test_if_method___merge__is_running_properly():
    annotated_folder = os.path.join(os.path.dirname(__file__), "data/data_merger/merge/annotated")
    train_folder = os.path.join(os.path.dirname(__file__), "data/data_merger/merge/train")
    train_result_folder = os.path.join(os.path.dirname(__file__), "data/data_merger/merge/train_result")
    
    clean_dir(train_result_folder)
    copy_tree(train_folder, train_result_folder)

    data_merger = DataMerger()

    annotated_folder_label_0 = os.path.join(annotated_folder, "0")
    train_folder_label_0 = os.path.join(train_result_folder, "0")
    data_merger.merge(annotated_folder_label_0, train_folder_label_0)
    
    train_label_0_files = glob.glob(train_folder_label_0 + '/*.png')
    assert len(train_label_0_files) == 9

    annotated_folder_label_1 = os.path.join(annotated_folder, "1")
    train_folder_label_1 = os.path.join(train_result_folder, "1")
    data_merger.merge(annotated_folder_label_1, train_folder_label_1)

    train_folder_label_1 = glob.glob(train_folder_label_1 + '/*.png')
    assert len(train_folder_label_1) == 9

"""|||||||||| DataPreparator ||||||||||
Testing the useful functions of the DataPreparator class.
"""

def test_if_method___prepare__to_eval_folder____is_running_properly():
    #input_data
    input_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_eval_folder")
    prepped_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_eval_folder")
    eval_folder = os.path.join(prepped_data,"diagnoz/mldata/eval_data")
    if os.path.isdir(eval_folder):
        clean_dir(eval_folder)
    shutil.rmtree(eval_folder, ignore_errors=True)
    
    data_merger = DataMerger()
    blob_manager = BlobManagerMoq()
    data_uploader = DataUploaderMoq(blob_manager)
    run = AzureMLRunMoq()
    data_peparator = DataPreparator(run)
    data_peparator.prepare(input_data,prepped_data,data_merger,data_uploader)

    eval_folder_label_0 = os.path.join(eval_folder,"0")
    eval_label_0_files = glob.glob(eval_folder_label_0 + '/*.png')
    assert len(eval_label_0_files) == 7
    eval_folder_label_1 = os.path.join(eval_folder,"1")
    eval_label_1_files = glob.glob(eval_folder_label_1 + '/*.png')
    assert len(eval_label_1_files) == 7

def test_if_method___prepare__to_train_folder____is_running_properly():
    #input_data
    input_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_train_folder")
    prepped_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_train_folder")
    train_folder = os.path.join(prepped_data,"diagnoz/mldata/train_data")
    if os.path.isdir(train_folder):
        clean_dir(train_folder)
    shutil.rmtree(train_folder, ignore_errors=True)
    
    data_merger = DataMerger()
    blob_manager = BlobManagerMoq()
    data_uploader = DataUploaderMoq(blob_manager)
    run = AzureMLRunMoq()
    data_peparator = DataPreparator(run)
    data_peparator.prepare(input_data,prepped_data,data_merger,data_uploader)

    train_folder_label_0 = os.path.join(train_folder,"0")
    train_label_0_files = glob.glob(train_folder_label_0 + '/*.png')
    assert len(train_label_0_files) == 7
    train_folder_label_1 = os.path.join(train_folder,"1")
    train_label_1_files = glob.glob(train_folder_label_1 + '/*.png')
    assert len(train_label_1_files) == 7

#if __name__ == "__main__":
#    test_if_method___prepare__to_train_folder____is_running_properly()